
'''
    source: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py

'''
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

from copy import deepcopy
from pathlib import Path

from Bio.PDB import PDBParser, Polypeptide


'''
    https://github.com/HySonLab/Protein_Pretrain/blob/main/downstreamtasks/Utils.py#L72
'''
from sklearn.preprocessing import LabelEncoder

# This is same as the multimodal-preteaining paper, so that we can use this one hot encoding as input to their pretrained VGAE

amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
label_encoder = LabelEncoder()
label_encoder.fit(list(amino_acids))
num_amino_acids = len(amino_acids)

def one_hot_encode_amino_acid(sequence = None, amino_acid_indices=None):
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot


from functools import lru_cache


'''
https://graphein.ai/_modules/graphein/protein/features/nodes/amino_acid.html
'''

@lru_cache()
def load_expasy_scales() -> pd.DataFrame:
    """
    Load pre-downloaded EXPASY scales.

    This helps with node featuarization.

    The function is LRU-cached in memory for fast access
    on each function call.

    :returns: pd.DataFrame containing expasy scales
    :rtype: pd.DataFrame
    """
    fpath = Path(__file__).parent / "aa_prop_files_from_graphein/amino_acid_properties.csv"
    return pd.read_csv(fpath, index_col=0)

@lru_cache()
def load_meiler_embeddings() -> pd.DataFrame:
    """
    Load pre-downloaded Meiler embeddings.

    This helps with node featurization.

    The function is LRU-cached in memory for fast access
    on each function call.

    :returns: pd.DataFrame containing Meiler Embeddings from Meiler et al. 2001
    :rtype: pd.DataFrame
    """
    fpath = Path(__file__).parent / "aa_prop_files_from_graphein/meiler_embeddings.csv"
    return pd.read_csv(fpath, index_col=0)



# meiler_df = load_meiler_embeddings()
# expasy_df = load_expasy_scales()

def expasy_scale_amino_acid(sequence):

    expasy_df = load_expasy_scales()
    node_embedding = []
    for x in sequence:
        try:
            node_embedding.append(np.array(expasy_df[Polypeptide.index_to_three(Polypeptide.one_to_index(x))]))
        except:
            raise ValueError(f"embedding not found for AA: {x}")

    return np.array(node_embedding)

def meiler_amino_acid(sequence):

    meiler_df = load_meiler_embeddings()
    node_embedding = []
    for x in sequence:
        try:
            node_embedding.append(np.array(meiler_df[Polypeptide.index_to_three(Polypeptide.one_to_index(x))]))
        except:
            raise ValueError(f"embedding not found for AA: {x}")
    return np.array(node_embedding)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF



class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, tokenizer, max_len_H, max_len_L,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, kNN_radius = -1, min_k=1, device="cpu", empty_graph = False,other_features = [], concat_AbLang2 = False, ):
        
        super(ProteinGraphDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.is_ablang = 'ablang' in str(type(tokenizer))
        self.ablang_version = int(str(type(tokenizer)).split('.')[-3][-1]) if self.is_ablang else 0
        self.concat_AbLang2 = concat_AbLang2
        self.max_len = [ max_len_H, max_len_L]
        self.other_features = other_features

        self.data_list = data_list
        self.processed_data_list = [None]*len(data_list)
        self.top_k = top_k
        self.min_k = min_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.kNN_radius = kNN_radius 
        # need to be specified, preferably one value for all datapoints, 
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204101 tries 5 to 15 angstrom, uses 9. (for different problem)
        self.empty_graph = empty_graph
        self.node_counts = [len(e['seq']) for e in data_list]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        print('making the sample ready in torch_geometric data format')
        self._preprocess_data()
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): 

        return self.processed_data_list[i] 
    
    def _preprocess_data(self):

        seq_dict = None

        if self.is_ablang:
            if self.ablang_version == 2:
                # for AbLang-2
                if self.concat_AbLang2:
                    input_ids_1 = self.tokenizer([[x['VH_seq'],x['VL_seq']] for x in self.data_list], pad=True, w_extra_tkns=True, device = 'cpu')
                    input_ids_2 = torch.zeros(*input_ids_1.shape, device='cpu')
                else:
                    input_ids_1 = self.tokenizer([x['VH_seq'] for x in self.data_list], pad=True, w_extra_tkns=False, device = 'cpu')
                    input_ids_2 = self.tokenizer([x['VL_seq'] for x in self.data_list], pad=True, w_extra_tkns=False, device = 'cpu')
            elif self.ablang_version == 1:
                # for AbLang-1
                input_ids_1 = self.tokenizer([x['VH_seq'] for x in self.data_list], pad=True, encode=True, device = 'cpu')
                input_ids_2 = self.tokenizer([x['VL_seq'] for x in self.data_list], pad=True, encode=True, device = 'cpu')



        for i in range(len(self.data_list)):

            if self.is_ablang:

                seq_dict = {
                  'input_ids': [input_ids_1[i], input_ids_2[i]],
                  'attention_mask': [torch.zeros(*input_ids_1[i].shape, device='cpu').masked_fill(input_ids_1[i]==21,1),
                                     torch.zeros(*input_ids_2[i].shape, device='cpu').masked_fill(input_ids_2[i]==21,1)],
                }

            if self.concat_AbLang2:
                # attention mask is not fed to the forward call of AbLang2. 
                # the forward function for AbLang2 creates attention mask internally, here this mask is used 
                # for picking up the residue embeddings
                seq_dict['attention_mask'][0] = seq_dict['attention_mask'][0].masked_fill(
                                        (input_ids_1[i]==0)+(input_ids_1[i]==22)+(input_ids_1[i]==25),1)

                
            # build the graph (x, y) tuple
            graph = self._featurize_as_graph(self.data_list[i], seq_dict)
            label = torch.tensor(self.data_list[i]["target"], dtype=torch.float32)

            # now check for zero edges
            # graph.edge_index is (2, n_edges)
            if graph.edge_index.numel() == 0 or graph.edge_index.size(1) == 0:
                print(
                    f"[Empty‐graph] sample #{i}: "
                    f"VH_seq={self.data_list[i]['VH_seq']}, "
                    f"VL_seq={self.data_list[i]['VL_seq']}, "
                    f"target={self.data_list[i]['target']}"
                )

            # finally stash it
            self.processed_data_list[i] = (graph, label)




    def _featurize_as_graph(self, protein, seq_dict = None, debug = False):


        '''
        process the sequence information
        '''


        if self.is_ablang:
            assert seq_dict is not None

        else:

            seq_dict = {
                  'input_ids': [],
                  'attention_mask': [],
                }

            for sequence,max_len in zip([protein['VH_seq'],protein['VL_seq'] ], self.max_len):
                encoding = self.tokenizer.encode_plus(
                  sequence,
                  add_special_tokens=True,
                  max_length=max_len +2, # should be max_len+2, two extra tokens at beginning and end
                  return_token_type_ids=False, 
                  padding= "max_length",
                  return_attention_mask=True,truncation = True,
                  return_tensors='pt',
                )
                
                seq_dict['input_ids'].append(encoding['input_ids'].flatten())
                seq_dict['attention_mask'].append(encoding['attention_mask'].flatten())



        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'],
                device=self.device, dtype=torch.float32)
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]         # (n_nodes, 3)
            n_nodes = X_ca.size(0)
            device = X_ca.device
            edges_src = []
            edges_dst = []

            # --------- Updated neighborhood code ---------
            for i in range(n_nodes):
                diffs = X_ca - X_ca[i]
                dists = diffs.pow(2).sum(-1).sqrt()
                dists[i] = float('inf')  # Exclude self

                # All nodes within kNN_radius
                in_radius = (dists <= self.kNN_radius).nonzero(as_tuple=True)[0]
                if in_radius.numel() > 0:
                    sorted_indices = in_radius[torch.argsort(dists[in_radius])]
                else:
                    sorted_indices = torch.LongTensor([], device=dists.device)

                # Keep at most top_k
                chosen = sorted_indices[:self.top_k]

                # If less than min_k, fill up with closest (even if outside radius)
                if chosen.numel() < self.min_k:
                    num_to_add = self.min_k - chosen.numel()
                    sorted_all = torch.argsort(dists)
                    extras = []
                    for idx in sorted_all:
                        if idx not in chosen and idx != i:
                            extras.append(idx.item())
                        if len(extras) >= num_to_add:
                            break
                    if len(extras) > 0:
                        extras_tensor = torch.tensor(extras, device=dists.device)
                        chosen = torch.cat([chosen, extras_tensor])

                for j in chosen:
                    edges_src.append(i)
                    edges_dst.append(j.item())
            # RIGHT HERE:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)

            # --- MAKE GRAPH UNDIRECTED ---
            edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
            # --- END UNDIRECTED ---
            # --------- End neighborhood code ---------

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))

            if 'ohe' in self.other_features:
                ohe_features = one_hot_encode_amino_acid(protein['seq'])
                ohe_features = torch.tensor(ohe_features, dtype=torch.float32)
            else:
                ohe_features = torch.empty((0, 0))

            if 'expasy' in self.other_features:
                expasy_features = expasy_scale_amino_acid(protein['seq'])
                expasy_features = torch.tensor(expasy_features, dtype=torch.float32)
            else:
                expasy_features = torch.empty((0, 0))

            if 'meiler' in self.other_features:
                meiler_features = meiler_amino_acid(protein['seq'])
                meiler_features = torch.tensor(meiler_features, dtype=torch.float32)
            else:
                meiler_features = torch.empty((0, 0))

            data = torch_geometric.data.Data(
                x=X_ca,
                input_ids = seq_dict['input_ids'],
                attention_mask = seq_dict['attention_mask'],
                node_s=node_s, node_v=node_v,
                ohe_features = ohe_features, expasy_features = expasy_features,
                meiler_features = meiler_features,
                edge_s=edge_s, edge_v=edge_v,
                edge_index=edge_index, mask=mask
            )
            if self.empty_graph:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
            return data               
            
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 





class ProteinGraphDataset_v2(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, tokenizer, max_len_H, max_len_L,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, kNN_radius = -1, min_k=1, device="cpu", empty_graph = False,other_features = [], concat_AbLang2 = False, VHH_data = False):
        
        super(ProteinGraphDataset_v2, self).__init__()
        
        self.VHH_data = VHH_data
        self.tokenizer = tokenizer
        self.is_ablang = 'ablang' in str(type(tokenizer))
        self.ablang_version = int(str(type(tokenizer)).split('.')[-3][-1]) if self.is_ablang else 0
        self.concat_AbLang2 = concat_AbLang2
        self.max_len = [ max_len_H, max_len_L]
        self.other_features = other_features

        self.data_list = data_list
        self.processed_data_list = [None]*len(data_list)
        self.top_k = top_k
        self.min_k = min_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.kNN_radius = kNN_radius
        # need to be specified, preferably one value for all datapoints, 
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204101 tries 5 to 15 angstrom, uses 9. (for different problem)
        self.empty_graph = empty_graph 
        self.node_counts = [len(e['seq']) for e in data_list]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        print('making the sample ready in torch_geometric data format')
        self._preprocess_data()
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): 

        # if processed_data_list[i] is None:
        #     self.processed_data_list[i] = (self._featurize_as_graph(self.data_list[i]), self.data_list[i]["target"])

        return self.processed_data_list[i] 
    
    def _preprocess_data(self):

        seq_dict = None

        if self.is_ablang:
            
            if self.ablang_version == 2:
                # for AbLang-2
                if self.concat_AbLang2:
                    input_ids_1 = self.tokenizer([[x['VH_seq'],x['VL_seq']] for x in self.data_list], pad=True, w_extra_tkns=True, device = 'cpu')
                    input_ids_2 = torch.zeros(*input_ids_1.shape, device='cpu')
                else:
                    input_ids_1 = self.tokenizer([x['VH_seq'] for x in self.data_list], pad=True, w_extra_tkns=False, device = 'cpu')
                    input_ids_2 = torch.zeros(*input_ids_1.shape, dtype=torch.long, device='cpu')

            elif self.ablang_version == 1:
                # for AbLang-1
                input_ids_1 = self.tokenizer([x['VH_seq'] for x in self.data_list], pad=True, encode=True, device = 'cpu')
                input_ids_2 = torch.zeros(*input_ids_1.shape, dtype=torch.long, device='cpu')


        for i in range(len(self.data_list)):

            if self.is_ablang:

                seq_dict = {
                  'input_ids': [input_ids_1[i], input_ids_2[i]],
                  'attention_mask': [torch.zeros(*input_ids_1[i].shape, device='cpu').masked_fill(input_ids_1[i]==21,1),
                                     torch.zeros(*input_ids_2[i].shape, device='cpu').masked_fill(input_ids_2[i]==21,1)],
                }

            if self.concat_AbLang2:
                # attention mask is not fed to the forward call of AbLang2. 
                # the forward function for AbLang2 creates attention mask internally, here this mask is used 
                # for picking up the residue embeddings
                seq_dict['attention_mask'][0] = seq_dict['attention_mask'][0].masked_fill(
                                        (input_ids_1[i]==0)+(input_ids_1[i]==22)+(input_ids_1[i]==25),1)

                
            # self.processed_data_list[i] = (self._featurize_as_graph(self.data_list[i], seq_dict),
            #     torch.tensor(self.data_list[i]["target"], dtype=torch.float32))
            # build the graph (x, y) tuple
            graph = self._featurize_as_graph(self.data_list[i], seq_dict)
            label = torch.tensor(self.data_list[i]["target"], dtype=torch.float32)

            # now check for zero edges
            # graph.edge_index is (2, n_edges)
            if graph.edge_index.numel() == 0 or graph.edge_index.size(1) == 0:
                print(
                    f"[Empty‐graph] sample #{i}: "
                    f"VH_seq={self.data_list[i]['VH_seq']}, "
                    f"VL_seq={self.data_list[i]['VL_seq']}, "
                    f"target={self.data_list[i]['target']}"
                )

            # finally stash it
            self.processed_data_list[i] = (graph, label)


    def _featurize_as_graph(self, protein, seq_dict = None, debug = False):


        '''
        process the sequence information
        '''


        if self.is_ablang:
            assert seq_dict is not None

        else:

            seq_dict = {
                  'input_ids': [],
                  'attention_mask': [],
                }

            for sequence,max_len in zip([protein['VH_seq'],protein['VL_seq'] ], self.max_len):
                if max_len != 0: 
                    encoding = self.tokenizer.encode_plus(
                      sequence,
                      add_special_tokens=True,
                      max_length=max_len +2, # should be max_len+2, two extra tokens at beginning and end
                      return_token_type_ids=False, 
                      padding= "max_length",
                      return_attention_mask=True,truncation = True,
                      return_tensors='pt',
                    )
                else:
                    '''
                        Warning: Make sure this empty encoding is not used in training, 
                        the prediction network needs to have a flag to indicate to use only the first input_ids like the concat Ablang2
                    '''
                    encoding = {'input_ids': torch.zeros(*seq_dict['input_ids'][-1].shape, dtype=torch.long, device='cpu'),
            'attention_mask': torch.zeros(*seq_dict['attention_mask'][-1].shape, dtype=torch.long, device='cpu')}


                seq_dict['input_ids'].append(encoding['input_ids'].flatten())
                seq_dict['attention_mask'].append(encoding['attention_mask'].flatten())



        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'],
                device=self.device, dtype=torch.float32)
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]         # (n_nodes, 3)
            n_nodes = X_ca.size(0)
            device = X_ca.device
            edges_src = []
            edges_dst = []

            # --------- Updated neighborhood code ---------
            for i in range(n_nodes):
                diffs = X_ca - X_ca[i]
                dists = diffs.pow(2).sum(-1).sqrt()
                dists[i] = float('inf')  # Exclude self

                # All nodes within kNN_radius
                in_radius = (dists <= self.kNN_radius).nonzero(as_tuple=True)[0]
                if in_radius.numel() > 0:
                    sorted_indices = in_radius[torch.argsort(dists[in_radius])]
                else:
                    sorted_indices = torch.LongTensor([], device=dists.device)

                # Keep at most top_k
                chosen = sorted_indices[:self.top_k]

                # If less than min_k, fill up with closest (even if outside radius)
                if chosen.numel() < self.min_k:
                    num_to_add = self.min_k - chosen.numel()
                    sorted_all = torch.argsort(dists)
                    extras = []
                    for idx in sorted_all:
                        if idx not in chosen and idx != i:
                            extras.append(idx.item())
                        if len(extras) >= num_to_add:
                            break
                    if len(extras) > 0:
                        extras_tensor = torch.tensor(extras, device=dists.device)
                        chosen = torch.cat([chosen, extras_tensor])

                for j in chosen:
                    edges_src.append(i)
                    edges_dst.append(j.item())
            # RIGHT HERE:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)

            # --- MAKE GRAPH UNDIRECTED ---
            edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
            # --- END UNDIRECTED ---
            # --------- End neighborhood code ---------

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))

            if 'ohe' in self.other_features:
                ohe_features = one_hot_encode_amino_acid(protein['seq'])
                ohe_features = torch.tensor(ohe_features, dtype=torch.float32)
            else:
                ohe_features = torch.empty((0, 0))

            if 'expasy' in self.other_features:
                expasy_features = expasy_scale_amino_acid(protein['seq'])
                expasy_features = torch.tensor(expasy_features, dtype=torch.float32)
            else:
                expasy_features = torch.empty((0, 0))

            if 'meiler' in self.other_features:
                meiler_features = meiler_amino_acid(protein['seq'])
                meiler_features = torch.tensor(meiler_features, dtype=torch.float32)
            else:
                meiler_features = torch.empty((0, 0))

            data = torch_geometric.data.Data(
                x=X_ca,
                input_ids = seq_dict['input_ids'],
                attention_mask = seq_dict['attention_mask'],
                node_s=node_s, node_v=node_v,
                ohe_features = ohe_features, expasy_features = expasy_features,
                meiler_features = meiler_features,
                edge_s=edge_s, edge_v=edge_v,
                edge_index=edge_index, mask=mask
            )
            if self.empty_graph:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
            return data

    def _featurize_as_graph_nafis(self, protein, seq_dict = None, debug = False):


        '''
        process the sequence information
        '''


        if self.is_ablang:
            assert seq_dict is not None

        else:

            seq_dict = {
                  'input_ids': [],
                  'attention_mask': [],
                }

            for sequence,max_len in zip([protein['VH_seq'],protein['VL_seq'] ], self.max_len):
                if max_len != 0: 
                    encoding = self.tokenizer.encode_plus(
                      sequence,
                      add_special_tokens=True,
                      max_length=max_len +2, # should be max_len+2, two extra tokens at beginning and end
                      return_token_type_ids=False, 
                      padding= "max_length",
                      return_attention_mask=True,truncation = True,
                      return_tensors='pt',
                    )
                else:
                    '''
                        Warning: Make sure this empty encoding is not used in training, 
                        the prediction network needs to have a flag to indicate to use only the first input_ids like the concat Ablang2
                    '''
                    encoding = {'input_ids': torch.zeros(*seq_dict['input_ids'][-1].shape, device='cpu'),
                                'attention_mask': torch.zeros(*seq_dict['attention_mask'][-1].shape, device='cpu')}


                seq_dict['input_ids'].append(encoding['input_ids'].flatten())
                seq_dict['attention_mask'].append(encoding['attention_mask'].flatten())



        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], 
                                     device=self.device, dtype=torch.float32)   
            # seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
            #                       device=self.device, dtype=torch.long)
            
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            if self.kNN_radius != -1:
                # remove edges that are longer than self.kNN radius
                mask = ((X_ca[edge_index[0]] - X_ca[edge_index[1]])**2).sum(-1) <= self.kNN_radius**2

                if debug:
                    original_degree = torch_geometric.utils.degree(edge_index[0])
                    new_degree = torch_geometric.utils.degree(edge_index[0, mask])

                edge_indx = edge_index[:,mask]


                assert edge_index.shape[-1] != 0, "the graph has no edge after removing larger edges"
                     

            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

            if 'ohe' in self.other_features:

                ohe_features = one_hot_encode_amino_acid(protein['seq'])
                ohe_features = torch.tensor(ohe_features, dtype=torch.float32)

            else:
                ohe_features = torch.empty((0, 0))

            if 'expasy' in self.other_features:

                expasy_features = expasy_scale_amino_acid(protein['seq'])
                expasy_features = torch.tensor(expasy_features, dtype=torch.float32)

            else:
                expasy_features = torch.empty((0, 0))

            if 'meiler' in self.other_features:

                meiler_features = meiler_amino_acid(protein['seq'])
                meiler_features = torch.tensor(meiler_features, dtype=torch.float32)

            else:
                meiler_features = torch.empty((0, 0))



        
        data = torch_geometric.data.Data(x=X_ca, input_ids = seq_dict['input_ids'], 
                                        attention_mask = seq_dict['attention_mask'],
                                        node_s=node_s, node_v=node_v, 
                                        ohe_features = ohe_features, expasy_features = expasy_features,
                                        meiler_features = meiler_features,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index, mask=mask)
        if self.empty_graph:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        if debug:
            return data, original_degree, new_degree
        else:
            return data
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 





def get_seq_coord(structure_obj, target_atoms=["N", "CA", "C", "O"]):
        coordinates = []
        sequence = ""

        for residue in structure_obj.get_residues():
            if 'CA' in residue:
                try:  
                    aa_code = Polypeptide.index_to_one(Polypeptide.three_to_index(residue.get_resname()))
                except KeyError:
                    aa_code = "X"
                sequence += aa_code
                coordinates.append([residue[x].get_coord() for x in target_atoms]) 

        coordinates = np.array(coordinates, dtype=np.float32)
        
        return sequence, coordinates


def process_seq_structure_data(seq_df, target_name, test_set = False, VHH_data = False):

    '''
    The folder address for the structure files are hardcoded. 
    Also, the format of the names of pdb files for all dataset is not uniform, so it is also have some hardcoded expression
    '''

    parser = PDBParser()
    all_data = []
    all_seqs = []
    all_VL_seqs = []
    all_VH_seqs = []
    all_coords = []
    
    if not VHH_data:

        for seq_id in seq_df['Serial number'].to_list():

            if test_set:
                pdb_path = f"./data/pdb_files/AF2mods/AF2_{seq_id}.pdb"
            else:

                if 'BET' not in str(seq_id):
                    pdb_path = f"./data/pdb_files/AF2_249_models/{seq_id}.pdb"
                else:

                    # this needs to be done since there is one sample with different name
                    seq_id_renamed = seq_id.split('_')[-1]
                    seq_id_renamed =  f"{seq_id_renamed}-uniq" if seq_id_renamed=='0' else seq_id_renamed 

                    # pdb path relative, needs to be fixed
                    # pdb_path = f"../../../mlab/mlab-kcmb802/BET_exp_model/models_AF2_BET_exp_1383/{seq_id_renamed}.pdb"
                    pdb_path = f"./data/pdb_files/models_AF2_BET_exp_1383/{seq_id_renamed}.pdb"
                    # raise ValueError('not implemented for BET yet')

            structure = parser.get_structure(f'seq_{seq_id}', pdb_path)

            seq, coords = get_seq_coord(structure)
            
            VH_seq = seq_df[seq_df['Serial number']==seq_id]['VH'].item()
            VL_seq = seq_df[seq_df['Serial number']==seq_id]['VL'].item()
            
            assert seq == VH_seq+VL_seq, f"sequence mismatch with pdb for {seq_id}"
            
            all_data.append({"seq": seq,
                           "VH_seq": VH_seq,
                           "VL_seq": VL_seq,
                           "coords": coords,
                           "target": seq_df[seq_df['Serial number']==seq_id][target_name].item()})
    else:
        # not the way I would like to do.
        structure_dup_counter = 0
        for seq_id, VH_seq in enumerate(seq_df['Sequence'].to_list()):

            parent_dir = './data/pdb_files/for_VHH'
            seq2pdb_map = pd.read_csv(os.path.join(parent_dir, 'all_groups.csv'))


            if test_set:
                raise NotImplementedError('not implemented for processing test data if any')
                pdb_path = ""
            else:
                
                pdb_name = seq2pdb_map['Sequence ID'][seq2pdb_map['Sequence'] == VH_seq].to_list()

                assert len(pdb_name)!=0, "Did not find the pdb with the same sequence from property csv file"

                if len(pdb_name) > 1:
                    structure_dup_counter += 1
                    # warnings.warn("found multiple pdb files with same sequence, picking the first one in according to the order from all_groups.csv")
                pdb_name = pdb_name[0]

                pdb_path = os.path.join(parent_dir,'all_pdbs', f"{pdb_name}.pdb")
                

            structure = parser.get_structure(f'seq_{seq_id}', pdb_path)

            seq, coords = get_seq_coord(structure)
            
            assert seq == VH_seq, f"sequence mismatch with pdb for {pdb_name}"
            
            all_data.append({"seq": seq,
                           "VH_seq": VH_seq,
                           "VL_seq": "",
                           "coords": coords,
                           "target": seq_df.iloc[seq_id][target_name]})
        
        
        
        if structure_dup_counter > 0:
            warnings.warn(f"found multiple pdb files with same sequence for {structure_dup_counter} samples, picked the first one in according to the order from all_groups.csv")
    
    return all_data