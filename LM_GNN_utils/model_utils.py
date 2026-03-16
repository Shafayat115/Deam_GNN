import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch_geometric.nn import GATConv, EdgePooling, GCNConv, VGAE
from torch_geometric.nn.models import GIN

from .gvp_utils import GVP, GVPConvLayer, LayerNorm

from torch_scatter import scatter_mean


import os
from torch_scatter import scatter_mean as _scatter_mean

def scatter_mean(src, index, dim=0):
    # if src and index disagree on how many nodes they have, slice them
    if src.size(0) != index.size(0):
        m = min(src.size(0), index.size(0))
        src   = src[:m]
        index = index[:m]
    return _scatter_mean(src, index, dim)

def safe_chain_forward(model, input_ids, attention_mask, hidden_dim):
    if attention_mask is not None and attention_mask.sum() > 0:
        return model(input_ids, attention_mask)
    else:
        B, L = input_ids.shape
        return type('Dummy', (), {'last_hidden_state': torch.zeros(B, L, hidden_dim, device=input_ids.device)})()
    
def safe_lm_forward(model, input_ids, attention_mask):
    # Only call model if there's at least one non-padding token
    if attention_mask is not None and attention_mask.sum() > 0:
        return model(input_ids, attention_mask)
    else:
        # Return zeros tensor shaped [batch_size, length, hidden_dim]
        # Hidden dim can be inferred from model config
        batch_size, length = input_ids.shape if len(input_ids.shape) == 2 else (1, input_ids.shape[0])
        hidden_dim = model.embeddings.word_embeddings.embedding_dim
        device = input_ids.device
        return type('Dummy', (), {'last_hidden_state': torch.zeros(batch_size, length, hidden_dim, device=device)})()


def load_pretrained_VGAE():

    ''' 
        pretrained Variational graph autoencoder from: 
        Duy Nguyen, Viet Thanh, and Truong Son Hy. "Multimodal pretraining for unsupervised protein representation learning." Biology Methods and Protocols (2024)
        https://github.com/HySonLab/Protein_Pretrain/tree/main
    '''
    class VariationalGCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(VariationalGCNEncoder, self).__init__()

            # Define GCN layers for the encoder
            self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

        def forward(self, x, edge_index):
            # Forward pass through the GCN layers with ReLU activation
            x = self.conv1(x, edge_index).relu()

            # Calculate mean (mu) and log standard deviation (logstd)
            mu = self.conv_mu(x, edge_index)
            logstd = self.conv_logstd(x, edge_index)

            return mu, logstd

    # Define the output dimensions for the model
    out_channels = 10
    num_features = 21
    os.path.abspath(__file__)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'multimodal_pretraining_utils/new_VGAE.pt')
    # new_VGAE.pt is saved after running the training script locally since the shared checkpoint file was not saved properly making it fail to be loaded.

    # Create an instance of VGAE using the VariationalGCNEncoder
    vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    vgae_model.load_state_dict(torch.load(model_path))
    vgae_model.eval()
    print(f'pretrained VGAE model loaded from {model_path}')
    return vgae_model


def _freeze_bert(
    bert_model, freeze_bert=True, freeze_layer_count=-1):
    """Freeze parameters in BertModel (in place)

    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether or not to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None

def _freeze_ablang2(
    ablang_model, freeze_ablang=True, freeze_layer_count=-1):
    """Freeze parameters in AbLang model(in place)

    Args:
        ablang_model
        freeze_bert: Bool whether or not to freeze the model
        freeze_layer_count: If freeze_ablamg, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_ablang:
        # freeze the entire bert model
        for param in ablang_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in ablang_model.AbRep.aa_embed_layer.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in ablang_model.AbRep.encoder_blocks[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None

def _freeze_ablang(
    ablang_model, freeze_ablang=True, freeze_layer_count=-1):
    """Freeze parameters in AbLang model(in place)

    Args:
        ablang_model
        freeze_bert: Bool whether or not to freeze the model
        freeze_layer_count: If freeze_ablang, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_ablang:
        # freeze the entire bert model
        for param in ablang_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in ablang_model.AbRep.AbEmbeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in ablang_model.AbRep.EncoderBlocks.Layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None


class PLM_GVP(nn.Module):
    
    def __init__(self, model, n_classes, node_in_dim, node_h_dim, 
        edge_in_dim, edge_h_dim, max_length, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1, input_mode = [],
        num_layers = 3, residual = True, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            node_in_dim = [6, 3]. [node_s.shape[1], node_v.shape[1]]
            node_h_dim = [ 256, 16] # in LM-GVP it was [100, 16]
            edge_in_dim = [32, 1]
            edge_h_dim = [ 32, 1]
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 
            residual: in AbProp it was hard coded to False, meaning no residual updates are used for node embedding
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GVP, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False

        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)

        
        self.residual = residual
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden


        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling

        node_in_dim = (node_in_dim[0] + self.last_config_layer_out, node_in_dim[1])


        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate= self.drop_rate)
            for _ in range(self.num_layers)
        )


        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * self.num_layers,
                node_h_dim[1] * self.num_layers,
            )
            
        self.ns, _ = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns) 
        # This is not the way EdgePooling is proposed to use, it is usually placed after conv block.
        # Can't work on GVP as node embedding has Vector terms.
        # can't use after GATCONV either since outputs of all GATCONV layers are concatenated, and edgepooling reduces the number of nodes,
        # so concatenation is not possible

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (self.ns, 0))
        )
        self.relu = nn.ReLU()
        

        self.pooling_dense = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.ns, eps = self.eps),
            self.dropout,
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device
    def forward_embedding(self, batch):
        # 1) unwrap singleton‐list batches
        if isinstance(batch, list):
            batch = batch[0]

        # 2) prep ESM inputs (we always use batch_size=1 here)
        B = getattr(batch, "num_graphs", 1)
        ids1 = batch["input_ids"][0].reshape(B, -1).long().to(self.device)
        mask1 = batch["attention_mask"][0].reshape(B, -1).to(self.device)
        ids2 = batch["input_ids"][1].reshape(B, -1).long().to(self.device)
        mask2 = batch["attention_mask"][1].reshape(B, -1).to(self.device)

        # 3) run each chain through the LM
        out1 = self.pretrained_model_1(ids1, mask1).last_hidden_state  # [B, L1, D]
        out2 = self.pretrained_model_2(ids2, mask2).last_hidden_state  # [B, L2, D]

        # 4) mask away CLS / SEP
        last1 = (mask1.sum(-1) - 1).long()
        m1 = mask1.detach().clone(); m1[:, 0] = 0
        idx = torch.arange(B, device=self.device)
        m1[idx, last1] = 0

        last2 = (mask2.sum(-1) - 1).long()
        m2 = mask2.detach().clone(); m2[:, 0] = 0
        m2[idx, last2] = 0

        # 5) gather only the “real” residue embeddings
        h_cat = torch.cat([out1, out2], dim=1)                       # [B, L1+L2, D]
        m_cat = torch.cat([m1, m2], dim=1).reshape(-1).bool()        # [L1+L2]
        seq_output = h_cat.reshape(-1, self.last_config_layer_out)[m_cat]  # [n_nodes, D]

        # 6) build structural feats
        h_V = (batch.node_s.to(self.device), batch.node_v.to(self.device))  # node_v is [n_nodes, v_dim, 3]
        h_E = (batch.edge_s.to(self.device), batch.edge_v.to(self.device))
        edge_index = batch.edge_index.to(self.device)

        # 7) fuse LM + structure
        h_V = (torch.cat([h_V[0], seq_output], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # 8) run through GVPConvLayers
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            node_s, node_v = h_V
        else:
            outs, curr = [], h_V
            for layer in self.layers:
                curr = layer(curr, edge_index, h_E)
                outs.append(curr)
            node_s = torch.cat([o[0] for o in outs], dim=-1)
            node_v = torch.cat([o[1] for o in outs], dim=-2)

        # 9) flatten the vector‐feature tensor and concat
        #    node_v: [n_nodes, v_dim, 3] → [n_nodes, v_dim * 3]
        node_v_flat = node_v.reshape(node_v.size(0), -1)
        node_embeddings = torch.cat([node_s, node_v_flat], dim=-1)      # [n_nodes, emb_dim]

        # 10) build a safe batch‐map
        batch_attr = getattr(batch, "batch", None)
        if isinstance(batch_attr, torch.Tensor):
            batch_map = batch_attr.to(self.device)
        else:
            batch_map = torch.zeros(
                node_embeddings.size(0),
                dtype=torch.long,
                device=self.device
            )

        return node_embeddings, batch_map


    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)

        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device) 

        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)



        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1( input_ids_1, attention_mask_1)
            output2 = self.pretrained_model_2( input_ids_2, attention_mask_2)
            last_index = attention_mask_1.sum(-1)-1
            mask_1 = attention_mask_1.detach().clone()    
            mask_1[:,0] = 0
            mask_1[range(len(last_index)),last_index] = 0
            
            last_index = attention_mask_2.sum(-1)-1
            mask_2 = attention_mask_2.detach().clone()    
            mask_2[:,0] = 0
            mask_2[range(len(last_index)), last_index.to(dtype=torch.long)] = 0
            
            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], 1)

            attention_mask = torch.cat([mask_1, mask_2],1)
            
            attention_mask_1d = attention_mask.reshape(-1)

            seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 1] # in AbLang the padding mask is 1, so 0 was used

        h_V = (batch.node_s.to(self.device), batch.node_v.to(self.device))
        h_E = (batch.edge_s.to(self.device), batch.edge_v.to(self.device))

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)

        h_V = (torch.cat([h_V[0], seq_output], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)



        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)
        
        out = self.dropout(self.relu(out))

        if self.universal_pooling:
            padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 


class PLM_GVP_VHH(nn.Module):
    
    def __init__(self, model, n_classes, node_in_dim, node_h_dim, 
        edge_in_dim, edge_h_dim, max_length, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1, input_mode = [],
        num_layers = 3, residual = True, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            node_in_dim = [6, 3]. [node_s.shape[1], node_v.shape[1]]
            node_h_dim = [ 256, 16] # in LM-GVP it was [100, 16]
            edge_in_dim = [32, 1]
            edge_h_dim = [ 32, 1]
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 
            residual: in AbProp it was hard coded to False, meaning no residual updates are used for node embedding
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GVP_VHH, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False

        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)

        
        self.residual = residual
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden


        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling

        node_in_dim = (node_in_dim[0] + self.last_config_layer_out, node_in_dim[1])


        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate= self.drop_rate)
            for _ in range(self.num_layers)
        )


        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * self.num_layers,
                node_h_dim[1] * self.num_layers,
            )
            
        self.ns, _ = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns) 
        # This is not the way EdgePooling is proposed to use, it is usually placed after conv block.
        # Can't work on GVP as node embedding has Vector terms.
        # can't use after GATCONV either since outputs of all GATCONV layers are concatenated, and edgepooling reduces the number of nodes,
        # so concatenation is not possible

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (self.ns, 0))
        )
        self.relu = nn.ReLU()
        

        self.pooling_dense = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.ns, eps = self.eps),
            self.dropout,
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

        
    def forward_embedding(self, batch):
        # 1) unwrap singleton‐list batches
        if isinstance(batch, list):
            batch = batch[0]

        # 2) prep ESM inputs (we always use batch_size=1 here)
        B = getattr(batch, "num_graphs", 1)
        ids1 = batch["input_ids"][0].reshape(B, -1).long().to(self.device)
        mask1 = batch["attention_mask"][0].reshape(B, -1).to(self.device)
        ids2 = batch["input_ids"][1].reshape(B, -1).long().to(self.device)
        mask2 = batch["attention_mask"][1].reshape(B, -1).to(self.device)

        # 3) run each chain through the LM
        out1 = self.pretrained_model_1(ids1, mask1).last_hidden_state  # [B, L1, D]
        out2 = self.pretrained_model_2(ids2, mask2).last_hidden_state  # [B, L2, D]

        # 4) mask away CLS / SEP
        last1 = (mask1.sum(-1) - 1).long()
        m1 = mask1.detach().clone(); m1[:, 0] = 0
        idx = torch.arange(B, device=self.device)
        m1[idx, last1] = 0

        last2 = (mask2.sum(-1) - 1).long()
        m2 = mask2.detach().clone(); m2[:, 0] = 0
        m2[idx, last2] = 0

        # 5) gather only the “real” residue embeddings
        h_cat = torch.cat([out1, out2], dim=1)                       # [B, L1+L2, D]
        m_cat = torch.cat([m1, m2], dim=1).reshape(-1).bool()        # [L1+L2]
        seq_output = h_cat.reshape(-1, self.last_config_layer_out)[m_cat]  # [n_nodes, D]

        # 6) build structural feats
        h_V = (batch.node_s.to(self.device), batch.node_v.to(self.device))  # node_v is [n_nodes, v_dim, 3]
        h_E = (batch.edge_s.to(self.device), batch.edge_v.to(self.device))
        edge_index = batch.edge_index.to(self.device)

        # 7) fuse LM + structure
        h_V = (torch.cat([h_V[0], seq_output], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # 8) run through GVPConvLayers
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            node_s, node_v = h_V
        else:
            outs, curr = [], h_V
            for layer in self.layers:
                curr = layer(curr, edge_index, h_E)
                outs.append(curr)
            node_s = torch.cat([o[0] for o in outs], dim=-1)
            node_v = torch.cat([o[1] for o in outs], dim=-2)

        # 9) flatten the vector‐feature tensor and concat
        #    node_v: [n_nodes, v_dim, 3] → [n_nodes, v_dim * 3]
        node_v_flat = node_v.reshape(node_v.size(0), -1)
        node_embeddings = torch.cat([node_s, node_v_flat], dim=-1)      # [n_nodes, emb_dim]

        # 10) build a safe batch‐map
        batch_attr = getattr(batch, "batch", None)
        if isinstance(batch_attr, torch.Tensor):
            batch_map = batch_attr.to(self.device)
        else:
            batch_map = torch.zeros(
                node_embeddings.size(0),
                dtype=torch.long,
                device=self.device
            )

        return node_embeddings, batch_map


    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)

        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device) 

        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)

        # ========== PLACE DEBUG PRINTS HERE ==========
        # print("=== ESM INPUT DEBUG ===")
        # print("input_ids_1:", input_ids_1)
        # print("attention_mask_1:", attention_mask_1)
        # print("input_ids_2:", input_ids_2)
        # print("attention_mask_2:", attention_mask_2)
        # print("input_ids_1: min", input_ids_1.min().item(), "max", input_ids_1.max().item())
        # print("input_ids_2: min", input_ids_2.min().item(), "max", input_ids_2.max().item())
        # print("Any NaN in input_ids_1?", torch.isnan(input_ids_1.float()).any().item())
        # print("Any NaN in input_ids_2?", torch.isnan(input_ids_2.float()).any().item())
        # print("attention_mask_1: sum", attention_mask_1.sum().item(), "shape", attention_mask_1.shape)
        # print("attention_mask_2: sum", attention_mask_2.sum().item(), "shape", attention_mask_2.shape)
        # =============================================


        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1(input_ids_1, attention_mask_1)
            hidden_dim = output1.last_hidden_state.shape[-1]
            output2 = safe_chain_forward(self.pretrained_model_2, input_ids_2, attention_mask_2, hidden_dim)

            last_index = attention_mask_1.sum(-1) - 1
            mask_1 = attention_mask_1.detach().clone()
            mask_1[:, 0] = 0
            mask_1[range(len(last_index)), last_index] = 0

            last_index = attention_mask_2.sum(-1) - 1
            mask_2 = attention_mask_2.detach().clone()
            mask_2[:, 0] = 0
            mask_2[range(len(last_index)), last_index.to(dtype=torch.long)] = 0

            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], 1)
            # print("[DEBUG] output1.last_hidden_state mean/std/min/max:",
            #     output1.last_hidden_state.mean().item(), output1.last_hidden_state.std().item(),
            #     output1.last_hidden_state.min().item(), output1.last_hidden_state.max().item())
            # print("[DEBUG] output2.last_hidden_state mean/std/min/max:",
            #     output2.last_hidden_state.mean().item(), output2.last_hidden_state.std().item(),
            #     output2.last_hidden_state.min().item(), output2.last_hidden_state.max().item())
            # print("[DEBUG] Any NaN in output1?", torch.isnan(output1.last_hidden_state).any().item())
            # print("[DEBUG] Any NaN in output2?", torch.isnan(output2.last_hidden_state).any().item())

            attention_mask = torch.cat([mask_1, mask_2],1)
            
            attention_mask_1d = attention_mask.reshape(-1)

            seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 1] # in AbLang the padding mask is 1, so 0 was used
            # print("==========[MASK DEBUG VHH]==========")
            # print("attention_mask_1d sum:", attention_mask_1d.sum().item())
            # print("attention_mask_1d shape:", attention_mask_1d.shape)
            # print("Unique mask values:", torch.unique(attention_mask_1d, return_counts=True))
            # print("output shape:", output.shape)
            # try:
            #     print("seq_output shape:", seq_output.shape)
            #     if hasattr(batch, "node_s"):
            #         print("batch.node_s.shape:", batch.node_s.shape)
            #     print("Sample attention_mask_1d[:20]:", attention_mask_1d[:20])
            # except Exception as e:
            #     print("Exception in debug printing shapes:", e)

            # if seq_output.numel() == 0:
            #     print("[ERROR][MASK] seq_output is empty! masking removed all residues.")
            # elif torch.isnan(seq_output).any():
            #     print("[ERROR][MASK] seq_output contains NaN values!")
            #     n_nan = torch.isnan(seq_output.flatten()).sum().item()
            #     print(f"[ERROR] Number of NaN values in seq_output: {n_nan}")
            # if hasattr(batch, "node_s"):
            #     if seq_output.shape[0] != batch.node_s.shape[0]:
            #         print(f"[WARNING] seq_output.shape[0]={seq_output.shape[0]} does not match batch.node_s.shape[0]={batch.node_s.shape[0]}.")
            # print("=====================================")

        h_V = (batch.node_s.to(self.device), batch.node_v.to(self.device))
        h_E = (batch.edge_s.to(self.device), batch.edge_v.to(self.device))

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)
        n_nodes = h_V[0].shape[0]
        if seq_output.shape[0] != n_nodes:
            #print(f"[PLM_GVP_VHH] Aligning seq_output from {seq_output.shape[0]} to {n_nodes}")
            seq_output = seq_output[:n_nodes]
        # print(
        #     '[DEBUG] Before Norm: seq_output mean', seq_output.mean().item(), 
        #     'std', seq_output.std().item(),
        #     'min', seq_output.min().item(), 
        #     'max', seq_output.max().item()
        # )
        h_V = (torch.cat([h_V[0], seq_output], dim=-1), h_V[1])
        # print(
        #     '[DEBUG] After Norm: h_V[0] mean', h_V[0].mean().item(), 
        #     'std', h_V[0].std().item(),
        #     'min', h_V[0].min().item(), 
        #     'max', h_V[0].max().item()
        # )
        h_V = (nn.LayerNorm(h_V[0].shape[-1]).to(h_V[0].device)(h_V[0]), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)



        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)
        
        out = self.dropout(self.relu(out))

        if self.universal_pooling:
            padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 


class PLM_GAT(nn.Module):
    
    def __init__(self, model, n_classes, max_length, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1, input_mode = [],
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GAT, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False
        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)

        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL        
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling

        self.conv_list = nn.ModuleList([GATConv(self.last_config_layer_out, 128, 4),
                                        GATConv(512, 128, 4),
                                        GATConv(512, 256, 4),
                                        GATConv(1024, 256, 4),
                                        ])

        # self.conv1 = GATConv(self.last_config_layer_out, 128, 4)
        # self.conv2 = GATConv(512, 128, 4)
        # self.conv3 = GATConv(512, 256, 4)
        # self.conv4 = GATConv(1024, 256, 4)

        self.conv_dict = {1:512,  # the dimension of each GATCONV layer output
                  2:1024,
                  3:2048,
                  4:3072}
        self.conv_out_dim = self.conv_dict[self.num_layers]

        
            
        self.ns = self.conv_out_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward_embedding(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        edge_index = batch.edge_index.to(self.device)
        batch_size = 1  # you load GAT graphs one at a time

        # 1) prep LM inputs
        ids1 = batch['input_ids'][0].reshape(batch_size, -1).to(self.device).long()
        m1   = batch['attention_mask'][0].reshape(batch_size, -1).to(self.device)
        ids2 = batch['input_ids'][1].reshape(batch_size, -1).to(self.device).long()
        m2   = batch['attention_mask'][1].reshape(batch_size, -1).to(self.device)

        # 2) forward pass through ESM halves
        out1 = self.pretrained_model_1(ids1, m1)
        out2 = self.pretrained_model_2(ids2, m2)

        # 3) build and apply masks (start/end removal)
        last1 = (m1.sum(dim=-1) - 1).long()
        mask1 = m1.detach().clone(); mask1[:, 0] = 0
        idx   = torch.arange(last1.size(0), device=self.device)
        mask1[idx, last1] = 0

        last2 = (m2.sum(dim=-1) - 1).long()
        mask2 = m2.detach().clone(); mask2[:, 0] = 0
        mask2[idx, last2] = 0

        # 4) concat hidden states & mask, select real residues
        h1 = out1.last_hidden_state   # [1, L1, D]
        h2 = out2.last_hidden_state   # [1, L2, D]
        cat_h = torch.cat([h1, h2], dim=1)                   # [1, L1+L2, D]
        cat_m = torch.cat([mask1, mask2], dim=1).reshape(-1).bool()
        seq_output = cat_h.reshape(-1, self.last_config_layer_out)[cat_m]

        # 5) run through GATConv layers (stop before pooling)
        conv_outs = [seq_output]
        for conv in self.conv_list[: self.num_layers]:
            conv_outs.append(conv(conv_outs[-1], edge_index))
        node_embeddings = torch.cat(conv_outs[1:], dim=-1)

        # 6) build batch map and return
        # Covers both missing attribute and present but None
        if hasattr(batch, "batch") and batch.batch is not None:
            batch_map = batch.batch.to(self.device)
        else:
            batch_map = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=self.device)
        return node_embeddings, batch_map

    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)
        
        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device)
        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)


        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1( input_ids_1, attention_mask_1)
            output2 = self.pretrained_model_2( input_ids_2, attention_mask_2)
            # 5) build mask for chain 1
            last_index_1 = (attention_mask_1.sum(dim=-1) - 1).long()
            mask_1 = attention_mask_1.detach().clone()
            mask_1[:, 0] = 0
            idx = torch.arange(last_index_1.size(0), device=last_index_1.device)
            mask_1[idx, last_index_1] = 0

            # 6) build mask for chain 2
            last_index_2 = (attention_mask_2.sum(dim=-1) - 1).long()
            mask_2 = attention_mask_2.detach().clone()
            mask_2[:, 0] = 0
            mask_2[idx, last_index_2] = 0

            # 7) concat and select real residues
            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], dim=1)
            attention_mask = torch.cat([mask_1, mask_2], dim=1).reshape(-1).bool()
            seq_output = output.reshape(-1, self.last_config_layer_out)[attention_mask]

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)

        conv_out_list = [seq_output]
        for conv_layer in self.conv_list[:self.num_layers]:
            conv_out_list.append(conv_layer(conv_out_list[-1], edge_index))

        out = torch.cat(conv_out_list[1:], dim = -1)

        out = self.dropout(self.relu(out))     

        if self.universal_pooling:
            # Patch: ensure attention_mask_1d is defined for all models/types
            if 'attention_mask_1d' not in locals():
                attention_mask_1d = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
            padded_out = torch.zeros(batch_size*self.n, self.ns, device=out.device)
            padded_out[:out.shape[0], :] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 


class PLM_GAT_VHH(nn.Module):
    
    def __init__(self, model, n_classes, max_length, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1, input_mode = [],
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GAT_VHH, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False
        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)

        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL        
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling

        self.conv_list = nn.ModuleList([GATConv(self.last_config_layer_out, 128, 4),
                                        GATConv(512, 128, 4),
                                        GATConv(512, 256, 4),
                                        GATConv(1024, 256, 4),
                                        ])

        # self.conv1 = GATConv(self.last_config_layer_out, 128, 4)
        # self.conv2 = GATConv(512, 128, 4)
        # self.conv3 = GATConv(512, 256, 4)
        # self.conv4 = GATConv(1024, 256, 4)

        self.conv_dict = {1:512,  # the dimension of each GATCONV layer output
                  2:1024,
                  3:2048,
                  4:3072}
        self.conv_out_dim = self.conv_dict[self.num_layers]

        
            
        self.ns = self.conv_out_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward_embedding(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        edge_index = batch.edge_index.to(self.device)
        batch_size = 1  # you load GAT graphs one at a time

        # 1) prep LM inputs
        ids1 = batch['input_ids'][0].reshape(batch_size, -1).to(self.device).long()
        m1   = batch['attention_mask'][0].reshape(batch_size, -1).to(self.device)
        ids2 = batch['input_ids'][1].reshape(batch_size, -1).to(self.device).long()
        m2   = batch['attention_mask'][1].reshape(batch_size, -1).to(self.device)

        # 2) forward pass through ESM halves
        out1 = self.pretrained_model_1(ids1, m1)
        out2 = self.pretrained_model_2(ids2, m2)

        # 3) build and apply masks (start/end removal)
        last1 = (m1.sum(dim=-1) - 1).long()
        mask1 = m1.detach().clone(); mask1[:, 0] = 0
        idx   = torch.arange(last1.size(0), device=self.device)
        mask1[idx, last1] = 0

        last2 = (m2.sum(dim=-1) - 1).long()
        mask2 = m2.detach().clone(); mask2[:, 0] = 0
        mask2[idx, last2] = 0

        # 4) concat hidden states & mask, select real residues
        h1 = out1.last_hidden_state   # [1, L1, D]
        h2 = out2.last_hidden_state   # [1, L2, D]
        cat_h = torch.cat([h1, h2], dim=1)                   # [1, L1+L2, D]
        cat_m = torch.cat([mask1, mask2], dim=1).reshape(-1).bool()
        seq_output = cat_h.reshape(-1, self.last_config_layer_out)[cat_m]

        # 5) run through GATConv layers (stop before pooling)
        conv_outs = [seq_output]
        for conv in self.conv_list[: self.num_layers]:
            conv_outs.append(conv(conv_outs[-1], edge_index))
        node_embeddings = torch.cat(conv_outs[1:], dim=-1)

        # 6) build batch map and return
        # Covers both missing attribute and present but None
        if hasattr(batch, "batch") and batch.batch is not None:
            batch_map = batch.batch.to(self.device)
        else:
            batch_map = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=self.device)
        return node_embeddings, batch_map

    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)
        
        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device)
        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)


        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1(input_ids_1, attention_mask_1)
            hidden_dim = output1.last_hidden_state.shape[-1]
            output2 = safe_chain_forward(self.pretrained_model_2, input_ids_2, attention_mask_2, hidden_dim)

            last_index_1 = (attention_mask_1.sum(dim=-1) - 1).long()
            mask_1 = attention_mask_1.detach().clone()
            mask_1[:, 0] = 0
            idx = torch.arange(last_index_1.size(0), device=last_index_1.device)
            mask_1[idx, last_index_1] = 0

            last_index_2 = (attention_mask_2.sum(dim=-1) - 1).long()
            mask_2 = attention_mask_2.detach().clone()
            mask_2[:, 0] = 0
            mask_2[idx, last_index_2] = 0

            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], dim=1)
            attention_mask = torch.cat([mask_1, mask_2], dim=1).reshape(-1).bool()
            seq_output = output.reshape(-1, self.last_config_layer_out)[attention_mask]

            n_nodes = batch.node_s.shape[0] if hasattr(batch, "node_s") else seq_output.shape[0]
            if seq_output.shape[0] != n_nodes:
                seq_output = seq_output[:n_nodes]

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)
        # === PATCH: Make sure seq_output matches graph node count ===
        n_nodes = batch.node_s.shape[0] if hasattr(batch, "node_s") else seq_output.shape[0]
        if seq_output.shape[0] != n_nodes:
            #print(f"[PLM_GAT_VHH] Aligning seq_output from {seq_output.shape[0]} to {n_nodes}")
            seq_output = seq_output[:n_nodes]
        # === END PATCH ===

        conv_out_list = [seq_output]
        for conv_layer in self.conv_list[:self.num_layers]:
            conv_out_list.append(conv_layer(conv_out_list[-1], edge_index))

        out = torch.cat(conv_out_list[1:], dim = -1)

        out = self.dropout(self.relu(out))     

        if self.universal_pooling:
            # Patch: ensure attention_mask_1d is defined for all models/types
            if 'attention_mask_1d' not in locals():
                attention_mask_1d = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
            padded_out = torch.zeros(batch_size*self.n, self.ns, device=out.device)
            padded_out[:out.shape[0], :] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 

class PLM_GIN(nn.Module):
    
    def __init__(self, model, n_classes, max_length, node_h_dim = 256, input_mode = [],use_jk = None, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1,
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GIN, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False
        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)


        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling
        self.use_jk = use_jk

        self.GIN = GIN(self.last_config_layer_out, node_h_dim, num_layers = self.num_layers,
                        jk = self.use_jk, dropout = drop_rate, train_eps = True)

            
        self.ns = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward_embedding(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        batch_size = 1
        ids1 = batch['input_ids'][0].reshape(batch_size, -1).long().to(self.device)
        m1   = batch['attention_mask'][0].reshape(batch_size, -1).to(self.device)
        ids2 = batch['input_ids'][1].reshape(batch_size, -1).long().to(self.device)
        m2   = batch['attention_mask'][1].reshape(batch_size, -1).to(self.device)

        # ESM → hidden states
        out1 = self.pretrained_model_1(ids1, m1).last_hidden_state
        out2 = self.pretrained_model_2(ids2, m2).last_hidden_state

        # mask away CLS/SEP
        last1 = (m1.sum(dim=-1) - 1).long()
        mask1 = m1.detach().clone(); mask1[:, 0] = 0
        idx   = torch.arange(batch_size, device=self.device)
        mask1[idx, last1] = 0

        last2 = (m2.sum(dim=-1) - 1).long()
        mask2 = m2.detach().clone(); mask2[:, 0] = 0
        mask2[idx, last2] = 0

        # gather only real residues
        cat_h = torch.cat([out1, out2], dim=1)                    # [1, L1+L2, D]
        cat_m = torch.cat([mask1, mask2], dim=1).reshape(-1).bool()
        seq_output = cat_h.reshape(-1, self.last_config_layer_out)[cat_m]

        # GIN conv
        edge_index = batch.edge_index.to(self.device)
        node_embeddings = self.GIN(seq_output, edge_index)

        # build batch_map, but only .to() if it actually exists
        batch_attr = getattr(batch, "batch", None)
        if batch_attr is not None:
            batch_map = batch_attr.to(self.device)
        else:
            batch_map = torch.zeros(
                node_embeddings.size(0),
                dtype=torch.long,
                device=self.device
            )

        return node_embeddings, batch_map
    


    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)
        
        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device)
        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)


        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1( input_ids_1, attention_mask_1)
            output2 = self.pretrained_model_2( input_ids_2, attention_mask_2)
            last_index = attention_mask_1.sum(-1)-1
            mask_1 = attention_mask_1.detach().clone()    
            mask_1[:,0] = 0
            mask_1[range(len(last_index)),last_index] = 0
            
            last_index = attention_mask_2.sum(-1)-1
            mask_2 = attention_mask_2.detach().clone()    
            mask_2[:,0] = 0
            mask_2[range(len(last_index)), last_index.to(dtype=torch.long)] = 0
            
            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], 1)

            attention_mask = torch.cat([mask_1, mask_2],1)
            
            attention_mask_1d = attention_mask.reshape(-1)

            seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 1] # in AbLang the padding mask is 1, so 0 was used


        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)

        out = self.GIN(seq_output, edge_index)
        out = self.dropout(self.relu(out))

        if self.universal_pooling:
            padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 





class PLM_GIN_VHH(nn.Module):
    
    def __init__(self, model, n_classes, max_length, node_h_dim = 256, input_mode = [],use_jk = None, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1,
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(PLM_GIN_VHH, self).__init__()
        self.concat_mode = True if 'concat' in input_mode else False
        if isinstance(model, list):

            self.is_ablang = 'ablang' in str(type(model[0]))
            self.ablang_version = 1
            assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
            _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
            _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model[0])
            self.pretrained_model_2 = deepcopy(model[1])

        else:
            self.is_ablang = 'ablang' in str(type(model))
            
            if self.is_ablang:
                self.ablang_version = 2
                _freeze_ablang2(model, freeze_bert, freeze_layer_count)
            else:
                self.ablang_version = -1
                _freeze_bert(model, freeze_bert, freeze_layer_count)
            
            self.pretrained_model_1 = deepcopy(model)
            if not self.concat_mode:
                self.pretrained_model_2 = deepcopy(model)


        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        if self.concat_mode:
            self.n = sum(self.max_length) + 5 #2 special tokens for VH, 2 for VL, 1 for split, hardcoded for AbLang2
        else:
            self.n = sum(self.max_length) + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling
        self.use_jk = use_jk

        self.GIN = GIN(self.last_config_layer_out, node_h_dim, num_layers = self.num_layers,
                        jk = self.use_jk, dropout = drop_rate, train_eps = True)

            
        self.ns = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward_embedding(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        batch_size = 1
        ids1 = batch['input_ids'][0].reshape(batch_size, -1).long().to(self.device)
        m1   = batch['attention_mask'][0].reshape(batch_size, -1).to(self.device)
        ids2 = batch['input_ids'][1].reshape(batch_size, -1).long().to(self.device)
        m2   = batch['attention_mask'][1].reshape(batch_size, -1).to(self.device)

        # ESM → hidden states
        out1 = self.pretrained_model_1(ids1, m1).last_hidden_state
        out2 = self.pretrained_model_2(ids2, m2).last_hidden_state

        # mask away CLS/SEP
        last1 = (m1.sum(dim=-1) - 1).long()
        mask1 = m1.detach().clone(); mask1[:, 0] = 0
        idx   = torch.arange(batch_size, device=self.device)
        mask1[idx, last1] = 0

        last2 = (m2.sum(dim=-1) - 1).long()
        mask2 = m2.detach().clone(); mask2[:, 0] = 0
        mask2[idx, last2] = 0

        # gather only real residues
        cat_h = torch.cat([out1, out2], dim=1)                    # [1, L1+L2, D]
        cat_m = torch.cat([mask1, mask2], dim=1).reshape(-1).bool()
        seq_output = cat_h.reshape(-1, self.last_config_layer_out)[cat_m]

        # GIN conv
        edge_index = batch.edge_index.to(self.device)
        n_nodes = batch.node_s.shape[0] if hasattr(batch, "node_s") else seq_output.shape[0]
        if seq_output.shape[0] != n_nodes:
            #print(f"[PLM_GIN_VHH] Aligning seq_output from {seq_output.shape[0]} to {n_nodes}")
            seq_output = seq_output[:n_nodes]
        node_embeddings = self.GIN(seq_output, edge_index)

        # build batch_map, but only .to() if it actually exists
        batch_attr = getattr(batch, "batch", None)
        if batch_attr is not None:
            batch_map = batch_attr.to(self.device)
        else:
            batch_map = torch.zeros(
                node_embeddings.size(0),
                dtype=torch.long,
                device=self.device
            )

        return node_embeddings, batch_map
    


    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)
        
        input_ids_1 = batch['input_ids'][0].reshape(batch_size,-1).to(self.device)
        attention_mask_1 = batch['attention_mask'][0].reshape(batch_size,-1).to(self.device)
        if not self.concat_mode:
            input_ids_2 = batch['input_ids'][1].reshape(batch_size,-1).to(self.device)
            attention_mask_2 = batch['attention_mask'][1].reshape(batch_size,-1).to(self.device)


        if self.is_ablang:

            if self.ablang_version == 2: 
                if not self.concat_mode:
                    output1 = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    output2 = self.pretrained_model_2( input_ids_2, return_last_hidden_state = True)               
                    attention_mask = torch.cat([attention_mask_1, attention_mask_2],1)
                    attention_mask_1d = attention_mask.reshape(-1)
                    output = torch.cat([output1, output2], 1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used
                else:
                    output = self.pretrained_model_1( input_ids_1, return_last_hidden_state = True)
                    attention_mask_1d = attention_mask_1.reshape(-1)
                    seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0] # in AbLang the padding mask is 1, so 0 was used

            elif self.ablang_version == 1:
                if self.concat_mode:
                    raise NotImplementedError('Concatenation of VH, VL is not implemented for AbLang1')

                output1 = self.pretrained_model_1( input_ids_1, attention_mask_1, return_last_hidden_state = True)
                output2 = self.pretrained_model_2( input_ids_2, attention_mask_2, return_last_hidden_state = True)
                output = torch.cat([output1, output2], 1)

                last_index = (attention_mask_1 == 0).sum(-1)-1
                mask_1 = attention_mask_1.detach().clone()    
                mask_1[:,0] = 1
                mask_1[range(len(last_index)),last_index] = 1

                last_index = (attention_mask_2 == 0).sum(-1)-1
                mask_2 = attention_mask_2.detach().clone()    
                mask_2[:,0] = 1
                mask_2[range(len(last_index)),last_index] = 1
                attention_mask = torch.cat([mask_1, mask_2],1)

                attention_mask_1d = attention_mask.reshape(-1)

                seq_output = output.reshape(-1,self.last_config_layer_out)[attention_mask_1d == 0]

        else:
            if self.concat_mode:
                raise NotImplementedError('Concatenation of VH, VL is not implemented for ESM model')

            output1 = self.pretrained_model_1(input_ids_1, attention_mask_1)
            hidden_dim = output1.last_hidden_state.shape[-1]
            output2 = safe_chain_forward(self.pretrained_model_2, input_ids_2, attention_mask_2, hidden_dim)

            last_index_1 = (attention_mask_1.sum(dim=-1) - 1).long()
            mask_1 = attention_mask_1.detach().clone()
            mask_1[:, 0] = 0
            mask_1[range(len(last_index_1)), last_index_1] = 0

            last_index_2 = (attention_mask_2.sum(dim=-1) - 1).long()
            mask_2 = attention_mask_2.detach().clone()
            mask_2[:, 0] = 0
            mask_2[range(len(last_index_2)), last_index_2] = 0

            output = torch.cat([output1.last_hidden_state, output2.last_hidden_state], dim=1)
            attention_mask = torch.cat([mask_1, mask_2], dim=1)
            attention_mask_1d = attention_mask.reshape(-1)

            seq_output = output.reshape(-1, self.last_config_layer_out)[attention_mask_1d == 1]
            n_nodes = batch.node_s.shape[0] if hasattr(batch, "node_s") else seq_output.shape[0]
            if seq_output.shape[0] != n_nodes:
                seq_output = seq_output[:n_nodes]

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)

        out = self.GIN(seq_output, edge_index)
        out = self.dropout(self.relu(out))

        if self.universal_pooling:
            padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            out = self.phi(padded_out).squeeze(-1)
            return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 




# {'ohe': 'ohe_features',
# 'expasy': 'expasy_features',
# 'meiler': 'meiler_features' }

class GVP_net(nn.Module):
    
    def __init__(self, model, n_classes, node_in_dim, node_h_dim, 
        edge_in_dim, edge_h_dim, max_length, use_VGAE = False, universal_pooling = False, feature_name = 'ohe', input_norm = False,
        num_layers = 3, residual = True, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            node_in_dim = [6, 3]. [node_s.shape[1], node_v.shape[1]]
            node_h_dim = [ 256, 16] # in LM-GVP it was [100, 16]
            edge_in_dim = [32, 1]
            edge_h_dim = [ 32, 1]
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 
            residual: in AbProp it was hard coded to False, meaning no residual updates are used for node embedding
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(GVP_net, self).__init__()

        # if isinstance(model, list):

        #     self.is_ablang = 'ablang' in str(type(model[0]))
        #     self.ablang_version = 1
        #     assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
        #     _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
        #     _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model[0])
        #     self.pretrained_model_2 = deepcopy(model[1])

        # else:
        #     self.is_ablang = 'ablang' in str(type(model))
            
        #     if self.is_ablang:
        #         self.ablang_version = 2
        #         _freeze_ablang2(model, freeze_bert, freeze_layer_count)
        #     else:
        #         self.ablang_version = -1
        #         _freeze_bert(model, freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model)
        #     self.pretrained_model_2 = deepcopy(model)

        self.feature_name = feature_name
        self.input_norm = input_norm
        self.residual = residual
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        self.n = sum(self.max_length) # + (0 if self.ablang_version == 2 else 4) #2 special tokens for VH, 2 for VL
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden
        
        self.use_VGAE = use_VGAE
        

        if self.use_VGAE:
            assert self.feature_name == 'ohe' , "to use pretrained VGAE, the node features need to be one-hot encoding"
            self.pretrained_gnn = deepcopy(load_pretrained_VGAE())
            for p in self.pretrained_gnn.parameters():
                p.requires_grad = False
            node_in_dim = (node_in_dim[0] - 21 + 10, node_in_dim[1]) # ohe dimension 20. VGAE latent dimension 10


        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        # self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling
        if self.input_norm:
            # self.norm_layer = nn.LayerNorm( node_in_dim[0] ,eps = self.eps)
            self.norm_layer = nn.BatchNorm1d(node_in_dim[0])
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate= self.drop_rate)
            for _ in range(self.num_layers)
        )


        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * self.num_layers,
                node_h_dim[1] * self.num_layers,
            )
            
        self.ns, _ = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns) 
        # This is not the way EdgePooling is proposed to use, it is usually placed after conv block.
        # Can't work on GVP as node embedding has Vector terms.
        # can't use after GATCONV either since outputs of all GATCONV layers are concatenated, and edgepooling reduces the number of nodes,
        # so concatenation is not possible

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (self.ns, 0))
        )
        self.relu = nn.ReLU()
        

        self.pooling_dense = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.ns, eps = self.eps),
            self.dropout,
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, convert2float = False ):

        batch_size = getattr(batch, 'num_graphs', 1)
        
        node_feat = batch[f'{self.feature_name}_features'].to(self.device)


        h_V = (batch.node_s.to(self.device), batch.node_v.to(self.device))
        h_E = (batch.edge_s.to(self.device), batch.edge_v.to(self.device))

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)

        if self.use_VGAE:
            node_feat = self.pretrained_gnn.encode(node_feat, edge_index)
        
        if self.input_norm:
            h_V = (self.norm_layer(torch.cat([h_V[0], node_feat], dim=-1)), h_V[1])
        else:
            h_V = (torch.cat([h_V[0], node_feat], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)



        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)
        
        out = self.dropout(self.relu(out))

        if self.universal_pooling:

            raise NotImplementedError('universal pooling is not implemented')
            # padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            # padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            # padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            # out = self.phi(padded_out).squeeze(-1)
            # return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term

class GAT_net(nn.Module):
    
    def __init__(self, model, n_classes, max_length, node_in_dim,use_VGAE = False,universal_pooling = False, input_norm = False, feature_name = 'ohe', freeze_bert = False, freeze_layer_count = -1,
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(GAT_net, self).__init__()
        
        # if isinstance(model, list):

        #     self.is_ablang = 'ablang' in str(type(model[0]))
        #     self.ablang_version = 1
        #     assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
        #     _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
        #     _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model[0])
        #     self.pretrained_model_2 = deepcopy(model[1])

        # else:
        #     self.is_ablang = 'ablang' in str(type(model))
            
        #     if self.is_ablang:
        #         self.ablang_version = 2
        #         _freeze_ablang2(model, freeze_bert, freeze_layer_count)
        #     else:
        #         self.ablang_version = -1
        #         _freeze_bert(model, freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model)
        #     if not self.concat_mode:
        #         self.pretrained_model_2 = deepcopy(model)

        self.feature_name = feature_name
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        self.n = sum(self.max_length)
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        self.use_VGAE = use_VGAE
        

        if self.use_VGAE:
            assert self.feature_name == 'ohe' , "to use pretrained VGAE, the node features need to be one-hot encoding"
            self.pretrained_gnn = deepcopy(load_pretrained_VGAE())
            for p in self.pretrained_gnn.parameters():
                p.requires_grad = False
            node_in_dim = node_in_dim - 21 + 10 # ohe dimension 20. VGAE latent dimension 10

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        # self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim

        self.universal_pooling = universal_pooling

        self.input_norm = input_norm

        if self.input_norm:
            # self.norm_layer = nn.LayerNorm( node_in_dim[0] ,eps = self.eps)
            self.norm_layer = nn.BatchNorm1d(node_in_dim)

        self.conv_list = nn.ModuleList([GATConv(node_in_dim, 128, 4),
                                        GATConv(512, 128, 4),
                                        GATConv(512, 256, 4),
                                        GATConv(1024, 256, 4),
                                        ])


        self.conv_dict = {1:512,  # the dimension of each GATCONV layer output
                  2:1024,
                  3:2048,
                  4:3072}
        self.conv_out_dim = self.conv_dict[self.num_layers]

        
            
        self.ns = self.conv_out_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, convert2float = False ):

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)
        
        node_feat = batch[f'{self.feature_name}_features'].to(self.device)

        if self.use_VGAE:
            node_feat = self.pretrained_gnn.encode(node_feat, edge_index)
        
        if self.input_norm:
            node_feat = self.norm_layer(node_feat)

        

        conv_out_list = [node_feat]
        for conv_layer in self.conv_list[:self.num_layers]:
            conv_out_list.append(conv_layer(conv_out_list[-1], edge_index))

        out = torch.cat(conv_out_list[1:], dim = -1)

        out = self.dropout(self.relu(out))     

        if self.universal_pooling:
            raise NotImplementedError('universal pooling is not implemented')

            # padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            # padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            # padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            # out = self.phi(padded_out).squeeze(-1)
            # return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term



class GIN_net(nn.Module):
    
    def __init__(self, model, n_classes, max_length, node_in_dim, node_h_dim = 256, input_norm = False, feature_name = 'ohe', use_jk = None,use_VGAE = False, universal_pooling = False, freeze_bert = False, freeze_layer_count = -1,
        num_layers = 3, n_hidden = 1.5, drop_rate = 0.1, layer_norm_epsilon = 1e-12, use_EdgePooling = False):

        '''
            max_length = [max length of VH, max length of VL], only needed when universal pooling is used
            num_layers = 3, in AbPROP it was 4 for both GVP and GAT
            layer_norm_epsilon: used in pooling all the node embeddings, 1e-12 used in 
            n_hidden: how many times larger the hidden dimension is compared to input dimension for pooling network,
                      in AbPROP it is 1.5, in LM-GVP it was 2.0
            universal_pooling: if false, do the mean pooling over the node attributes

        '''
        super(GIN_net, self).__init__()

        # if isinstance(model, list):

        #     self.is_ablang = 'ablang' in str(type(model[0]))
        #     self.ablang_version = 1
        #     assert self.is_ablang, "the list of models must include AbLang1 heavy and light"
        #     _freeze_ablang(model[0], freeze_bert, freeze_layer_count)
        #     _freeze_ablang(model[1], freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model[0])
        #     self.pretrained_model_2 = deepcopy(model[1])

        # else:
        #     self.is_ablang = 'ablang' in str(type(model))
            
        #     if self.is_ablang:
        #         self.ablang_version = 2
        #         _freeze_ablang2(model, freeze_bert, freeze_layer_count)
        #     else:
        #         self.ablang_version = -1
        #         _freeze_bert(model, freeze_bert, freeze_layer_count)
            
        #     self.pretrained_model_1 = deepcopy(model)
        #     if not self.concat_mode:
        #         self.pretrained_model_2 = deepcopy(model)

        self.feature_name = feature_name
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.max_length = max_length
        self.n = sum(self.max_length)
        self.eps = layer_norm_epsilon
        self.n_hidden = n_hidden

        self.use_VGAE = use_VGAE
        

        if self.use_VGAE:
            assert self.feature_name == 'ohe' , "to use pretrained VGAE, the node features need to be one-hot encoding"
            self.pretrained_gnn = deepcopy(load_pretrained_VGAE())
            for p in self.pretrained_gnn.parameters():
                p.requires_grad = False
            node_in_dim = node_in_dim - 21 + 10 # ohe dimension 20. VGAE latent dimension 10        

        # self.last_config_layer_out = model.hparams.vocab_size if self.is_ablang else model.embeddings.word_embeddings.embedding_dim
        # in AbPROP they use the output of AbHead, i.e. logits over the vacabs for each residues, in AbLEF they use the last hidden state of AbRep
        # self.last_config_layer_out = self.pretrained_model_1.hparams.hidden_embed_size if self.is_ablang else self.pretrained_model_1.embeddings.word_embeddings.embedding_dim
        self.input_norm = input_norm 

        self.universal_pooling = universal_pooling
        self.use_jk = use_jk

        if self.input_norm:
            # self.norm_layer = nn.LayerNorm( node_in_dim[0] ,eps = self.eps)
            self.norm_layer = nn.BatchNorm1d(node_in_dim)

        self.GIN = GIN(node_in_dim, node_h_dim, num_layers = self.num_layers,
                        jk = self.use_jk, dropout = drop_rate, train_eps = True)

            
        self.ns = node_h_dim

        self.edge_pooling = use_EdgePooling

        assert not (self.edge_pooling and self.universal_pooling), "cannot use universal pooling and edge pooling together"

        self.edge_pool_layer = EdgePooling(self.ns)

        self.dropout = nn.Dropout(p=self.drop_rate)

        self.relu = nn.ReLU()
        
        # different than GVP
        self.pooling_dense = nn.Sequential(
            nn.Linear(self.ns, int(self.n_hidden*self.ns)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n_hidden*self.ns), n_classes),
        )
        
        self.phi = nn.Sequential(
            nn.Linear(self.ns, int(self.ns*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.ns*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.ns*self.n_hidden), 1),
        )
        self.rho = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n, eps = self.eps),
            self.dropout,
            nn.Linear(self.n, int(self.n*self.n_hidden)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(self.n*self.n_hidden), eps = self.eps),
            self.dropout,
            nn.Linear(int(self.n*self.n_hidden), n_classes)
        )
        
        
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, convert2float = False ):

        edge_index = batch.edge_index.to(self.device)
        batch_size = getattr(batch, 'num_graphs', 1)
        
        node_feat = batch[f'{self.feature_name}_features'].to(self.device)

        if self.use_VGAE:
            node_feat = self.pretrained_gnn.encode(node_feat, edge_index)        

        if self.input_norm:
            node_feat = self.norm_layer(node_feat)

        out = self.GIN(node_feat, edge_index)
        out = self.dropout(self.relu(out))

        if self.universal_pooling:
            raise NotImplementedError('universal pooling is not implemented')
            # padded_out = torch.zeros(batch_size*self.n, self.ns).to(self.device)
            # padded_out[attention_mask_1d == 1*(not self.is_ablang),:] = out
            # padded_out = padded_out.reshape(batch_size, self.n, self.ns)
            # out = self.phi(padded_out).squeeze(-1)
            # return self.rho(out)
        else:
            if self.edge_pooling:
                out, edge_index, new_batch, _ = self.edge_pool_layer(out, edge_index, batch.batch.to(self.device) )
                out = scatter_mean(out, new_batch, dim=0)
            else:
                out = scatter_mean(out, batch.batch.to(self.device), dim=0)

            return self.pooling_dense(out).squeeze(-1) + 0.5 # LM-GVP has this 0.5 term 
