import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from transformers import get_scheduler, EsmModel, AutoTokenizer
from functools import partial
from peft import LoraConfig
from Bio.PDB import PDBParser
from sklearn.metrics import average_precision_score
import random
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from LM_GNN_utils.data_utils import ProteinGraphDataset_v2
from LM_GNN_utils.model_utils import PLM_GVP, PLM_GAT, PLM_GIN

STRUCTURE_ROOT = "structure_data/deamid"
CHAINS_ORDER = ["A", "B", "C", "D", "E", "F"]

def normalize_name(name):
    name = str(name).split('_')[0]
    return name.replace('-', '')

def process_seq_structure_data_deamid(df, structure_dir=STRUCTURE_ROOT,label_col="Rate_cut_2"):

    parser = PDBParser(QUIET=True)

    pdb_files = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]
    norm_pdb_map = {}
    for filename in pdb_files:
        root = normalize_name(os.path.splitext(filename)[0])
        norm_pdb_map.setdefault(root, []).append(os.path.join(structure_dir, filename))

    all_data = []
    missing_rows = []
    for idx, row in df.iterrows():
        mol_norm = normalize_name(row['MolName'])
        chain_id = str(row["Chain"]).strip()

        pdb_candidates = norm_pdb_map.get(mol_norm, [])
        if not pdb_candidates:
            print(f"WARNING: No PDB for mol_norm '{mol_norm}'")
            missing_rows.append(idx)
            continue

        pdb_path = random.choice(pdb_candidates)
        structure = parser.get_structure(f'seq_{mol_norm}', pdb_path)
        try:
            seq, coords = get_seq_coord_chain(structure, chain_id)
        except Exception as e:
            # Print what chains are found for debug
            found_chains = []
            for model in structure:
                for chain in model:
                    found_chains.append(chain.id)
            print(f"Chain {chain_id} missing in PDB for {pdb_path}, skipping. "
                  f"Chains found: {found_chains}. Error: {e}")
            missing_rows.append(idx)
            continue
        all_data.append({
            "seq": seq,
            "VH_seq": seq,
            "VL_seq": "",
            "coords": coords,
            "target": row[label_col],
            "row_index": idx,
        })
    if missing_rows:
        print(f"\nWARNING: {len(missing_rows)} rows dropped due to missing pdb/chain.")
    return all_data, missing_rows

def get_seq_coord_chain(structure_obj, chain_id, target_atoms=["N", "CA", "C", "O"]):

    sequence = ""
    coordinates = []
    found = False
    for model in structure_obj:
        for chain in model:
            if chain.id == chain_id:
                found = True
                for residue in chain.get_residues():
                    if 'CA' not in residue:
                        continue
                    try:
                        aa_code = index_to_one(three_to_index(residue.get_resname()))
                    except Exception:
                        aa_code = "X"
                    sequence += aa_code
                    coordinates.append([residue[x].get_coord() for x in target_atoms])
    if not found:
        raise ValueError(f"Chain {chain_id} not found in structure")
    coordinates = np.array(coordinates, dtype=np.float32)
    return sequence, coordinates

def get_window_indices(center_idx, seq_len, window_size):
    idxs = []
    if window_size == 1:
        idxs = [center_idx]
    elif window_size == 2:
        idxs = [center_idx, min(center_idx+1, seq_len-1)]
    elif window_size == 3:
        idxs = [max(center_idx-1,0), center_idx, min(center_idx+1, seq_len-1)]
    elif window_size == 4:
        idxs = [max(center_idx-1,0), center_idx, min(center_idx+1,seq_len-1), min(center_idx+2,seq_len-1)]
    elif window_size == 5:
        idxs = [max(center_idx-2,0), max(center_idx-1,0), center_idx, min(center_idx+1,seq_len-1), min(center_idx+2,seq_len-1)]
    else:
        raise ValueError(f"Unsupported window size {window_size}")
    idxs = sorted(list(set(idxs)))
    return idxs

class WindowMLPDataset(torch.utils.data.Dataset):
    def __init__(self, df, gnn_dataset, row_index_map, window_size, label_col):
        self.df = df
        self.label_col = label_col
        self.gnn_dataset = gnn_dataset
        self.row_index_map = row_index_map
        self.window_size = window_size
        self._prepare_index()
    def _prepare_index(self):
        self.indices = []
        for idx, row in self.df.iterrows():
            gnn_idx = self.row_index_map[idx]
            seq = self.df.at[idx, "Sequence"]
            residue_idx = int(row["ResNum"]) - 1
            idxs = get_window_indices(residue_idx, len(seq), self.window_size)
            self.indices.append((gnn_idx, idxs, row[self.label_col]))
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        gnn_idx, idxs, label = self.indices[i]
        return (gnn_idx, idxs, label)

class EndToEndGNNwithMLP(nn.Module):
    def __init__(self, gnn_backbone, gnn_embedding_dim, mlp_hidden_dim=256, window_size=1, window_pooling="mean"):
        super().__init__()
        self.gnn = gnn_backbone
        self.window_size = window_size
        self.window_pooling = window_pooling
        if window_pooling == "concat":
            mlp_input_dim = gnn_embedding_dim * window_size
        elif window_pooling == "mean":
            mlp_input_dim = gnn_embedding_dim
        else:
            raise ValueError("window_pooling must be 'mean' or 'concat'")
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 2)
        )
    
    def forward(self, gnn_batch, sample_indices, window_indices_list):
        # get per-node embeddings + batch map from your GNN backbone
        gnn_node_embeddings, batch_map = self.gnn.forward_embedding(gnn_batch)

        # if your backbone didn’t supply a batch_map, assume all nodes==graph 0
        if batch_map is None:
            batch_map = torch.zeros(
                gnn_node_embeddings.size(0),
                dtype=torch.long,
                device=gnn_node_embeddings.device
            )

        reps = []
        for sample_idx, window_idxs in zip(sample_indices, window_indices_list):
            # find which nodes belong to this sample/graph
            mask = (batch_map == sample_idx)
            indices = mask.nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                raise RuntimeError(
                    f"No nodes found for sample_idx={sample_idx}; batch_map={batch_map.tolist()}"
                )

            # take the minimum node index as the start of that graph’s nodes
            node_start = indices.min().item()

            # gather the window of embeddings
            embds = [gnn_node_embeddings[node_start + w] for w in window_idxs]
            embds = torch.stack(embds, dim=0)

            # pool them
            if self.window_pooling == "concat":
                rep = embds.flatten()
            else:  # mean
                rep = embds.mean(dim=0)

            reps.append(rep)

        # stack per-sample representations and run through the MLP head
        window_X = torch.stack(reps, dim=0)
        return self.mlp(window_X)



class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = np.inf
    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

def train_one_epoch(model, gnn_loader, mlp_dataset, optimizer, device, scheduler, loss_fn, grad_clip=True):
    model.train()
    losses = []
    correct_predictions = 0
    with tqdm(total=len(mlp_dataset), desc="Training", mininterval=2.0) as pbar:
        for batch_gnn_indices in range(len(mlp_dataset)):
            gnn_idx, idxs, label = mlp_dataset[batch_gnn_indices]
            gnn_data, _ = gnn_loader.dataset[gnn_idx]
            batch = Batch.from_data_list([gnn_data])
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            output = model(batch, [0], [idxs])
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            _, preds = torch.max(output, dim=1)

            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            correct_predictions += torch.sum(preds == target).detach().item()
            losses.append(loss.item())
        return correct_predictions / len(mlp_dataset), np.mean(losses)

def eval_model(model, gnn_loader, mlp_dataset, device, loss_fn):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for idx in range(len(mlp_dataset)):
            gnn_idx, idxs, label = mlp_dataset[idx]
            gnn_data, _ = gnn_loader.dataset[gnn_idx]
            batch           = Batch.from_data_list([gnn_data])
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            output = model(batch, [0], [idxs])
            _, preds = torch.max(output, dim=1)
            loss = loss_fn(output, target)
            correct_predictions += torch.sum(preds == target).detach().item()
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy()[0])
            all_targets.append(target.cpu().numpy()[0])
            # For binary/multiclass, use the probability/logit for the positive class if output.shape[1] == 2, else as appropriate
            all_logits.append(output.detach().cpu().numpy()[0, 1] if output.shape[1] > 1 else output.detach().cpu().numpy()[0, 0])
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)

    # Compute AUPRC
    try:
        auprc = average_precision_score(all_targets, all_logits)
    except Exception:
        auprc = float('nan')

    other_stats = {
        'acc': accuracy_score(all_targets, all_preds),
        'prec': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_targets, all_logits) if len(np.unique(all_targets)) > 1 else np.nan,
        'auprc': auprc,
        'f1': f1_score(all_targets, all_preds, zero_division=0)
    }
    return correct_predictions / len(mlp_dataset), np.mean(losses), other_stats

def get_model_instance(model_class, PTM_copy, LoRA_config=None, **kwargs):
    from peft import get_peft_model
    if LoRA_config is None:
        fin_model = model_class(PTM_copy, 2, **kwargs)
    else:
        fin_model = get_peft_model(model_class(PTM_copy, 2, **kwargs), LoRA_config)
        fin_model.print_trainable_parameters()
    return fin_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help="Loss function: ce (weighted cross-entropy) or focal")
    parser.add_argument("--structure_dir", default="structure_data/deamid")
    parser.add_argument("--save_dir", default="./logs_deam")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--model_choice", type=int, default=0)
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--LoRA", action='store_true')
    parser.add_argument("--window_size", type=int, default=1,
                help="GNN node embedding window size to feed to final MLP")
    parser.add_argument("--gradclip", action='store_true')
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--kNN_radius", type=int, default=-1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--empty_graph", action='store_true')
    parser.add_argument("--freeze_bert", action='store_true')
    parser.add_argument("--freeze_layer", type=int, default=-1)
    parser.add_argument("--concat_AbLang2", action='store_true')
    parser.add_argument("--window_pooling", type=str, default="mean", choices=["mean","concat"],
                help="How to pool GNN node embeddings for multi-residue window (mean or concat)")
    parser.add_argument('--cv', action='store_true', help="Run cross-validation then final test as in ESM-MLP workflow")
    parser.add_argument('--split_window_size', type=int, default=1, help="Split window size for split/cv column")
    parser.add_argument('--rate_cutoff', type=int, default=2, help="Rate cutoff for split/cv column")

    hparams = parser.parse_args()
      # ─── reproducibility ───
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(42)
    os.makedirs(os.path.join(hparams.save_dir, 'results'), exist_ok=True)

    model_list = {
        0: partial(PLM_GVP, universal_pooling=False),
        1: partial(PLM_GVP, universal_pooling=True),
        2: partial(PLM_GAT, universal_pooling=False),
        3: partial(PLM_GAT, universal_pooling=True),
        4: partial(PLM_GIN, universal_pooling=False),
        5: partial(PLM_GIN, universal_pooling=True),
    }
    model_class = model_list[hparams.model_choice]

    pretrained_model_checkpoint = "facebook/esm2_t6_8M_UR50D"
    pretrained_model = EsmModel.from_pretrained(
        pretrained_model_checkpoint, cache_dir='../esm_models', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_checkpoint, cache_dir='../esm_models', local_files_only=True)

    cluster_col = f"win_{hparams.split_window_size}_cut_{hparams.rate_cutoff}_ClusterID"

    # Decide the correct label/target column
    if hparams.rate_cutoff == 2:
        label_col = "Rate_cut_2"
    elif hparams.rate_cutoff == 5:
        label_col = "Rate_cut_5"
    else:
        raise ValueError(f"rate_cutoff {hparams.rate_cutoff} not recognized; expected 2 or 5")

    df = pd.read_csv(hparams.data_file)
    df["Sequence"] = df.apply(lambda row: row[f"Chain{row['Chain']}_Seq"], axis=1)
    print(f"Read data: {len(df)} rows, {df['MolName'].nunique()} unique MolNames.")

    if hparams.cv:
        cluster_folds = sorted([int(x) for x in df[cluster_col].dropna().unique() if x != -1])
        all_val_metrics = []
        epoch_counts = []

        for val_fold in cluster_folds:
            print(f"\n==== CV Fold {val_fold} ====")
            train_df = df[(df['is_test'] == 0) & (df[cluster_col] != val_fold)].reset_index(drop=True)
            val_df   = df[(df['is_test'] == 0) & (df[cluster_col] == val_fold)].reset_index(drop=True)

            print(f"Train: {len(train_df)}, Val: {len(val_df)}")

            max_len = max(
                train_df['Sequence'].map(len).max(),
                val_df['Sequence'].map(len).max()
            )

            train_data, train_missing = process_seq_structure_data_deamid(train_df, hparams.structure_dir,label_col)
            val_data, val_missing = process_seq_structure_data_deamid(val_df, hparams.structure_dir,label_col)
            train_indices = [d["row_index"] for d in train_data]
            val_indices = [d["row_index"] for d in val_data]
            train_df = train_df.loc[train_indices]
            val_df = val_df.loc[val_indices]

            train_gnn = ProteinGraphDataset_v2(train_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)
            val_gnn = ProteinGraphDataset_v2(val_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)

            train_row_map = {d['row_index']: i for i, d in enumerate(train_data)}
            val_row_map = {d['row_index']: i for i, d in enumerate(val_data)}

            train_mlp = WindowMLPDataset(train_df, train_gnn, train_row_map, hparams.window_size,label_col)
            val_mlp = WindowMLPDataset(val_df, val_gnn, val_row_map, hparams.window_size,label_col)

            train_gnn_loader = DataLoader(train_gnn, batch_size=1, shuffle=False, num_workers=hparams.num_workers)
            val_gnn_loader = DataLoader(val_gnn, batch_size=1, shuffle=False, num_workers=hparams.num_workers)

            PTM_copy = deepcopy(pretrained_model)
            LoRA_cfg = None
            if hparams.LoRA:
                LoRA_cfg = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["query", "key", "value", "dense"],
                    lora_dropout=0.0,
                    bias="none",
                )
            example_graph, _ = train_gnn[0]
                # [insert your model_kwargs logic here as before for each model_class]
                        # [your model_kwargs logic for model_kwargs]
            if "GVP" in str(model_class.func):
                model_kwargs = {
                    'node_in_dim':    (example_graph.node_s.shape[1], example_graph.node_v.shape[1]),
                    'node_h_dim':     (256, 16),
                    'edge_in_dim':    (example_graph.edge_s.shape[1], example_graph.edge_v.shape[1]),
                    'edge_h_dim':     (32, 1),
                    'num_layers':     hparams.num_layers,
                    'max_length':     [max_len, 0],
                    'freeze_bert':    hparams.freeze_bert,
                    'freeze_layer_count': hparams.freeze_layer,
                    'residual':       False,
                    'input_mode': ['concat'] if hparams.concat_AbLang2 else [],
                }
            elif "GAT" in str(model_class.func):
                model_kwargs = {
                    'max_length':         [max_len, 0],
                    'universal_pooling':  model_class.keywords.get("universal_pooling", False),
                    'freeze_bert':        hparams.freeze_bert,
                    'freeze_layer_count': hparams.freeze_layer,
                    'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
                    'num_layers':         hparams.num_layers,
                    'n_hidden':           1.5,
                    'drop_rate':          0.1,
                    'layer_norm_epsilon': 1e-12,
                    'use_EdgePooling':    False,
                }
            elif "GIN" in str(model_class.func):
                model_kwargs = {
                    'max_length':         [max_len, 0],
                    'node_h_dim':         256,
                    'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
                    'universal_pooling':  model_class.keywords.get("universal_pooling", False),
                    'freeze_bert':        hparams.freeze_bert,
                    'freeze_layer_count': hparams.freeze_layer,
                    'num_layers':         hparams.num_layers,
                    'n_hidden':           1.5,
                    'drop_rate':          0.1,
                    'layer_norm_epsilon': 1e-12,
                    'use_EdgePooling':    False,
                    'use_jk':             None,
                }
            else:
                raise ValueError(f"Unknown model_class={model_class}")
            gnn_backbone = get_model_instance(model_class, PTM_copy, LoRA_cfg, **model_kwargs)
            example_batch = Batch.from_data_list([example_graph])
            emb_out, _ = gnn_backbone.forward_embedding(example_batch)
            emb_dim = emb_out.size(1)

            full_model = EndToEndGNNwithMLP(
                gnn_backbone,
                emb_dim,
                mlp_hidden_dim=256,
                window_size=hparams.window_size,
                window_pooling=hparams.window_pooling
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            full_model = full_model.to(device)
            y_train = train_df[label_col].to_numpy()
            classes = np.unique(y_train)
            class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes=classes, y=y_train)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
            if hparams.loss_type == "ce":
                loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
            elif hparams.loss_type == "focal":
                loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
            else:
                raise ValueError("Unknown loss function type: %s" % hparams.loss_type)
            trainable = [p for p in full_model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable, lr=hparams.lr, weight_decay=0.01)
            total_steps = hparams.n_epochs * len(train_mlp)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            best_f1 = -np.inf
            best_model = None
            best_epoch = 0
            early_stopper = EarlyStopper(patience=5, min_delta=1e-4)
            for epoch in range(hparams.n_epochs):
                train_acc, train_loss = train_one_epoch(full_model, train_gnn_loader, train_mlp, optimizer, device, scheduler, loss_fn, grad_clip=hparams.gradclip)
                val_acc, val_loss, val_stats = eval_model(full_model, val_gnn_loader, val_mlp, device, loss_fn)
                print(f"Epoch {epoch+1}:  Train acc={train_acc:.4f}, Train loss={train_loss:.4f}, Val acc={val_acc:.4f}, Val loss={val_loss:.4f}, Val F1={val_stats['f1']:.4f}")
                if val_stats["f1"] > best_f1:
                    best_f1 = val_stats["f1"]
                    best_model = deepcopy(full_model)
                    best_epoch = epoch + 1
                if early_stopper(val_loss):
                    print(f"Early stopping at epoch {epoch+1} for fold {val_fold}")
                    break
            _, _, best_val_stats = eval_model(best_model, val_gnn_loader, val_mlp, device, loss_fn)
            all_val_metrics.append(best_val_stats)
            epoch_counts.append(best_epoch)

            del full_model, optimizer, scheduler
            torch.cuda.empty_cache()
            import gc; gc.collect()

        print("\n==== Mean Validation Metrics (across folds) ====")
        metrics = ["acc", "prec", "recall", "f1", "roc_auc", "auprc"]
        for key in metrics:
            vals = np.array([m[key] for m in all_val_metrics])
            print(f"{key}: {vals.mean():.4f} ± {vals.std():.4f}")
        mean_epochs = int(np.round(np.mean(epoch_counts)))
        print(f"\n==== Mean number of epochs before early stopping (across folds): {np.mean(epoch_counts):.2f} ± {np.std(epoch_counts):.2f}")
        print(f"Mean epochs to use for final train: {mean_epochs}")

        # --- FINAL TRAIN AND TEST ---
        print("\n==== Retraining final model on FULL TRAIN SET ====")
        all_train_df = df[df["is_test"] == 0].reset_index(drop=True)
        test_df      = df[df["is_test"] == 1].reset_index(drop=True)
        max_len = max(
            all_train_df['Sequence'].map(len).max(),
            test_df['Sequence'].map(len).max()
        )

        all_train_data, _ = process_seq_structure_data_deamid(all_train_df, hparams.structure_dir,label_col)
        test_data, _ = process_seq_structure_data_deamid(test_df, hparams.structure_dir, label_col)
        all_train_indices = [d["row_index"] for d in all_train_data]
        test_indices      = [d["row_index"] for d in test_data]
        all_train_df = all_train_df.loc[all_train_indices]
        test_df      = test_df.loc[test_indices]

        all_train_gnn = ProteinGraphDataset_v2(all_train_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)
        test_gnn      = ProteinGraphDataset_v2(test_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)

        train_row_map = {d['row_index']: i for i, d in enumerate(all_train_data)}
        test_row_map = {d['row_index']: i for i, d in enumerate(test_data)}

        all_train_mlp = WindowMLPDataset(all_train_df, all_train_gnn, train_row_map, hparams.window_size,label_col)
        test_mlp      = WindowMLPDataset(test_df, test_gnn, test_row_map, hparams.window_size,label_col)

        all_train_gnn_loader = DataLoader(all_train_gnn, batch_size=1, shuffle=False, num_workers=hparams.num_workers)
        test_gnn_loader      = DataLoader(test_gnn, batch_size=1, shuffle=False, num_workers=hparams.num_workers)

        PTM_copy = deepcopy(pretrained_model)
        example_graph, _ = all_train_gnn[0]
        # [your model_kwargs logic for model_kwargs]
        if "GVP" in str(model_class.func):
            model_kwargs = {
                'node_in_dim':    (example_graph.node_s.shape[1], example_graph.node_v.shape[1]),
                'node_h_dim':     (256, 16),
                'edge_in_dim':    (example_graph.edge_s.shape[1], example_graph.edge_v.shape[1]),
                'edge_h_dim':     (32, 1),
                'num_layers':     hparams.num_layers,
                'max_length':     [max_len, 0],
                'freeze_bert':    hparams.freeze_bert,
                'freeze_layer_count': hparams.freeze_layer,
                'residual':       False,
                'input_mode': ['concat'] if hparams.concat_AbLang2 else [],
            }
        elif "GAT" in str(model_class.func):
            model_kwargs = {
                'max_length':         [max_len, 0],
                'universal_pooling':  model_class.keywords.get("universal_pooling", False),
                'freeze_bert':        hparams.freeze_bert,
                'freeze_layer_count': hparams.freeze_layer,
                'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
                'num_layers':         hparams.num_layers,
                'n_hidden':           1.5,
                'drop_rate':          0.1,
                'layer_norm_epsilon': 1e-12,
                'use_EdgePooling':    False,
            }
        elif "GIN" in str(model_class.func):
            model_kwargs = {
                'max_length':         [max_len, 0],
                'node_h_dim':         256,
                'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
                'universal_pooling':  model_class.keywords.get("universal_pooling", False),
                'freeze_bert':        hparams.freeze_bert,
                'freeze_layer_count': hparams.freeze_layer,
                'num_layers':         hparams.num_layers,
                'n_hidden':           1.5,
                'drop_rate':          0.1,
                'layer_norm_epsilon': 1e-12,
                'use_EdgePooling':    False,
                'use_jk':             None,
            }
        else:
            raise ValueError(f"Unknown model_class={model_class}")
        gnn_backbone = get_model_instance(model_class, PTM_copy, LoRA_cfg, **model_kwargs)
        example_batch = Batch.from_data_list([example_graph])
        emb_out, _ = gnn_backbone.forward_embedding(example_batch)
        emb_dim = emb_out.size(1)

        final_model = EndToEndGNNwithMLP(
            gnn_backbone,
            emb_dim,
            mlp_hidden_dim=256,
            window_size=hparams.window_size,
            window_pooling=hparams.window_pooling
        ).to(device)

        y_train = all_train_df[label_col].to_numpy()
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
        
        if hparams.loss_type == "ce":
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        elif hparams.loss_type == "focal":
            loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
        else:
            raise ValueError("Unknown loss function type: %s" % hparams.loss_type)
        trainable = [p for p in final_model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=hparams.lr, weight_decay=0.01)
        total_steps = mean_epochs * len(all_train_mlp)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(mean_epochs):
            train_acc, train_loss = train_one_epoch(final_model, all_train_gnn_loader, all_train_mlp, optimizer, device, scheduler, loss_fn, grad_clip=hparams.gradclip)
            print(f"Final Model Epoch {epoch+1}/{mean_epochs} Train acc={train_acc:.4f} Train loss={train_loss:.4f}")

        print("\n==== FINAL TEST SET EVALUATION ====")
        test_acc, test_loss, test_stats = eval_model(final_model, test_gnn_loader, test_mlp, device, loss_fn)
        print("TEST   ACC={acc:.4f} PREC={prec:.4f} REC={recall:.4f} F1={f1:.4f} ROC={roc_auc:.4f} AUPRC={auprc:.4f}".format(**test_stats))


























































    # all_val_metrics = []
    # all_test_metrics = []
    # for val_fold in all_folds:
    #     print(f"\n==== CV Fold {val_fold} ====")
    #     train_folds = [f for f in all_folds if f != val_fold]
    #     train_df = df[df['ClusterID'].isin(train_folds)].reset_index(drop=True)
    #     val_df = df[df['ClusterID'] == val_fold].reset_index(drop=True)

    #     print(f"Train samples (before dropping missing): {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(holdout_df)}")

    #     max_len = max(
    #         train_df['Sequence'].map(len).max(),
    #         val_df['Sequence'].map(len).max(),
    #         holdout_df['Sequence'].map(len).max()
    #     )
    #     train_data, train_missing = process_seq_structure_data_deamid(train_df, hparams.structure_dir)
    #     val_data, val_missing = process_seq_structure_data_deamid(val_df, hparams.structure_dir)
    #     test_data, test_missing = process_seq_structure_data_deamid(holdout_df, hparams.structure_dir)

    #     train_indices = [d["row_index"] for d in train_data]
    #     val_indices = [d["row_index"] for d in val_data]
    #     test_indices = [d["row_index"] for d in test_data]
    #     train_df = train_df.loc[train_indices]
    #     val_df = val_df.loc[val_indices]
    #     holdout_df = holdout_df.loc[test_indices]

    #     print(f"Train samples (after dropping missing): {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(holdout_df)}")

    #     train_gnn = ProteinGraphDataset_v2(train_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)
    #     val_gnn = ProteinGraphDataset_v2(val_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)
    #     test_gnn = ProteinGraphDataset_v2(test_data, tokenizer, max_len, 0, 16, hparams.top_k, 16, kNN_radius=hparams.kNN_radius, empty_graph=hparams.empty_graph)

    #     train_row_map = {d['row_index']: i for i, d in enumerate(train_data)}
    #     val_row_map = {d['row_index']: i for i, d in enumerate(val_data)}
    #     test_row_map = {d['row_index']: i for i, d in enumerate(test_data)}

    #     train_mlp = WindowMLPDataset(train_df, train_gnn, train_row_map, hparams.window_size)
    #     val_mlp = WindowMLPDataset(val_df, val_gnn, val_row_map, hparams.window_size)
    #     test_mlp = WindowMLPDataset(holdout_df, test_gnn, test_row_map, hparams.window_size)

    #     train_gnn_loader = DataLoader(train_gnn, batch_size=1, shuffle=False, num_workers=0)
    #     val_gnn_loader = DataLoader(val_gnn, batch_size=1, shuffle=False, num_workers=0)
    #     test_gnn_loader = DataLoader(test_gnn, batch_size=1, shuffle=False, num_workers=0)

    #         # make a fresh copy of the pretrained LM
    #     PTM_copy = deepcopy(pretrained_model)
    #     # add this:
    #     LoRA_cfg = None
    #     if hparams.LoRA:
    #         LoRA_cfg = LoraConfig(
    #             r=8,
    #             lora_alpha=16,
    #             target_modules=["query", "key", "value", "dense"],
    #             lora_dropout=0.0,
    #             bias="none",
    #         )
    #         print(">>> LoRA will inject adapters into:", LoRA_cfg.target_modules)
    #     # grab one example PyG graph + label from your ProteinGraphDataset
    #     example_graph, _ = train_gnn[0]

    #     # build the right kwargs for whichever backbone you chose,
    #     # using the example_graph to infer node/edge feature dims
    #     if "GVP" in str(model_class.func):
    #         model_kwargs = {
    #             'node_in_dim':    (example_graph.node_s.shape[1], example_graph.node_v.shape[1]),
    #             'node_h_dim':     (256, 16),
    #             'edge_in_dim':    (example_graph.edge_s.shape[1], example_graph.edge_v.shape[1]),
    #             'edge_h_dim':     (32, 1),
    #             'num_layers':     hparams.num_layers,
    #             'max_length':     [max_len, 0],
    #             'freeze_bert':    hparams.freeze_bert,
    #             'freeze_layer_count': hparams.freeze_layer,
    #             'residual':       False,
    #             'input_mode': ['concat'] if hparams.concat_AbLang2 else [],
    #         }
    #     elif "GAT" in str(model_class.func):
    #         model_kwargs = {
    #             'max_length':         [max_len, 0],
    #             'universal_pooling':  model_class.keywords.get("universal_pooling", False),
    #             'freeze_bert':        hparams.freeze_bert,
    #             'freeze_layer_count': hparams.freeze_layer,
    #             'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
    #             'num_layers':         hparams.num_layers,
    #             'n_hidden':           1.5,
    #             'drop_rate':          0.1,
    #             'layer_norm_epsilon': 1e-12,
    #             'use_EdgePooling':    False,
    #         }
    #     elif "GIN" in str(model_class.func):
    #         model_kwargs = {
    #             'max_length':         [max_len, 0],
    #             'node_h_dim':         256,
    #             'input_mode':         ['concat'] if hparams.concat_AbLang2 else [],
    #             'universal_pooling':  model_class.keywords.get("universal_pooling", False),
    #             'freeze_bert':        hparams.freeze_bert,
    #             'freeze_layer_count': hparams.freeze_layer,
    #             'num_layers':         hparams.num_layers,
    #             'n_hidden':           1.5,
    #             'drop_rate':          0.1,
    #             'layer_norm_epsilon': 1e-12,
    #             'use_EdgePooling':    False,
    #             'use_jk':             None,
    #         }
    #     else:
    #         raise ValueError(f"Unknown model_class={model_class}")

    #     # now instantiate your GNN backbone
    #     gnn_backbone = get_model_instance(model_class, PTM_copy, LoRA_cfg, **model_kwargs)
    #     # print("=== GNN backbone trainable parameters ===")
    #     # for name, p in gnn_backbone.named_parameters():
    #     #     if p.requires_grad:
    #     #         print(name, p.shape)

    #     # run a dummy forward to learn the true per-node embedding size
    #     # ─ wrap your single graph into a PyG Batch so that .num_graphs exists ─
        
    #     example_batch = Batch.from_data_list([example_graph])
    #     emb_out, _ = gnn_backbone.forward_embedding(example_batch)
    #     emb_dim = emb_out.size(1)

    #     # finally build your end-to-end GNN→MLP model
    #     full_model = EndToEndGNNwithMLP(
    #         gnn_backbone,
    #         emb_dim,
    #         mlp_hidden_dim=256,
    #         window_size=hparams.window_size,
    #         window_pooling=hparams.window_pooling
    #     )

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     full_model = full_model.to(device)
    #     y_train = train_df["Target"].to_numpy()
    #     classes = np.unique(y_train)
    #     class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes=classes, y=y_train)
    #     trainable = [p for p in full_model.parameters() if p.requires_grad]
    #     optimizer = AdamW(trainable, lr=hparams.lr, weight_decay=0.01)
    #     total_steps = hparams.n_epochs * len(train_mlp)
    #     scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    #     best_f1 = -np.inf
    #     best_model = None
    #     early_stopper = EarlyStopper(patience=5, min_delta=1e-4)
    #     epoch_counts = []
    #     for epoch in range(hparams.n_epochs):
    #         train_acc, train_loss = train_one_epoch(full_model, train_gnn_loader, train_mlp, optimizer, device, scheduler, grad_clip=hparams.gradclip)
    #         val_acc, val_loss, val_stats = eval_model(full_model, val_gnn_loader, val_mlp, device)
    #         print(f"Epoch {epoch+1}:  Train acc={train_acc:.4f}, Train loss={train_loss:.4f}, Val acc={val_acc:.4f}, Val loss={val_loss:.4f}, Val F1={val_stats['f1']:.4f}")
    #         if val_stats["f1"] > best_f1:
    #             best_f1 = val_stats["f1"]
    #             best_model = deepcopy(full_model)
    #         if early_stopper(val_loss):
    #             print(f"Early stopping at epoch {epoch+1} for fold {val_fold}")
    #             break
    #     epoch_counts.append(epoch+1)
    #     _, _, best_val_stats = eval_model(best_model, val_gnn_loader, val_mlp, device)
    #     all_val_metrics.append(best_val_stats)
    #     # _, _, test_stats = eval_model(best_model, test_gnn_loader, test_mlp, device)
    #     # print(f"  [Fold {val_fold}] HOLDOUT: ACC={test_stats['acc']:.4f}, PREC={test_stats['prec']:.4f}, RECALL={test_stats['recall']:.4f}, F1={test_stats['f1']:.4f}, ROC={test_stats['roc_auc']:.4f}")
    #     # all_test_metrics.append(test_stats)

    #     del full_model, optimizer, scheduler
    #     torch.cuda.empty_cache()
    #     import gc; gc.collect()

    # print("\n==== Mean Validation Metrics (across folds) ====")
    # for key in ["acc", "prec", "recall", "f1", "roc_auc"]:
    #     vals = np.array([m[key] for m in all_val_metrics])
    #     print(f"{key}: {vals.mean():.4f} ± {vals.std():.4f}")
    # print(f"\n==== Mean number of epochs before early stopping (across folds): {np.mean(epoch_counts):.2f} ± {np.std(epoch_counts):.2f}")
    # # print("\n==== Final averaged holdout test set results (over folds) ====")
    # # for key in ["acc", "prec", "recall", "f1", "roc_auc"]:
    # #     vals = np.array([m[key] for m in all_test_metrics])
    # #     print(f"{key}: {vals.mean():.4f} ± {vals.std():.4f}")