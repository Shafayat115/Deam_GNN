import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k == 'window_pos':
            collated[k] = [b[k] for b in batch]
        elif k == 'target':
            collated[k] = torch.stack([b[k] for b in batch])
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated

def make_window_indices(seq, i, window_size):
    out = [i]
    d = 1
    while len(out) < window_size:
        if len(out) < window_size:
            if (i + d) < len(seq):
                out.append(i + d)
            else:
                out.append(None)
        if len(out) < window_size:
            if (i - d) >= 0:
                out.append(i - d)
            else:
                out.append(None)
        d += 1
    return out

def extract_windows_and_labels_with_resnum(sequences, resnums, labels, window_size):
    window_idxs = []
    window_labels = []
    for seq_idx, (seq, resnum, label) in enumerate(zip(sequences, resnums, labels)):
        pos = int(resnum) - 1
        idxs = make_window_indices(seq, pos, window_size)
        window_idxs.append((seq_idx, idxs))
        window_labels.append(label)
    return None, window_idxs, window_labels

class WindowMLPDataset(Dataset):
    def __init__(self, sequences, window_idxs, labels, tokenizer, max_length):
        self.sequences = sequences
        self.window_idxs = window_idxs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.window_idxs)

    def __getitem__(self, idx):
        seq_idx, window_pos = self.window_idxs[idx]
        enc = self.tokenizer(
            self.sequences[seq_idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["window_pos"] = window_pos
        item["target"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def __call__(self, metric):
        # For loss, lower is better
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"No improvement ({self.counter}/{self.patience})", flush=True)
            return self.counter >= self.patience

class ESM_MLP_WindowClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, window_size, window_pooling='mean', mlp_hidden=256, mlp_dropout=0.2, model_choice='frozen'):
        super().__init__()
        self.backbone = backbone
        self.window_size = window_size
        self.pooling = window_pooling
        if model_choice == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.pooling == 'mean':
            mlp_input_dim = hidden_size
        elif self.pooling == 'concat':
            mlp_input_dim = hidden_size * window_size
        else:
            raise ValueError(f"Unknown pooling: {window_pooling}")
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 2)
        )

    def forward(self, input_ids, attention_mask, window_pos):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = out.last_hidden_state
        batch_embs = []
        for b, pos_list in enumerate(window_pos):
            toks = []
            for p in pos_list:
                if (p is None) or (p < 0) or (p >= hidden.shape[1]):
                    toks.append(torch.zeros(hidden.shape[2], device=hidden.device, dtype=hidden.dtype))
                else:
                    toks.append(hidden[b, p, :])
            toks = torch.stack(toks, dim=0)
            if self.pooling == 'mean':
                pooled = toks.mean(dim=0)
            elif self.pooling == 'concat':
                pooled = toks.flatten()
            batch_embs.append(pooled)
        features = torch.stack(batch_embs, dim=0)
        return self.mlp(features)

def compute_metrics(y_true, y_pred, y_prob):
    return dict(
        acc   = accuracy_score(y_true, y_pred),
        prec  = precision_score(y_true, y_pred, zero_division=0),
        recall= recall_score(y_true, y_pred, zero_division=0),
        f1    = f1_score(y_true, y_pred),
        roc   = roc_auc_score(y_true, y_prob),
        auprc = average_precision_score(y_true, y_prob)
    )
def extract_features_xgboost(sequences, window_idxs, tokenizer, max_length, backbone, window_pooling):
    dataset = WindowMLPDataset(sequences, window_idxs, [0]*len(window_idxs), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    backbone.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting XGBoost Features"):
            ids = batch['input_ids'].to(backbone.device)
            mask = batch['attention_mask'].to(backbone.device)
            window_pos = batch['window_pos']
            out = backbone(input_ids=ids, attention_mask=mask, return_dict=True)
            hidden = out.last_hidden_state
            batch_embs = []
            for b, pos_list in enumerate(window_pos):
                toks = []
                for p in pos_list:
                    if (p is None) or (p < 0) or (p >= hidden.shape[1]):
                        toks.append(torch.zeros(hidden.shape[2], device=hidden.device, dtype=hidden.dtype))
                    else:
                        toks.append(hidden[b, p, :])
                toks = torch.stack(toks, dim=0)
                if window_pooling == 'mean':
                    pooled = toks.mean(dim=0)
                elif window_pooling == 'concat':
                    pooled = toks.flatten()
                batch_embs.append(pooled.cpu().numpy())
            features.extend(batch_embs)
    return np.array(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",   default="Data/combined_deamid_splits.csv")
    parser.add_argument("--save_dir",    default="./logs_deam")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--PTM_choice",  type=int, default=0)
    parser.add_argument('--window_pooling', choices=['mean', 'concat'], default='mean')
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mlp_hidden', type=int, default=256)
    parser.add_argument('--mlp_dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument("--classifier", choices=["mlp", "xgboost"], default="mlp")
    parser.add_argument("--model_choice", choices=["frozen", "full", "lora"], default="frozen")
    parser.add_argument('--loss_type', choices=['ce', 'focal'], default='ce')
    parser.add_argument('--rate_cutoff', type=int, default=2, help='Use 2 or 5 to define target split')
    parser.add_argument('--label_col', type=str, default=None, help="Use this for custom target override.")
    parser.add_argument('--cv', action="store_true", help="If set, run CV workflow and then final test using mean epochs.")
    parser.add_argument('--split_window_size', type=int, default=1, help="Window size for data grouping/CV split")
    parser.add_argument('--window_size', type=int, default=1, help="Window size for MLP model input")   
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = "facebook/esm2_t6_8M_UR50D" if args.PTM_choice == 0 else "esm_models/ESM2-650M_paired-fine-tuning"
    device = get_device()

    print("\n========== EXPERIMENT CONFIG ==========")
    print(f"Data file             : {args.data_file}")
    print(f"Window size           : {args.window_size}")
    cluster_col = f"win_{args.split_window_size}_cut_{args.rate_cutoff}_ClusterID"
    print(f"CV column             : {cluster_col}")
    print(f"Model checkpoint      : {ckpt}")
    print(f"Classifier            : {args.classifier}")
    print(f"Model choice          : {args.model_choice}")
    print(f"Confidence threshold  : {args.conf_thresh}")
    print(f"Window pooling        : {args.window_pooling}")
    print(f"Batch size            : {args.batch_size}")
    print(f"Results dir           : {args.save_dir}")
    print("=======================================\n", flush=True)

    df = pd.read_csv(args.data_file)
    label_col = args.label_col if args.label_col else f'Rate_cut_{args.rate_cutoff}'
    max_len = df['Sequence'].apply(len).max()

    tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir="esm_models", local_files_only=True)
    base = EsmModel.from_pretrained(ckpt, cache_dir="esm_models", local_files_only=True)
    hidden_size = base.config.hidden_size

    if args.cv:
        print("\n--- Running CV and final test ---")
        cluster_folds = sorted([int(x) for x in df[cluster_col].dropna().unique()])
        all_fold_metrics = []
        epoch_counts     = []

        for fold in cluster_folds:
            print(f"\n==== CV Fold {fold} ====")
            set_seed(42 + fold)
            train_df = df[(df['is_test'] == 0) & (df[cluster_col] != fold)].reset_index(drop=True)
            val_df   = df[(df['is_test'] == 0) & (df[cluster_col] == fold)].reset_index(drop=True)

            train_seqs = train_df["Sequence"].tolist()
            train_resnums = train_df["ResNum"].tolist()
            train_labels = train_df[label_col].tolist()
            val_seqs = val_df["Sequence"].tolist()
            val_resnums = val_df["ResNum"].tolist()
            val_labels = val_df[label_col].tolist()
            _, train_idxs, train_lbls = extract_windows_and_labels_with_resnum(train_seqs, train_resnums, train_labels, args.window_size)
            _, val_idxs, val_lbls = extract_windows_and_labels_with_resnum(val_seqs, val_resnums, val_labels, args.window_size)
            max_len_cv = max([len(seq) for seq in train_seqs + val_seqs])

            train_loader = DataLoader(
                WindowMLPDataset(train_seqs, train_idxs, train_lbls, tokenizer, max_len_cv),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=True, collate_fn=custom_collate_fn
            )
            val_loader = DataLoader(
                WindowMLPDataset(val_seqs, val_idxs, val_lbls, tokenizer, max_len_cv),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                pin_memory=True, collate_fn=custom_collate_fn
            )
            if args.classifier == "mlp":
                set_seed(42 + fold)  # extra repeatability
                base_fold = deepcopy(base)
                model = ESM_MLP_WindowClassifier(
                    backbone=base_fold,
                    hidden_size=hidden_size,
                    window_size=args.window_size,
                    window_pooling=args.window_pooling,
                    mlp_hidden=args.mlp_hidden,
                    mlp_dropout=args.mlp_dropout,
                    model_choice=args.model_choice
                ).to(device)

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
                cw = class_weight.compute_class_weight("balanced", classes=np.array([0,1]), y=train_lbls)
                weight = torch.tensor(cw, dtype=torch.float).to(device)
                if args.loss_type == "focal":
                    class FocalLoss(nn.Module):
                        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
                            super().__init__()
                            self.alpha = alpha
                            self.gamma = gamma
                            self.reduction = reduction
                        def forward(self, logits, targets):
                            logpt = F.log_softmax(logits, dim=1)
                            pt = torch.exp(logpt)
                            logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
                            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
                            focal_term = (1 - pt) ** self.gamma
                            if self.alpha is not None:
                                at = self.alpha.gather(0, targets)
                                logpt = logpt * at
                            loss = -focal_term * logpt
                            if self.reduction == 'mean':
                                return loss.mean()
                            elif self.reduction == 'sum':
                                return loss.sum()
                            else:
                                return loss
                    loss_fn = FocalLoss(alpha=weight, gamma=2.0, reduction='mean').to(device)
                else:
                    loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)
                stopper = EarlyStopper(patience=args.patience)
                best_state_dict = None
                best_val_loss = float('inf')
                n_epochs_run = 0
                for epoch in range(args.n_epochs):
                    model.train()
                    for batch in tqdm(train_loader, leave=False, desc=f"Fold {fold} Epoch {epoch+1}"):
                        ids = batch['input_ids'].to(device)
                        mask = batch['attention_mask'].to(device)
                        window_pos = batch['window_pos']
                        y   = batch['target'].to(device)
                        logits = model(input_ids=ids, attention_mask=mask, window_pos=window_pos)
                        loss = loss_fn(logits, y)
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(trainable_params, 0.1)
                        optimizer.step()

                    # Validation after epoch
                    model.eval()
                    val_losses, all_p, all_t, all_probs = [], [], [], []
                    with torch.no_grad():
                        for batch in val_loader:
                            ids = batch['input_ids'].to(device)
                            mask = batch['attention_mask'].to(device)
                            window_pos = batch['window_pos']
                            y = batch['target'].to(device)
                            logits = model(input_ids=ids, attention_mask=mask, window_pos=window_pos)
                            val_loss = loss_fn(logits, y)
                            val_losses.append(val_loss.item())
                            prob = F.softmax(logits, 1)[:, 1]
                            pred = (prob >= args.conf_thresh).long()
                            all_p.append(pred.cpu().numpy())
                            all_t.append(y.cpu().numpy())
                            all_probs.append(prob.cpu().numpy())
                    all_p = np.concatenate(all_p)
                    all_t = np.concatenate(all_t)
                    all_probs = np.concatenate(all_probs)
                    val_metrics = compute_metrics(all_t, all_p, all_probs)
                    mean_val_loss = np.mean(val_losses)

                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_state_dict = deepcopy(model.state_dict())
                        best_val_metrics = val_metrics.copy()
                        n_epochs_run = epoch + 1

                    print(f"[Fold {fold}] Epoch {epoch+1} VAL LOSS={mean_val_loss:.4f} ACC={val_metrics['acc']:.4f} PREC={val_metrics['prec']:.4f} "
                        f"REC={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} ROC={val_metrics['roc']:.4f} AUPRC={val_metrics['auprc']:.4f}")

                    if stopper(mean_val_loss): break

            if args.classifier == "xgboost":
                X_train = extract_features_xgboost(train_seqs, train_idxs, tokenizer, max_len_cv, base, args.window_pooling)
                X_val = extract_features_xgboost(val_seqs, val_idxs, tokenizer, max_len_cv, base, args.window_pooling)
                y_train = np.array(train_lbls, dtype=int)
                y_val = np.array(val_lbls, dtype=int)

                clf = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    n_estimators=args.n_epochs,
                    use_label_encoder=False,
                    random_state=42+fold
                )
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=args.patience, verbose=False)
                # Manual per-epoch metrics logging
                best_f1 = -1
                best_epoch = 0
                for epoch in range(1, clf.best_iteration + 2):
                    val_probs = clf.predict_proba(X_val, ntree_limit=epoch)[:, 1]
                    val_preds = (val_probs >= args.conf_thresh).astype(int)
                    val_metrics = compute_metrics(y_val, val_preds, val_probs)
                    print(f"[Fold {fold}] Boosting Round {epoch}: "
                        f"ACC={val_metrics['acc']:.4f} PREC={val_metrics['prec']:.4f} REC={val_metrics['recall']:.4f} "
                        f"F1={val_metrics['f1']:.4f} ROC={val_metrics['roc']:.4f} AUPRC={val_metrics['auprc']:.4f}")
                    if val_metrics['f1'] > best_f1:
                        best_f1 = val_metrics['f1']
                        best_epoch = epoch
                        best_val_metrics = val_metrics.copy()
                print(f"Fold {fold} best F1={best_f1:.4f} in {best_epoch} epochs.")
                n_epochs_run = best_epoch
            

            
            all_fold_metrics.append(best_val_metrics)
            epoch_counts.append(n_epochs_run)
            print(f"Fold {fold} best F1={best_val_metrics['f1']:.4f} in {n_epochs_run} epochs.")

        # --- Summary: mean/std for CV ---
        keys = ["acc","prec","recall","f1","roc","auprc"]
        vals = {k: np.array([m[k] for m in all_fold_metrics]) for k in keys}
        print("\n==== CV MEAN ± STD ====")
        for k in keys:
            print(f"{k.upper():6s}: {vals[k].mean():.4f} ± {vals[k].std():.4f}")
        mean_epochs = int(round(np.mean(epoch_counts)))
        print(f"MEAN EPOCHS to use for final model: {mean_epochs} (rounded)")

        # --- Final training on full train set ---
        print("\n==== Retraining on FULL TRAINING DATA ====")
        all_train_df = df[df["is_test"] == 0].reset_index(drop=True)
        all_train_seqs = all_train_df["Sequence"].tolist()
        all_train_resnums = all_train_df["ResNum"].tolist()
        all_train_labels = all_train_df[label_col].tolist()
        _, all_train_idxs, all_train_lbls = extract_windows_and_labels_with_resnum(
            all_train_seqs, all_train_resnums, all_train_labels, args.window_size)
        final_max_len = max([len(seq) for seq in all_train_seqs])
        final_train_loader = DataLoader(
            WindowMLPDataset(all_train_seqs, all_train_idxs, all_train_lbls, tokenizer, final_max_len),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, collate_fn=custom_collate_fn
        )
        if args.classifier == "mlp":
            final_model = ESM_MLP_WindowClassifier(
                backbone=deepcopy(base),
                hidden_size=hidden_size,
                window_size=args.window_size,
                window_pooling=args.window_pooling,
                mlp_hidden=args.mlp_hidden,
                mlp_dropout=args.mlp_dropout,
                model_choice=args.model_choice
            ).to(device)
            trainable_params = [p for p in final_model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
            cw = class_weight.compute_class_weight("balanced", classes=np.array([0,1]), y=all_train_lbls)
            weight = torch.tensor(cw, dtype=torch.float).to(device)
            if args.loss_type == "focal":
                class FocalLoss(nn.Module):
                    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
                        super().__init__()
                        self.alpha = alpha
                        self.gamma = gamma
                        self.reduction = reduction
                    def forward(self, logits, targets):
                        logpt = F.log_softmax(logits, dim=1)
                        pt = torch.exp(logpt)
                        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
                        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
                        focal_term = (1 - pt) ** self.gamma
                        if self.alpha is not None:
                            at = self.alpha.gather(0, targets)
                            logpt = logpt * at
                        loss = -focal_term * logpt
                        if self.reduction == 'mean':
                            return loss.mean()
                        elif self.reduction == 'sum':
                            return loss.sum()
                        else:
                            return loss
                loss_fn = FocalLoss(alpha=weight, gamma=2.0, reduction='mean').to(device)
            else:
                loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)
            set_seed(101)
            for epoch in range(mean_epochs):
                final_model.train()
                for batch in tqdm(final_train_loader, leave=False, desc=f"Retrain Epoch {epoch+1}"):
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    window_pos = batch['window_pos']
                    y = batch['target'].to(device)
                    logits = final_model(input_ids=ids, attention_mask=mask, window_pos=window_pos)
                    loss = loss_fn(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable_params, 0.1)
                    optimizer.step()
        elif args.classifier == "xgboost":
            X_train = extract_features_xgboost(all_train_seqs, all_train_idxs, tokenizer, final_max_len, base, args.window_pooling)
            y_train = np.array(all_train_lbls, dtype=int)
            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=mean_epochs,
                use_label_encoder=False,
                random_state=101
            )
            clf.fit(X_train, y_train, verbose=True)
        

        

        # ==== Holdout/test set evaluation ====
        # Save the final model (after retraining)
        # final_model_path = os.path.join(args.save_dir, f"final_model_window{args.window_size}_cutoff{args.rate_cutoff}.pt")
        # torch.save(final_model.state_dict(), final_model_path)
        # print(f"Saved final trained model to {final_model_path}")

        
        print("\n==== FINAL TEST SET EVALUATION ====")
        test_df = df[df["is_test"] == 1].reset_index(drop=True)
        test_seqs = test_df["Sequence"].tolist()
        test_resnums = test_df["ResNum"].tolist()
        test_labels = test_df[label_col].tolist()
        _, test_idxs, test_lbls = extract_windows_and_labels_with_resnum(
            test_seqs, test_resnums, test_labels, args.window_size)
        max_len_test = max([len(seq) for seq in test_seqs])
        if args.classifier == "mlp":
            test_loader = DataLoader(
                WindowMLPDataset(test_seqs, test_idxs, test_lbls, tokenizer, max_len_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                pin_memory=True, collate_fn=custom_collate_fn
            )
            final_model.eval()
            all_p, all_t, all_probs = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    window_pos = batch['window_pos']
                    y = batch['target'].to(device)
                    logits = final_model(input_ids=ids, attention_mask=mask, window_pos=window_pos)
                    prob = F.softmax(logits, 1)[:, 1]
                    pred = (prob >= args.conf_thresh).long()
                    all_p.append(pred.cpu().numpy())
                    all_t.append(y.cpu().numpy())
                    all_probs.append(prob.cpu().numpy())
            all_p = np.concatenate(all_p)
            all_t = np.concatenate(all_t)
            all_probs = np.concatenate(all_probs)
            test_metrics = compute_metrics(all_t, all_p, all_probs)
            print("TEST ACC={acc:.4f} PREC={prec:.4f} REC={recall:.4f} F1={f1:.4f} ROC={roc:.4f} AUPRC={auprc:.4f}".format(**test_metrics))

        elif args.classifier == "xgboost":
            X_test = extract_features_xgboost(test_seqs, test_idxs, tokenizer, max_len_test, base, args.window_pooling)
            y_test = np.array(test_lbls, dtype=int)
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= args.conf_thresh).astype(int)
            test_metrics = compute_metrics(y_test, test_preds, test_probs)
            print("TEST ACC={acc:.4f} PREC={prec:.4f} REC={recall:.4f} F1={f1:.4f} ROC={roc:.4f} AUPRC={auprc:.4f}".format(**test_metrics))