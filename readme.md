# Deam_GNN

Deam_GNN contains training code for deamidation prediction using:

- an **ESM-only baseline** (`esm_deamidation.py`)
- an **ESM + structure-aware GNN model** (`deamid_GNN.py`)

The repository currently includes utility modules under `LM_GNN_utils/`, a small dummy CSV under `Data/`, and an `environment.yml` file for creating the software environment. The included CSV is **dummy/example data only** and is meant to show the expected format and repository structure. It is not intended to be a full training dataset.

---

## Repository structure

```text
Deam_GNN/
├── Data/
│   └── dummy.csv
├── LM_GNN_utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── gvp_utils.py
│   └── model_utils.py
├── deamid_GNN.py
├── esm_deamidation.py
├── environment.yml
└── README.md
```

---

## Included in this repository

- `Data/dummy.csv`: example input table showing expected format
- `LM_GNN_utils/`: utility code for data processing and model components
- `esm_deamidation.py`: sequence-only ESM baseline
- `deamid_GNN.py`: ESM + GNN training script
- `environment.yml`: conda environment specification

---

## You need to provide

- Your real CSV dataset
- Your PDB structure files
- The required ESM model checkpoint files under `esm_models/`

---

## Input requirements

### 1. CSV dataset

Both scripts expect a CSV file as input.

Your dataset should contain fields such as:

- `MolName`
- `Chain`
- Sequence-related columns
- Target columns such as `Rate_cut_2` or `Rate_cut_5`
- Split-related columns such as `win_<split_window_size>_cut_<rate_cutoff>_ClusterID`
- `is_test`

Depending on your dataset format, the code may also expect chain-specific sequence columns such as:

- `ChainA_Seq`
- `ChainB_Seq`

The included `Data/dummy.csv` is only a placeholder/example file.

### 2. Structure files

The GNN pipeline requires **PDB files**.

Recommended structure:

```text
Deam_GNN/
├── structure_data/
│   └── deamid/
│       ├── 1abc.pdb
│       ├── 2xyz.pdb
│       └── ...
```

By default, `deamid_GNN.py` expects structure files under:

```text
structure_data/deamid/
```

Rows without matching structures or chains may be skipped during preprocessing.

### 3. ESM model files

The scripts load ESM checkpoints locally, so you need to download the model files from Hugging Face and store them under:

```text
esm_models/
```

Recommended structure:

```text
Deam_GNN/
├── esm_models/
│   └── facebook_esm2_t6_8M_UR50D/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── ...
```

The main model used by the scripts is:

```
facebook/esm2_t6_8M_UR50D
```

If you use the alternate PTM option in `esm_deamidation.py`, you may also need:

```text
esm_models/ESM2-650M_paired-fine-tuning
```

---

## Installation and environment setup

It is recommended to use a dedicated conda environment.

### Option 1: Create the environment from `environment.yml`

```bash
git clone https://github.com/Shafayat115/Deam_GNN.git
cd Deam_GNN
conda env create -f environment.yml
conda activate Ab_dev
```

> **Important note about `environment.yml`:** If environment creation fails, open `environment.yml` and remove any hardcoded `prefix:` line before creating the environment.

### Option 2: Create a new environment manually

```bash
conda create -n deam_gnn python=3.12 -y
conda activate deam_gnn
conda env update -n deam_gnn -f environment.yml
```

---

## Downloading the ESM model

You need to download the ESM model from Hugging Face and place it under `esm_models/`.

Expected local location:

```text
Deam_GNN/esm_models/
```

Main checkpoint used in this repository:

```
facebook/esm2_t6_8M_UR50D
```

Make sure the model files are present locally before running either script.

---

## Running the ESM-only baseline

The sequence-only baseline is:

```text
esm_deamidation.py
```

### Example command

```bash
python esm_deamidation.py \
  --data_file Data/dummy.csv \
  --save_dir logs_esm \
  --batch_size 32 \
  --n_epochs 20 \
  --lr 1e-4 \
  --classifier mlp \
  --model_choice frozen \
  --rate_cutoff 2 \
  --split_window_size 1 \
  --window_size 1 \
  --cv
```

### Common arguments

| Argument | Description |
|---|---|
| `--data_file` | Path to the input CSV file |
| `--save_dir` | Directory for saving logs and outputs |
| `--num_workers` | Number of data loader workers |
| `--batch_size` | Training batch size |
| `--PTM_choice` | PTM model choice |
| `--window_pooling` | Window pooling strategy |
| `--conf_thresh` | Confidence threshold |
| `--n_epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--mlp_hidden` | MLP hidden layer size |
| `--mlp_dropout` | MLP dropout rate |
| `--patience` | Early stopping patience |
| `--classifier` | Classifier type |
| `--model_choice` | ESM model variant |
| `--loss_type` | Loss function type |
| `--rate_cutoff` | Rate cutoff threshold |
| `--label_col` | Target label column name |
| `--cv` | Enable cross-validation |
| `--split_window_size` | Split window size |
| `--window_size` | Sequence window size |

---

## Running the ESM + GNN model

The structure-aware model is:

```text
deamid_GNN.py
```

### Example command

```bash
python deamid_GNN.py \
  --data_file Data/dummy.csv \
  --structure_dir structure_data/deamid \
  --save_dir logs_gnn \
  --batch_size 8 \
  --n_epochs 20 \
  --lr 1e-4 \
  --model_choice 0 \
  --top_k 20 \
  --num_layers 3 \
  --rate_cutoff 2 \
  --split_window_size 1 \
  --window_size 1 \
  --cv
```

### Example with LoRA

```bash
python deamid_GNN.py \
  --data_file Data/dummy.csv \
  --structure_dir structure_data/deamid \
  --save_dir logs_gnn_lora \
  --batch_size 8 \
  --n_epochs 20 \
  --lr 1e-4 \
  --model_choice 0 \
  --LoRA \
  --rate_cutoff 2 \
  --split_window_size 1 \
  --window_size 1
```

### Common arguments

| Argument | Description |
|---|---|
| `--data_file` | Path to the input CSV file |
| `--loss_type` | Loss function type |
| `--structure_dir` | Path to PDB structure files |
| `--save_dir` | Directory for saving logs and outputs |
| `--num_workers` | Number of data loader workers |
| `--batch_size` | Training batch size |
| `--lr` | Learning rate |
| `--n_epochs` | Number of training epochs |
| `--model_choice` | GNN model variant (see mapping below) |
| `--early_stop` | Enable early stopping |
| `--LoRA` | Enable LoRA fine-tuning |
| `--window_size` | Sequence window size |
| `--gradclip` | Gradient clipping value |
| `--top_k` | Top-K neighbors for graph construction |
| `--kNN_radius` | Radius for kNN graph |
| `--num_layers` | Number of GNN layers |
| `--empty_graph` | Use empty graph (no edges) |
| `--freeze_bert` | Freeze ESM encoder weights |
| `--freeze_layer` | Number of ESM layers to freeze |
| `--concat_AbLang2` | Concatenate AbLang2 embeddings |
| `--window_pooling` | Window pooling strategy |
| `--cv` | Enable cross-validation |
| `--split_window_size` | Split window size |
| `--rate_cutoff` | Rate cutoff threshold |

### Model choice mapping

| Value | Model |
|---|---|
| `0` | PLM_GVP |
| `1` | PLM_GVP with universal pooling |
| `2` | PLM_GAT |
| `3` | PLM_GAT with universal pooling |
| `4` | PLM_GIN |
| `5` | PLM_GIN with universal pooling |

---

## Recommended workflow

1. Clone the repository
2. Create the conda environment
3. Download the ESM checkpoint into `esm_models/`
4. Add your real CSV file
5. Add your PDB files under `structure_data/deamid/`
6. Run `esm_deamidation.py` for the sequence baseline
7. Run `deamid_GNN.py` for the structure-aware model

---

## Dummy data note

The included `Data/dummy.csv` is only a placeholder/example file. It is useful for showing the expected repository structure and input style, but it is not meant to be used as a real training dataset.

Structure files are also not included in this repository. You must provide your own PDB files under `structure_data/deamid/`.

---

## Troubleshooting

### Model files not found

If you get an error related to Hugging Face or ESM model loading, check that:

- The model has been downloaded locally
- The files are placed under the correct `esm_models/` location
- The code path matches your local folder layout

### Missing structures

If the GNN pipeline skips rows, check:

- Whether the corresponding `.pdb` file exists
- Whether the PDB filename matches `MolName`
- Whether the requested `Chain` exists in the PDB file

### Environment creation fails

If `conda env create -f environment.yml` fails, remove any hardcoded `prefix:` line from `environment.yml` and try again.

---

## Notes

- Replace `Data/dummy.csv` with your actual dataset before real training
- Add your own structure files before running the GNN model
- Make sure the ESM checkpoint is downloaded locally before running either script
- It is recommended to standardize both scripts to use the same `esm_models/` path

---

## Acknowledgment

This repository currently provides the training code, utility modules, and an example project layout. Users should replace the dummy data and add the real model files and structure files before running full experiments.