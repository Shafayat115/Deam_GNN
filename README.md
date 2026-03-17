# Deam_GNN

Deam_GNN contains training code for deamidation prediction using:

- An **ESM-only baseline** (`ESM_deamidation.py`)
- An **ESM + structure-aware GNN model** (`Deamid_GNN.py`)

The repository includes:

- Utility modules under `LM_GNN_utils/`
- An example dataset under `Data/`
- Example structure files under `Structure_data/`
- A data splitting script (`Data_Split.py`)
- An `environment.yml` file for environment setup

The included CSV and structure files are **example/demo data only**. They are provided to show the expected format and workflow, not as a full training dataset.

---

## Repository structure

```text
Deam_GNN/
├── Data/
│   └── example.csv
├── LM_GNN_utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── gvp_utils.py
│   └── model_utils.py
├── Structure_data/
│   ├── ...
├── Data_Split.py
├── Deamid_GNN.py
├── ESM_deamidation.py
├── environment.yml
└── README.md
```

---

## Included in this repository

- `Data/example.csv`: example input table showing expected format
- `Data_Split.py`: data partitioning script
- `LM_GNN_utils/`: utility code for data processing and model components
- `ESM_deamidation.py`: sequence-only ESM baseline
- `Deamid_GNN.py`: ESM + GNN training script
- `environment.yml`: conda environment specification

---

## You need to provide

- Your dataset
- Your PDB structure files
- The required ESM model checkpoint files under `esm_models/`

---

## Preparing CV splits with `Data_Split.py`

`Data_Split.py` prepares cross-validation splits from a single input CSV.

It:

1. Loads and filters the dataset
2. Extracts the sequence from chain-specific sequence columns
3. Generates local window sequences around the target residue
4. Groups samples by window sequence
5. Assigns grouped samples into approximately balanced folds
6. Refines folds to improve class balance
7. Writes an output CSV containing fold assignments
8. Writes summary CSV files
9. Generates diagnostic plots

### Default configuration

```python
main_csv = "Data/input.csv"
output_csv = "Data/input_with_splits.csv"
summary_dir = "summaries_joint"
outdir = "plots_iso_deam_splits"
window_min = 1
window_max = 6
n_folds = 5
rate_cutoffs = [2, 5]
```

### Required input

You should provide:

- `Data/input.csv`: the dataset to split into CV folds

### Expected columns

The script expects columns such as:

- `MolName`
- `Chain`
- `ResNum`
- `Rate`
- `N+1`
- Optionally `pH`
- Chain-specific sequence columns such as `ChainA_Seq`, `ChainB_Seq`, `ChainC_Seq`, `ChainD_Seq`, `ChainE_Seq`, `ChainF_Seq`

If present, the script also renames:

- `N.1` → `N-1`
- `N.1.1` → `N+1`

### Filtering behavior

The script removes rows with `N+1 == "PRO"`. If `pH` is present, it keeps only rows with `pH` in `[5.5, 6.0]`.

### Output files

After running `Data_Split.py`, the script generates:

- `Data/input_with_splits.csv`
- Summary CSV files under `summaries_joint/`
- Diagnostic plots under `plots_iso_deam_splits/`

The output CSV contains:

- Generated binary target columns such as `Rate_cut_2` and `Rate_cut_5`
- Generated window sequence columns such as `WindowSeq_win_1`, `WindowSeq_win_2`, etc.
- Generated fold assignment columns such as `win_1_cut_2_ClusterID`, `win_3_cut_5_ClusterID`, etc.

### How to run

```bash
python Data_Split.py
```

If needed, edit the configuration values at the top of the script before running.

### How to use the generated splits

For a chosen window size `w`, cutoff `c`, and fold `f`:

```python
train_df = full_df[full_df[f"win_{w}_cut_{c}_ClusterID"] != f]
val_df   = full_df[full_df[f"win_{w}_cut_{c}_ClusterID"] == f]
```

### Notes

- `Data_Split.py` only prepares splits — it does not train a model
- The generated split columns can be used by both `ESM_deamidation.py` and `Deamid_GNN.py`

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

The included `Data/example.csv` is only a placeholder/example file.

### 2. Structure files

The GNN pipeline requires **PDB files**.

Recommended structure:

```text
Deam_GNN/
├── Structure_data/
│       ├── 1abc.pdb
│       ├── 2xyz.pdb
│       └── ...
```

By default, `Deamid_GNN.py` expects structure files under:

```text
Structure_data/
```

This repository includes example structure files corresponding to the example dataset. For real training and evaluation, you should replace or extend these with your own structure files.

Rows without matching structures or chains may be skipped during preprocessing.

### 3. ESM model files

The scripts load ESM checkpoints locally, so you need to download the model files from Hugging Face and store them under:

```text
esm_models/
```

Or you can update the code to call it from hugginface directly.

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

If you use the alternate PTM option in `ESM_deamidation.py`, you may also need:

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
ESM_deamidation.py
```

### Example command

```bash
python ESM_deamidation.py \
  --data_file Data/example.csv \
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
Deamid_GNN.py
```

### Example command

```bash
python Deamid_GNN.py \
  --data_file Data/example.csv \
  --structure_dir Structure_data/ \
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
python Deamid_GNN.py \
  --data_file Data/example.csv \
  --structure_dir Structure_data/ \
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
4. Test the pipeline using the included `Data/example.csv` and example structure files in `Structure_data/`
5. Replace the example data with your real CSV and real PDB files for actual experiments
6. Run `ESM_deamidation.py` for the sequence baseline
7. Run `Deamid_GNN.py` for the structure-aware model

---

## Example data note

The included `Data/example.csv` and the example files under `Structure_data/` are provided only to demonstrate the expected project layout, file formats, and execution workflow. They are not intended to represent the full dataset required for real model development, benchmarking, or final experiments.

For actual use, you should provide your own complete CSV dataset and matching structure files.

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

- You can use the included example csv and example structure files for a quick test run
- Replace `Data/example.csv` with your actual dataset before real training
- Replace or extend the example structure files before running full experiments
- Make sure the ESM checkpoint is downloaded locally before running either script
- It is recommended to standardize both scripts to use the same `esm_models/` path

---

## Acknowledgment

