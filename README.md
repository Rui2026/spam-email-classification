# Spam Email Classification Using Multiple Machine Learning Models

A course-level text binary classification project: under the same train/test split, compare **Logistic Regression**, **KNN**, **Multinomial Naive Bayes**, **MLP**, and **LSTM** on spam detection, and produce metrics, confusion matrices, and comparison plots.

## Project Structure

```
spam_email_classification/
├── data/
│   ├── raw/              # Raw CSV (sample: emails.csv)
│   └── processed/        # Preprocessed data (generated at runtime)
├── notebooks/            # Optional Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── config.py         # Paths and hyperparameters
│   ├── preprocess.py     # Text cleaning + optional stopwords
│   ├── features.py       # BoW / TF-IDF; LSTM vocab + padding
│   ├── ml_models.py      # sklearn model constructors
│   ├── dl_models.py      # PyTorch MLP / LSTM
│   ├── dataset.py        # torch Dataset wrappers
│   ├── train_ml.py       # Classical ML training + persistence
│   ├── train_dl.py       # Deep learning training + best-checkpoint
│   ├── evaluate.py       # Metrics, confusion matrices, comparison plots
│   └── utils.py          # IO, label encoding, directory creation
├── models/               # Trained .pkl / .pt files
├── outputs/
│   ├── figures/          # Confusion matrices and comparison PNGs
│   ├── metrics/          # JSON metrics
│   └── predictions/      # Test-set prediction CSVs
├── main.py               # Unified entry point
├── requirements.txt
└── README.md
```

## Installation

Requires **Python 3.10+**. We recommend **Python 3.12** (or 3.10/3.11) for the virtual environment: some dependencies are not yet fully stable on **Python 3.14**, which can cause slow installs or runtime errors.

```bash
cd spam_email_classification
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If **stopword removal** is enabled (set `REMOVE_STOPWORDS = True` in `src/config.py`), NLTK will auto-download the `stopwords` corpus on first run.

### macOS "Python quit unexpectedly" / Matplotlib hangs

`main.py` already sets `MPLCONFIGDIR` to a project-local directory and `src/evaluate.py` forces the non-interactive **`Agg`** backend (`matplotlib.use("Agg")`) to avoid clashes between the default GUI backend and the font cache on some systems. If issues persist, recreate the venv with **Python 3.12** before reinstalling dependencies.

## Dataset Format

The CSV must contain at least two columns (column names are configurable in `src/config.py`; defaults are `text` and `label`):

| Column | Description |
|--------|-------------|
| `text` | Email body, subject, or other raw text |
| `label`| `spam` / `ham`, or `1` / `0` (also accepts aliases like `yes/no`, `true/false`) |

Place the file under `data/raw/`, or pass its path via `--data`. The repo includes a tiny sample `data/raw/emails.csv` for quick smoke tests.

### Using a Kaggle Dataset

There are three options — pick whichever you prefer.

**Option 0 — `kagglehub` (recommended, no token setup needed):**

```bash
pip install kagglehub
python scripts/download_kaggle.py \
    --slug abdmental01/email-spam-dedection \
    --output data/raw/email_spam.csv
```

`scripts/download_kaggle.py` will:

1. Download the dataset to the local cache (default: `~/.cache/kagglehub/...`) via `kagglehub`.
2. Locate a CSV inside the extracted directory (or list multiple CSVs; specify with `--csv-name` if needed).
3. Call `prepare_kaggle.py` to normalise the columns to `text,label` and write to `data/raw/email_spam.csv`.

**Option 1 — Official Kaggle CLI (token configured):**

```bash
pip install kaggle
kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/raw --unzip
```

**Option 2 — Web download**: download from the dataset page on Kaggle, unzip, and place the CSV under `data/raw/`.

After downloading, choose one of the following:

A. **Convert to the project's standard format** (recommended, easier to reuse):

   ```bash
   python scripts/prepare_kaggle.py \
       --input data/raw/spam.csv \
       --output data/raw/sms_spam.csv \
       --encoding latin-1 \
       --dedup
   ```

   The script auto-detects common column names (`v1/v2`, `Category/Message`, etc.) and writes a standardised CSV with `text,label` columns plus a class-distribution log.

   Then train as usual:

   ```bash
   python main.py --mode all --feature tfidf --runs 3 \
       --data data/raw/sms_spam.csv
   ```

B. **Use the original column names directly** (good for one-off experiments):

   ```bash
   python main.py --mode ml --feature tfidf \
       --data data/raw/spam.csv \
       --text-col v2 --label-col v1 \
       --encoding latin-1
   ```

   `--text-col` / `--label-col` / `--encoding` let `main.py` consume any CSV with arbitrary column names or encodings.

## Running

From the project root (`spam_email_classification/`):

```bash
# Classical ML only (LR / KNN / MultinomialNB)
python main.py --mode ml --feature tfidf

# Deep learning only (MLP + LSTM)
python main.py --mode dl --feature tfidf

# All models
python main.py --mode all --feature bow
```

Common arguments:

- `--mode`: `ml` | `dl` | `all`
- `--feature`: `bow` | `tfidf` (applies to sklearn models and MLP; LSTM uses its own vocabulary)
- `--data`: path to the CSV (default: `data/raw/emails.csv`)
- `--seed`: base random seed (default 42)
- `--runs N`: repeat the experiment N times with seeds `seed, seed+1, ...`, automatically aggregating per-model **mean ± std**
- `--epochs / --batch-size / --lr`: override DL hyperparameters (otherwise read from `config.py`)
- `--text-col / --label-col`: CSV column names (default `text` / `label`; useful for raw Kaggle files)
- `--encoding`: CSV encoding (use `latin-1` for the Kaggle SMS Spam Collection)
- `--no-class-weight`: disable class-balanced loss for DL models (default: enabled; useful for ablations)
- `--tag NAME`: write all artifacts under `outputs/<NAME>/` and `models/<NAME>/` so different experiments don't overwrite each other; `--tag auto` uses a timestamp
- `--log-level`: `DEBUG` / `INFO` / `WARNING` / `ERROR`

### Recommended Commands for Reports

Place each experiment under its own `--tag` subdirectory so they don't overwrite each other and can be summarised together later:

```bash
# 1) Baseline: all models + TF-IDF + class-balanced loss
python main.py --mode all --feature tfidf --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --tag baseline

# 2) Ablation: disable class weighting to study the effect of imbalance
python main.py --mode dl --feature tfidf --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --no-class-weight --tag no_class_weight

# 3) Feature comparison: BoW vs TF-IDF
python main.py --mode all --feature bow --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --tag bow

# 4) Quick classical-ML pass (seconds; handy for parameter sweeps)
python main.py --mode ml --feature tfidf --runs 5 --seed 42 --tag ml_quick
```

If you don't pass `--tag`, the original flat structure is used (`outputs/figures/...`, `models/...`). That works for a final one-shot report, but each new run will **overwrite** any previous artifacts with matching names.

### Class Imbalance Handling

Datasets like `abdmental01/email-spam-dedection` are roughly 88% ham and 12% spam. Without intervention, the LSTM would settle into the trivial "always predict ham" local minimum (validation loss stuck around 0.36, exactly matching the prior cross-entropy), giving spam recall ≈ 0.

The project addresses this with three mechanisms (all enabled by default):

1. **Class-balanced cross-entropy** (`config.USE_CLASS_WEIGHT = True`): per-class weight `N / (K · n_c)`, equivalent to sklearn's `class_weight='balanced'`. Disable with `--no-class-weight` for ablations.
2. **Gradient clipping** (`config.GRAD_CLIP_NORM = 5.0`): `clip_grad_norm_` to prevent LSTM gradient explosions.
3. **Bidirectional LSTM** (`config.LSTM_BIDIRECTIONAL = True`): concatenate the final forward and backward hidden states; typically +1–3% F1 on short text.

Empirically on `email_spam.csv`, enabling these lifts the LSTM test F1 from ≈ 0 to 0.80+, with spam recall ≈ 0.86.

## Outputs

`<tag>/` below is the optional `--tag` subdirectory; without `--tag` everything lands in flat `outputs/` and `models/`.

| Path | Contents |
|------|----------|
| `data/processed/processed.csv` | Original columns plus `clean_text` |
| `models/<tag>/vectorizer_{bow|tfidf}.pkl` | Text vectoriser |
| `models/<tag>/{logistic_regression,knn,naive_bayes}_{feature}.pkl` | sklearn models |
| `models/<tag>/mlp_{feature}.pt` | MLP checkpoint |
| `models/<tag>/lstm.pt` | LSTM checkpoint (includes vocab) |
| `outputs/<tag>/metrics/metrics_{mode}_{feature}_run*_seed*.json` | Per-run, per-model metrics |
| `outputs/<tag>/metrics/summary_{mode}_{feature}.csv` | Summary table (with `*_mean / *_std` columns when multiple runs are used; ready to paste into a report) |
| `outputs/<tag>/metrics/summary_{mode}_{feature}.json` | JSON version of the summary |
| `outputs/<tag>/metrics/label_mapping.json` | Mapping from raw labels to 0/1 |
| `outputs/<tag>/figures/cm_*.png` | Confusion matrices per model |
| `outputs/<tag>/figures/curves_{mlp,lstm}_*.png` | **Training/validation loss curves** |
| `outputs/<tag>/figures/model_comparison_*.png` | Bar chart comparing models on accuracy / precision / recall / F1 |
| `outputs/<tag>/predictions/*.csv` | Test-set predictions |
