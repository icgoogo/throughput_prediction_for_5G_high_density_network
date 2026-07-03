# HDD Modular Pipeline

This is a modular Python refactor of the original `1500id_base_ANN_RF.ipynb`-style notebook.
It separates the notebook into reusable modules for data ingestion, feature engineering,
splitting, Random Forest experiments, optional sequence modeling, reporting, and Kaggle entrypoints.

## Structure

```text
src/hdd_pipeline/
  config.py            # dataclass configs and Kaggle/local paths
  data.py              # raw CSV chunk ingestion + master parquet loader
  features.py          # feature engineering, filters, feature sets, VFL partition
  splits.py            # user-based train/val/test split and missing-value fill
  metrics.py           # regression metrics
  rf_pipeline.py       # RF training/search/final sequential full-training evaluation
  sequence_pipeline.py # optional PyTorch Hybrid LSTM + MLP model
  reporting.py         # run_report.json helper
  orchestrate.py       # CLI orchestration
scripts/
  smoke_test.py        # synthetic small-data smoke test
  run_rf_pipeline.py   # local/Kaggle RF pipeline runner
  run_sequence_pipeline.py
kaggle_kernel/
  train_kernel.py
  kernel-metadata.json
legacy/
  original_converted_script.py
```

## Install locally

```bash
cd hdd_modular_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run smoke test locally

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

## Run RF pipeline on Kaggle/local data

```bash
PYTHONPATH=src python scripts/run_rf_pipeline.py \
  --input-root /kaggle/input/datasets/hammadsyarif/l5ghdd-ds \
  --working-dir /kaggle/working
```

For a validation-only run without final full training:

```bash
PYTHONPATH=src python scripts/run_rf_pipeline.py --no-final
```

## Important full-training change

The final RF training is now **sequential**:

1. Build Stage 1 matrix
2. Train/save Stage 1 model
3. Delete Stage 1 matrix and collect memory
4. Build Stage 2 active-only matrix
5. Train/save Stage 2 model
6. Predict/evaluate in batches

This avoids holding both full Stage 1 and Stage 2 NumPy matrices in RAM at the same time.
It still uses all rows unless you explicitly configure sampling.

By default, final RF uses `full_safe_*` settings to bound tree memory. This is not row undersampling;
it limits tree complexity. Use `--unsafe-original-full-rf` only if you intentionally want the original
high-RAM 100-tree style.

## Kaggle

Edit `kaggle_kernel/kernel-metadata.json`, then push with Kaggle CLI:

```bash
kaggle kernels push -p kaggle_kernel
kaggle kernels status your_kaggle_username/hdd-full-training-experiment
kaggle kernels output your_kaggle_username/hdd-full-training-experiment -p outputs/kaggle -o
```

`/kaggle/working/run_report.json` is written for automation/agent review.
