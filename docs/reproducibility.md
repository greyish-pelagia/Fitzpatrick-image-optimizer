# Reproducibility

## Environment

Use Python 3.11 or newer.

```bash
uv sync --extra dev
```

## Demo Run

```bash
uv run fitzopt create-demo-data --output_dir demo_assets --count 6
uv run fitzopt train --model residual-filter --csv_path demo_assets/labels.csv --max_samples 6 --epochs 1 --batch_size 2 --output_dir models/demo
uv run fitzopt evaluate --model residual-filter --model_path models/demo/residual-filter.pth --csv_path demo_assets/labels.csv --split test --metrics_json results/demo-metrics.json --max_samples 2
```

## Full Dataset Run

1. Download Fitzpatrick17k images according to the dataset source terms.
2. Store images as `data/images/<md5hash>.jpg`.
3. Generate synthetic pairs:

```bash
uv run python data/training_preprocess.py --csv_path data/fitzpatrick17k.csv --images_dir data/images --output_images_dir data/training_images --output_csv_path data/labels.csv --seed 42
```

4. Train and evaluate on deterministic splits:

```bash
uv run fitzopt train --model residual-filter --csv_path data/labels.csv --epochs 50 --batch_size 32 --output_dir models
uv run fitzopt evaluate --model residual-filter --model_path models/residual-filter.pth --csv_path data/labels.csv --split test --metrics_json results/residual-filter-test.json
```
