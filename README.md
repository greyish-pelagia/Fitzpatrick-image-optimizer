# Fitzpatrick Image Optimizer

Experimental PyTorch project for Fitzpatrick-conditioned illumination normalization on dermatology-style images. The repository demonstrates an end-to-end ML workflow: synthetic degradation, model training, holdout evaluation, grouped metrics, inference, and reproducible demo commands.

![Example comparison grid](comparison_grid.png)

## What This Project Shows

- Package-quality PyTorch code with train/evaluate/infer CLIs.
- Deterministic synthetic degradation for paired image restoration experiments.
- Two model families:
  - `residual-filter`: a ResNet-conditioned residual image filter inspired by DeepLPF-style parameter prediction.
  - `illumination-unet`: a FiLM-conditioned U-Net with Sobel texture features and refinement CNN.
- Holdout evaluation with identity baseline metrics grouped by Fitzpatrick scale.
- A generated tiny demo dataset so the project runs from a fresh clone.

## Non-Clinical Scope

This is not a diagnostic system and is not validated for clinical use. The project explores image normalization under synthetic lighting shifts. Claims about fairness, diagnostic accuracy, or medical utility require separate downstream experiments and clinical review.

## Quickstart

```bash
uv sync --extra dev
uv run fitzopt create-demo-data --output_dir demo_assets --count 6
uv run fitzopt train --model residual-filter --csv_path demo_assets/labels.csv --max_samples 6 --epochs 1 --batch_size 2 --output_dir models/demo
uv run fitzopt evaluate --model residual-filter --model_path models/demo/residual-filter.pth --csv_path demo_assets/labels.csv --split test --metrics_json results/demo-metrics.json --max_samples 2
uv run fitzopt infer --model residual-filter --model_path models/demo/residual-filter.pth --csv_path demo_assets/labels.csv --output_dir results/demo-inference --max_samples 2
```

## Dataset

The full experiments use Fitzpatrick17k metadata and locally downloaded images. The repository does not vendor the full image dataset. Use `data/training_preprocess.py` to create synthetic pairs after downloading source images.

## Evaluation

Metrics are reported on deterministic train/validation/test splits. Public reports must include:

- L1
- MSE
- PSNR
- SSIM
- identity baseline comparison
- grouped metrics by Fitzpatrick scale

See `docs/results-schema.md` for the JSON report format.

## Limitations

- Synthetic degradation does not cover all real clinical acquisition artifacts.
- Fitzpatrick labels are used as conditioning metadata, not as a guarantee of fairness.
- The residual-filter model is DeepLPF-inspired but does not implement the full DeepLPF filter family.
- The illumination-unet model uses Sobel texture features, not a full color-difference histogram method.

## Development

```bash
uv run pytest
uv run ruff check .
python -m compileall src data
```
