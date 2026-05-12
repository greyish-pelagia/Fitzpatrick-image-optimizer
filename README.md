# Fitzpatrick Image Optimizer

A PyTorch-based CLI project for Fitzpatrick-conditioned illumination normalization on dermatology-style images, with a reproducible workflow for synthetic degradation, training, evaluation, grouped metrics, and inference.

This is a research-style project, not a clinical diagnostic system. It demonstrates how image restoration models can be conditioned on Fitzpatrick-scale metadata and evaluated with both aggregate and grouped metrics.

## Results

### Baseline comparison

The models were evaluated against an identity / no-op baseline, which returns the degraded input image unchanged. This checks whether learned restoration improves over doing nothing.

| Method                              |        L1 ↓ |       MSE ↓ |    PSNR ↑ |     SSIM ↑ |
| ----------------------------------- | ----------: | ----------: | --------: | ---------: |
| Baseline                            |     0.14564 |     0.03204 |     15.16 |     0.8218 |
| DeepLPF-inspired residual-filter    |     0.11832 |     0.02226 | **16.86** | **0.8701** |
| FiLM-conditioned illumination U-Net | **0.11559** | **0.02197** |     16.74 |     0.8134 |

> *Note: PSNR is averaged from batch-level PSNR values rather than recomputed from the final mean MSE. Because of that, method ranking may differ slightly between mean MSE and mean PSNR.*

The residual-filter model performs best overall, improving SSIM from 0.8218 to 0.8701 and worst-group SSIM from 0.8193 to 0.8679. The illumination U-Net reduces pixel-level error but underperforms the identity baseline on SSIM, so it is treated as an experimental architecture rather than the primary model.

#### Grouped Fitzpatrick evaluation

| Method                              | FST I-II SSIM | FST III-IV SSIM | FST V-VI SSIM | Worst-group SSIM |
| ----------------------------------- | ------------: | --------------: | ------------: | ---------------: |
| Baseline                            |        0.8235 |          0.8193 |        0.8224 |           0.8193 |
| DeepLPF-inspired residual-filter    |    **0.8718** |      **0.8679** |    **0.8699** |           0.8679 |
| FiLM-conditioned illumination U-Net |        0.8147 |          0.8116 |        0.8133 |           0.8116 |

#### Notes

- The identity / no-op baseline returns the degraded input unchanged and is evaluated against the clean target image.
- Metrics are reported on the same deterministic split for all methods.
- Grouped metrics are computed by Fitzpatrick scale groups: I-II, III-IV, and V-VI.
- The residual-filter is the primary model because it improves both aggregate SSIM and worst-group SSIM.
- The illumination U-Net is retained as an experimental architecture: it improves L1/MSE but does not improve SSIM.

### Processed images demo

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
