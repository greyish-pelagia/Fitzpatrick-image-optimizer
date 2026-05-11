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

### Pre-trained model weights

To run inference on the proposed models, download the pre-trained model weights to `./models/` or train your own on your own dataset

You can download our pre-trained model weights to use in testing from:
- [DeepLPF weights](https://drive.google.com/file/d/1jHoWKzvn4XPR-o4hmhq_G45eKlEbC57r/view?usp=sharing)
- [Retinex-Fuzzy-CNN](https://drive.google.com/file/d/1zlCs7tr93TvBxbGWz7aaJE87zn8KFJBP/view?usp=sharing)

## 1. Dataset Used
**Dataset:** Fitzpatrick17k Dataset
The dataset comprises roughly 17,000 high-quality medical images annotated with the Fitzpatrick skin-type scale ranging from 1 to 6. The original, well-lit Fitzpatrick imagery acts as our Ground Truth ($Y$) labels during model training.

## 2. Data Preprocessing (Synthetic Degradation)
Because natural "Before-and-After" pairs are unavailable, we generate our training data dynamically. We pass the Ground Truth ($Y$) into a Synthetic Degradation Pipeline to artificially generate our degraded inputs ($X$):

1. **Random Gamma Shift:** Simulates camera over/underexposure. Applies a power-law transformation $X = Y^\gamma$ by sampling $\gamma \sim U(0.4, 2.5)$.
2. **Random Color Cast:** Simulates improper clinical white balances. Multiplies Red, Green, and Blue channels independently using values sampled from $U(0.75, 1.25)$.
3. **Histogram Denormalization:** Simulates flattened lighting and low contrast by mathematically compressing the dynamic range.

## 3. The Models
We developed two separate automated end-to-end architectures:

### Architecture 1: Parameter-Driven DeepLPF
- **Concept:** A Convolutional Neural Network predicts optimal mathematical parameters for conventional, physics-based image filters.
- **Method:** Accepts a $4 \times H \times W$ concatenated tensor (the RGB image plus the normalized Fitzpatrick scalar). A modified ResNet50 backbone extracts features, feeding them into an MLP head that predicts exactly 76 mathematical parameters to control Graduated, Elliptical, and Polynomial filters.

### Architecture 2: Hybrid Retinex-Fuzzy-CNN
- **Concept:** An end-to-end, fully differentiable pixel-level translation pipeline.
- **Methodology:**
   - **Stage 1 (Illumination Normalization):** A RetinexDIP U-Net extracts dynamic lighting properties. It conditions the network on the Fitzpatrick scale via a Featurewise Linear Modulation (FiLM) bottleneck, isolating Reflectance and Illumination maps.
   - **Stage 2 (Deterministic Feature Extraction):** A frozen, parameter-free Color Difference Histogram (CDH) module utilizes Sobel operations to extract deterministic edge and texture maps.
   - **Stage 3 (Fuzzy-CNN Refinement):** A shallow CNN utilizing Fuzzy Logic Membership functions rebuilds the final image. **Critically, this stage utilizes a Skip Connection**, concatenating the raw original input image ($X$) directly alongside the CDH and Reflectance maps to perfectly preserve high-frequency micro-textures and structural integrity.

## 4. Expected Results for Downstream Medical Images Classification
Standard Medical Computer Vision classification networks typically underperform heavily when analyzing skin lesions under defective, shifting clinical lighting environments. By normalizing raw clinical imagery through either **Architecture 1** or **Architecture 2** prior to diagnostic classification, you can expect:
* Elimination of illumination acquisition biases associated with darker/lighter Fitzpatrick skin types.
* Substantial boosts in diagnostic **Accuracy**, **AUC-ROC**, and **F1 metrics** across otherwise degraded data inputs.

---

## 5. Usage & Execution Instructions

### Architecture 1: Parameter-Driven DeepLPF

**Features Included:**
1. **Standardized DeepLPF Architecture:**
   - Modified the standard `ResNet50` backbone to accept the expanded $4 \times H \times W$ tensor format (3 for visual RGB bands, and an injected feature layer for the `s_norm` scalar representation).
   - Replaced the standard ImageNet classifier with a custom MLP output head specifically configured to regress 76 variables (Graduated, Elliptical, and Polynomial parameters).
   - Applied differentiable mathematical parameter implementations equivalent to synthetic exposure filters to maintain PyTorch `autograd` flow.
2. **Dataset & Training Loop Parameters:**
   - Command line arguments dictate `--epochs` for configurable runs.
   - The `--scale_dataset <int>` parameter samples chunks of data for quick debugging before full batch rendering. Use `0` to run across the full dataset.
3. **Hardware Fallback Infrastructure:**
   - Fully utilizes `cuda`, falls back to Apple Silicon `mps`, and defaults to standard `cpu` automatically without errors.
  
**Execution Commands:**

*Quick trial (e.g., 1 epoch over a small subset of 100 images):*
```bash
uv run python src/train_deeplpf.py --scale_dataset 100 --epochs 1 --batch_size 8
```

*Full training capability:*
```bash
uv run python src/train_deeplpf.py --scale_dataset 0 --epochs 50 --batch_size 32
```

*Run trained model evaluation:*
```bash
uv run python src/evaluate_deeplpf.py --num_samples 100 --batch_size 8
```

- [residual-filter weights](https://drive.google.com/file/d/1jHoWKzvn4XPR-o4hmhq_G45eKlEbC57r/view?usp=sharing)
- [illumination-unet weights](https://drive.google.com/file/d/1zlCs7tr93TvBxbGWz7aaJE87zn8KFJBP/view?usp=sharing)

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
