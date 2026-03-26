# Fitzpatrick Image Optimizer

This repository implements two advanced PyTorch-based neural network pipelines designed to automatically restore poorly lit, color-cast, and low-contrast medical imagery back to uniform, optimal diagnostic lighting conditions. By conditioning reconstructions directly on Fitzpatrick skin type scalars, these architectures effectively correct visual acquisition biases.

## Results showcase

![Fitzpatric-image-optimizer results](comparison_grid.png)

#### DeepLPF evaluation metrics

- Total Test Samples Validated : 5000
- Mean Absolute Error (L1)     : 0.02464 (lower is better)
- Mean Squared Error (MSE)     : 0.00118 (lower is better)
- Peak Signal-Noise (PSNR)     : 29.30 dB (higher is better)
- Structural Similarity (SSIM) : 0.9593 (max 1.0, higher is better)
- Average preprocessing time per image: 0.27 seconds

#### Retinex-Fuzzy-CNN evaluation metrics

- Total Test Samples Validated : 5000
- Mean Absolute Error (L1)     : 0.07747 (lower is better)
- Mean Squared Error (MSE)     : 0.01013 (lower is better)
- Peak Signal-Noise (PSNR)     : 19.96 dB (higher is better)
- Structural Similarity (SSIM) : 0.9277 (max 1.0, higher is better)
- Average preprocessing time per image: 0.31 seconds


### Pre-trained model weights

To run inference on the proposed models, download the pre-trained model weights to `./models/` or train your own on your own dataset

You can download our pre-trained model weights to use in testing from:
- [DeepLPF weights](https://drive.google.com/file/d/1jHoWKzvn4XPR-o4hmhq_G45eKlEbC57r/view?usp=sharing)
- Retinex-Fuzzy-CNN WIP

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

*Run inference on new images:*
```bash
uv run python src/infer_deeplpf.py --model_path models/deeplpf.pth --csv_path data/labels.csv
```
*(Optional Inference Parameters: `--max_samples 5` to limit processing, `--output_dir output` to specify the save location).*

---

### Architecture 2: Hybrid Retinex-Fuzzy-CNN

**Design Implementations Included:**
1. **Retinex FiLM U-Net (Stage 1):** Built the complete structural encoder-decoder U-Net, including isolating the conditional bottleneck via a **Featurewise Linear Modulation (FiLM)** MLP block mapping `S` into the dynamic `gamma` and `beta` tensor adjustments to reconstruct the Illumination ($L$) and Reflectance ($R$) layers.
2. **Deterministic CDH Extractor (Stage 2):** Engineered the explicit PyTorch convolutional block with fixed, mathematically defined static Sobel-driven edge parameters running without trainable tracking (`requires_grad=False`).
3. **Fuzzy-CNN Mapping (Stage 3):** Constructed the parameter-free Fuzzy Membership logic, replacing standard ReLUs on a shallow CNN to handle residual translation outputs explicitly.
4. **Matched Modularity:** Carries over the exact documentation styles, CLI configuration logic, SSIM constraints, and automated hardware scaling parameters found in the DeepLPF pipeline.

**Execution Commands:**

*Quick trial training:*
```bash
uv run python src/train_retinex.py --scale_dataset 100 --epochs 5 --batch_size 4
```

*Full training capability:*
```bash
uv run python src/train_retinex.py --scale_dataset 0 --epochs 50 --batch_size 32
```

*Run trained model evaluation:*
```bash
uv run python src/evaluate_retinex.py --num_samples 100 --batch_size 8
```

*Run inference on new images:*
```bash
uv run python src/infer_retinex.py --model_path models/hybrid_retinex.pth --csv_path data/labels.csv
```
*(Optional Inference Parameters: `--max_samples 5` to limit processing, `--output_dir output` to specify the save location).*
