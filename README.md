# Fitzpatrick Image Optimizer

This repository implements two advanced PyTorch-based neural network pipelines designed to automatically restore poorly lit, color-cast, and low-contrast medical imagery back to uniform, optimal diagnostic lighting conditions. By conditioning reconstructions directly on Fitzpatrick skin type scalars, these architectures effectively correct visual acquisition biases.

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
- **Methodology:** - **Stage 1 (Illumination Normalization):** A RetinexDIP U-Net extracts dynamic lighting properties. It conditions the network on the Fitzpatrick scale via a Featurewise Linear Modulation (FiLM) bottleneck, isolating Reflectance and Illumination maps.
   - **Stage 2 (Deterministic Feature Extraction):** A frozen, parameter-free Color Difference Histogram (CDH) module utilizes Sobel operations to extract deterministic edge and texture maps.
   - **Stage 3 (Fuzzy-CNN Refinement):** A shallow CNN utilizing Fuzzy Logic Membership functions rebuilds the final image. **Critically, this stage utilizes a Skip Connection**, concatenating the raw original input image ($X$) directly alongside the CDH and Reflectance maps to perfectly preserve high-frequency micro-textures and structural integrity.

## 4. Expected Results for Downstream Medical Images Classification
Standard Medical Computer Vision classification networks typically underperform heavily when analyzing skin lesions under defective, shifting clinical lighting environments. By normalizing raw clinical imagery through either **Pipeline 1** or **Pipeline 2** prior to diagnostic classification, you can expect:
* Elimination of illumination acquisition biases associated with darker/lighter Fitzpatrick skin types.
* Substantial boosts in diagnostic **Accuracy**, **AUC-ROC**, and **F1 metrics** across otherwise degraded data inputs.