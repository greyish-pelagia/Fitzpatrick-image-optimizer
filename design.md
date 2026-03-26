### The Training Loop (Synthetic Degradation Strategy)
To train either pipeline effectively without natural "before/after" pairs, you must dynamically generate aligned data using synthetic degradation.

1.  **Forward Pass:** Pass the artificially degraded image ($X$) and the ground-truth Fitzpatrick scalar ($S$) into your network.
2.  **Output:** The network predicts and outputs the restored image ($\hat{Y}$).
3.  **Loss Calculation:** Calculate the error directly between the network's output ($\hat{Y}$) and the original, untouched image ($Y$). Use a combination of **L1 Loss** (for color/luminance accuracy) and **SSIM Loss** (Structural Similarity, to preserve skin textures and lesions).
4.  **Backpropagation:** The network updates its weights to learn exactly how to reverse the randomized gamma, contrast, and color shifts applied during degradation.

---

### Pipeline 1: Parameter-Driven DeepLPF (Simple)
**Concept:** A CNN predicts optimal mathematical parameters for conventional filters, maintaining an unbroken computational graph for end-to-end training.

**1. Input Representation:**
* **Image Tensor ($X$):** $3 \times H \times W$ (RGB).
* **Fitzpatrick Scalar ($S$):** Normalized to $[0, 1]$.
* **Fusion:** Pass $S$ through a linear embedding layer, spatially expand it to $1 \times H \times W$, and concatenate with $X$ to create a **$4 \times H \times W$ input tensor**.

**2. Backbone Architecture (PyTorch):**
* **Model:** `torchvision.models.resnet50(pretrained=True)`.
* **Modification:** Replace the first convolution layer (`conv1`) to accept 4 input channels instead of 3.
* **Regression Head:** Replace the final Fully Connected (FC) layer with a Multi-Layer Perceptron (MLP) ending in a linear layer with $\approx 76$ output nodes (8 for Graduated + 8 for Elliptical + 60 for Polynomial parameters).

**3. Output Inference Architecture:**
* **Forward Pass:** Tensor $[X, S] \rightarrow$ ResNet50 $\rightarrow$ Parameter Vector $P$.
* **Slicing:** Split $P$ into $P_{grad}$, $P_{ellip}$, and $P_{poly}$.
* **Conventional Filtering:** Pass the original Image $X$ and the parameters $P$ into **custom PyTorch tensor operations** (e.g., `torch.mul`, `torch.add`) representing the mathematical formulas of the DeepLPF filters. *Note: Do not use OpenCV/NumPy here, as it will break differentiability.*
* **Final Output:** A single enhanced image tensor $Y$.

---

### Pipeline 2: Hybrid Retinex-Fuzzy-CNN (Advanced)
**Concept:** An end-to-end, fully differentiable pixel-level translation pipeline using chained neural networks and deterministic feature extraction.

**1. Illumination Normalization (Stage 1):**
* **Architecture:** RetinexDIP implemented via a U-Net.
* **Input:** Image tensor $X$ ($3 \times H \times W$).
* **Conditioning:** Inject the Fitzpatrick scalar $S$ into the bottleneck layer of the U-Net via Featurewise Linear Modulation (FiLM) to guide the expected base reflectance.
* **Output:** Decomposed Illumination map ($L$) and Reflectance map ($R$). The adjusted reflectance $R_{adj}$ is passed forward.

**2. Deterministic Feature Extraction (Stage 2):**
* **Operation:** Apply the Color Difference Histogram (CDH) algorithm to extract edges and textures. *Note: This must be implemented using **fixed, non-trainable PyTorch convolutional kernels** (`torch.nn.Conv2d` with `requires_grad=False`) to maintain gradient flow back to Stage 1.*
* **Input:** $R_{adj}$.
* **Output:** Edge and texture feature maps ($T$).

**3. Fuzzy-CNN Refinement (Stage 3):**
* **Architecture:** A shallow CNN (e.g., 3-5 layers). The first layer replaces standard ReLUs with parameter-free Fuzzy Logic Membership Functions (e.g., Gaussian curves implemented as custom PyTorch activation layers) to handle residual noise.
* **Input:** Concatenated tensor of $[R_{adj}, T]$.
* **Output:** Reconstructed $3 \times H \times W$ RGB image.

**4. Output Inference Architecture:**
* Pass $X$ and $S$ into the Retinex U-Net $\rightarrow$ Obtain $R_{adj}$.
* Compute CDH maps $T$ from $R_{adj}$ via frozen PyTorch convolutions.
* Concatenate $R_{adj}$ and $T \rightarrow$ Pass through Fuzzy-CNN.
* **Final Output:** Fully normalized pixel tensor $Y$.