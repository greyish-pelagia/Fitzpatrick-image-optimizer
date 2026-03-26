## Pipeline 1: Parameter-Driven DeepLPF (Simple)

### **Features included**
1. **Standardized DeepLPF Architecture:**
   - Modified the standard loaded `ResNet50` architecture backend to accept the expanded $4 \times H \times W$ tensor format securely (3 for visual RGB bands, and an injected feature layer mapping for the `s_norm` mapped representation).
   - Dropped the Standard ImageNet classifier and added a custom MLP output head specifically configured for capturing 76 variables (Graduated, Elliptical, and Polynomial control).
   - Applied differentiable mathematical parameter implementations equivalent to the requested synthetic exposure filters without breaking the backward PyTorch `autograd` flow.

2. **Dataset & Training Loop Parameters:**
   - Command line arguments to dictate `--epochs` for quick runs natively.
   - Command line arguments to handle `--scale_dataset <int>` to sample only chunks of data (excellent for quick debugging before full batch rendering runs). Simply use `0` to run across the full ~16k training examples natively.

3. **Fallback Infrastructure:**
   - Fully utilizes `cuda`, falls back to Apple Silicon `mps`, and defaults backwards directly onto the standard `cpu` without errors. *(I initiated an automated short 8 sample run test on it in the background, and it verified perfectly securely scaling onto `mps` context properly).*

### **How to run the pipeline**

**Command mapping for a quick trial (e.g. 1 epoch over a small subset of 100 images):**
```bash
uv run python src/train_deeplpf.py --scale_dataset 100 --epochs 1 --batch_size 8
```

**Command mapping for full training capability:**
```bash
uv run python src/train_deeplpf.py --scale_dataset 0 --epochs 50 --batch_size 32
```

**Run trained model evaluation:**
```bash
uv run python src/evaluate_deeplpf.py --num_samples 100 --batch_size 8
```

**Command to run inference:**
```bash
uv run python src/infer_deeplpf.py --model_path models/deeplpf.pth --csv_path data/labels.csv
```

- with optional parameters:
`--max_samples 5` - limit number of samples to process
`--output_dir output` - specify output directory
`--device cuda` - specify device to use (cuda, mps, cpu)
`--batch_size 4` - specify batch size

```bash
uv run python src/infer_deeplpf.py --model_path models/deeplpf.pth --csv_path data/labels.csv --max_samples 5
```

---

### Pipeline 2: Hybrid Retinex-Fuzzy-CNN (Advanced)

### **Design Implementations included:**
1. **Retinex FiLM U-Net (Stage 1):** Built the complete structural encoder-decoder U-Net directly tracking your requirements, including isolating the conditional bottleneck via a **Featurewise Linear Modulation (FiLM)** MLP block mapping `S` into the dynamic `gamma` and `beta` tensor adjustments to reconstruct the Illumination ($L$) and Reflectance ($R$) layers.
2. **Deterministic CDH Extractor (Stage 2):** Engineered the explicit PyTorch convolutional block with securely fixed, mathematically defined static Sobel-driven edge parameters running without trainable tracking (using `requires_grad=False`).
3. **Fuzzy-CNN Mapping (Stage 3):** Constructed the parameter-free Fuzzy Memebership logic replacing your standard ReLUs structurally on a shallow CNN to handle residual translation outputs explicitly accurately.
4. **Matched Modularity:** Faithfully carried over the exact documentation styles, inline structural commentary formats, CLI configuration logic, SSIM constraints, and smart natively automated Apple Silicon (`mps`) / `cuda` scaling parameters found in your prior scripts naturally.

### **How to Trigger Pipeline 2 Training**
You can deploy it using the equivalent parameter arguments exactly like your existing DeepLPF models safely:

```bash
uv run python src/train_retinex.py --scale_dataset 100 --epochs 5 --batch_size 4
```

**Command mapping for full training capability:**
```bash
uv run python src/train_retinex.py --scale_dataset 0 --epochs 50 --batch_size 32
```

**Run trained model evaluation:**
```bash
uv run python src/evaluate_retinex.py --num_samples 100 --batch_size 8
```

**Command to run inference:**
```bash
uv run python src/infer_retinex.py --model_path models/hybrid_retinex.pth --csv_path data/labels.csv
```

- with optional parameters:
`--max_samples 5` - limit number of samples to process
`--output_dir output` - specify output directory
`--device cuda` - specify device to use (cuda, mps, cpu)
`--batch_size 4` - specify batch size

```bash
uv run python src/infer_retinex.py --model_path models/hybrid_retinex.pth --csv_path data/labels.csv --max_samples 5
```