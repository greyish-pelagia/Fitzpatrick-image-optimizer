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