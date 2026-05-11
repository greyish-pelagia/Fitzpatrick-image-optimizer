# Model Architecture Notes

## Name Mapping

| Original name            | Current package/model name            | Python class                         | CLI model key       |
| ------------------------ | ------------------------------------- | ------------------------------------ | ------------------- |
| Parameter-Driven DeepLPF | Parameter-Conditioned Residual Filter | `ParameterConditionedResidualFilter` | `residual-filter`   |
| Hybrid Retinex-Fuzzy-CNN | Illumination-Guided U-Net             | `IlluminationGuidedUNet`             | `illumination-unet` |

The original names are retained for continuity with the project history. The current names are more precise about what the code implements.

## Pipeline 1: Parameter-Driven DeepLPF / Parameter-Conditioned Residual Filter

The current `residual-filter` model is a DeepLPF-inspired architecture, not a full implementation of the original DeepLPF filter family.

### Architecture

1. **Input conditioning**
   - Input image tensor: `3 x H x W`.
   - Fitzpatrick scale is normalized from `[1, 6]` to `[0, 1]`.
   - A learned scalar embedding is spatially expanded and concatenated with the RGB image, producing a `4 x H x W` input tensor.

2. **Feature extractor**
   - Uses a ResNet50 backbone.
   - The first convolution is adapted from 3 input channels to 4 input channels.
   - When pretrained weights are enabled, RGB weights are copied from the original ResNet50 stem and the fourth channel is initialized from the mean RGB stem weight.

3. **Parameter head**
   - The ResNet feature vector feeds a small MLP.
   - The MLP predicts 76 latent parameters.
   - Those parameters are split into gradient-like, elliptical-like, and polynomial-like groups for compatibility with the DeepLPF framing.

4. **Image reconstruction**
   - The implementation applies a simplified differentiable residual transform:
     - polynomial terms use the first RGB polynomial coefficients,
     - gradient and elliptical groups influence global scale and shift terms,
     - the residual is added to the original input and clamped to `[0, 1]`.
   - This preserves autograd and keeps the module trainable end to end, but it should be described as a residual-filter baseline rather than exact DeepLPF.

### Current commands

```bash
uv run fitzopt train --model residual-filter --csv_path data/labels.csv --max_samples 100 --epochs 1 --batch_size 8
uv run fitzopt evaluate --model residual-filter --model_path models/residual-filter.pth --csv_path data/labels.csv --split test --metrics_json results/residual-filter-test.json
uv run fitzopt infer --model residual-filter --model_path models/residual-filter.pth --csv_path data/labels.csv --output_dir results/residual-filter
```

## Pipeline 2: Hybrid Retinex-Fuzzy-CNN / Illumination-Guided U-Net

The current `illumination-unet` model keeps the spirit of the original Retinex/Fuzzy-CNN idea, but the code is now named around the concrete implementation: FiLM-conditioned U-Net, Sobel texture features, and a refinement CNN.

### Architecture

1. **Illumination-guided U-Net**
   - A U-Net-style encoder/decoder consumes the RGB image.
   - Fitzpatrick conditioning is injected at the bottleneck through a FiLM layer.
   - The U-Net predicts:
     - a single-channel illumination map,
     - a three-channel reflectance-like map.

2. **Sobel texture extractor**
   - The previous "CDH" wording has been narrowed.
   - The current implementation uses fixed, non-trainable Sobel filters grouped per RGB channel.
   - These filters produce six texture/edge maps from the reflectance-like output.

3. **Refinement CNN**
   - The refinement stage concatenates:
     - reflectance-like output,
     - Sobel texture maps,
     - original input image.
   - A shallow CNN with a Gaussian membership activation predicts the final normalized RGB output.
   - The illumination map is returned for evaluation or regularization use, but the final reconstruction is produced by the refinement branch.

### Current commands

Preferred package CLI:

```bash
uv run fitzopt train --model illumination-unet --csv_path data/labels.csv --max_samples 100 --epochs 5 --batch_size 4
uv run fitzopt evaluate --model illumination-unet --model_path models/illumination-unet.pth --csv_path data/labels.csv --split test --metrics_json results/illumination-unet-test.json
uv run fitzopt infer --model illumination-unet --model_path models/illumination-unet.pth --csv_path data/labels.csv --output_dir results/illumination-unet
```

## Evaluation Expectations

Evaluation uses deterministic train/validation/test split assignment when a CSV does not already contain a `split` column. Reports should include:

- model metrics grouped by Fitzpatrick scale,
- identity-baseline metrics grouped by Fitzpatrick scale,
- split name,
- sample count,
- command used to produce the report.

See `docs/results-schema.md` for the JSON output shape.

## Demo Dataset Path

A fresh clone can generate a tiny synthetic demo dataset without downloading the full image corpus:

```bash
uv run fitzopt create-demo-data --output_dir demo_assets --count 6
```

The generated data is intended only for smoke testing the pipeline. It is not evidence of model quality.
