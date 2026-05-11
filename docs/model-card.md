# Model Card

## Project

Fitzpatrick Image Optimizer is an experimental image restoration demo for synthetic illumination shifts in dermatology-style images.

## Intended Use

- Portfolio demonstration of PyTorch modeling and ML engineering workflow.
- Research-style exploration of image normalization under synthetic degradation.
- Non-clinical educational experiments.

## Out-of-Scope Use

- Medical diagnosis.
- Clinical decision support.
- Claims of improved diagnostic accuracy.
- Claims of fairness or bias elimination without downstream validation.

## Models

### residual-filter

ResNet-conditioned residual image filter inspired by parameter-prediction approaches. The implementation predicts 76 latent parameters but applies a simplified differentiable residual transform.

### illumination-unet

FiLM-conditioned U-Net that predicts illumination and reflectance-like maps, extracts fixed Sobel texture features, and refines the output through a shallow CNN.

## Data

Training uses paired synthetic examples generated from available images and Fitzpatrick scale metadata. The synthetic degradation applies gamma shift, channel-wise color cast, and contrast compression.

## Evaluation

Required report fields:

- split name
- sample count
- identity baseline metrics
- model metrics
- metrics grouped by Fitzpatrick scale

## Risks and Limitations

Synthetic image restoration metrics do not prove clinical usefulness. Performance can vary across image source, diagnosis category, acquisition conditions, and Fitzpatrick scale groups.
