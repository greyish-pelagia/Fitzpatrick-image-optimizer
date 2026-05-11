# Results Schema

Evaluation writes JSON with this shape:

```json
{
  "model": "residual-filter",
  "split": "test",
  "count": 100,
  "model_grouped": {
    "1": {"count": 10, "l1": 0.1, "mse": 0.02, "psnr": 20.0, "ssim": 0.9}
  },
  "identity_baseline_grouped": {
    "1": {"count": 10, "l1": 0.2, "mse": 0.04, "psnr": 17.0, "ssim": 0.8}
  }
}
```

Public README metrics must cite the model checkpoint, dataset split, sample count, and command used to produce the file.
