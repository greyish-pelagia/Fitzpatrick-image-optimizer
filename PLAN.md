# Public Portfolio Demo Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn this repository into a credible primary portfolio demo for a Mid AI/ML Engineer by making it reproducible, importable, testable, honestly documented, and runnable from a fresh clone.

**Architecture:** Convert script-level code into an installable `fitzpatrick_optimizer` package with clear modules for data, preprocessing, models, metrics, training, evaluation, and inference. Keep the current two model ideas, but rename/describe them according to what the code actually implements, add deterministic splits and grouped metrics, and provide a tiny generated demo dataset so the public quickstart works without private local data.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, OpenCV, pandas, NumPy, pytest, ruff, uv, GitHub Actions.

---

## Portfolio Bar

This project is meant to be the first repo a recruiter or hiring manager opens. The implementation should signal:

- ML judgment: honest claims, baselines, holdout metrics, group metrics by Fitzpatrick scale, reproducibility.
- Engineering maturity: importable package, tests, CI, CLI commands, typed small modules, no hidden local paths.
- Demo polish: one fresh-clone command path, small sample assets, result images, clear model cards and limitations.
- Safety and ethics: no unsupported clinical/diagnostic claims, explicit non-clinical intended use.

## File Structure

Create this package structure while keeping thin compatibility wrappers for existing script names:

```text
src/
  fitzpatrick_optimizer/
    __init__.py
    cli.py
    config.py
    data.py
    demo.py
    evaluate.py
    imaging.py
    infer.py
    metrics.py
    randomness.py
    splits.py
    train.py
    models/
      __init__.py
      illumination_unet.py
      residual_filter.py
  evaluate_deeplpf.py
  evaluate_retinex.py
  infer_deeplpf.py
  infer_retinex.py
  train_deeplpf.py
  train_retinex.py
tests/
  conftest.py
  test_cli.py
  test_data.py
  test_demo.py
  test_metrics.py
  test_models.py
  test_splits.py
docs/
  model-card.md
  reproducibility.md
  results-schema.md
.github/
  workflows/
    ci.yml
```

The old `src/train_deeplpf.py`, `src/train_retinex.py`, `src/evaluate_*.py`, and `src/infer_*.py` files become wrappers that import from the package. This preserves README commands while proving the code is package-quality.

---

### Task 1: Tooling, Package Skeleton, and Import Smoke Tests

**Files:**
- Modify: `pyproject.toml`
- Modify: `.python-version`
- Create: `src/fitzpatrick_optimizer/__init__.py`
- Create: `src/fitzpatrick_optimizer/models/__init__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing import and metadata tests**

Create `tests/test_cli.py`:

```python
import importlib.metadata


def test_package_imports():
    import fitzpatrick_optimizer

    assert fitzpatrick_optimizer.__version__ == "0.1.0"


def test_distribution_metadata_has_real_description():
    metadata = importlib.metadata.metadata("fitzpatrick-image-optimizer")

    assert metadata["Name"] == "fitzpatrick-image-optimizer"
    assert "experimental" in metadata["Summary"].lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL because `pytest` is not configured and `fitzpatrick_optimizer` does not exist.

- [ ] **Step 3: Add package metadata and package skeleton**

Replace `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fitzpatrick-image-optimizer"
version = "0.1.0"
description = "Experimental Fitzpatrick-conditioned image illumination normalization demo."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "matplotlib>=3.8",
    "numpy>=1.26",
    "opencv-python>=4.9",
    "pandas>=2.2",
    "torch>=2.2",
    "torchvision>=0.17",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "ruff>=0.8",
]

[project.scripts]
fitzopt = "fitzpatrick_optimizer.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Replace `.python-version` with:

```text
3.11
```

Create `src/fitzpatrick_optimizer/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `src/fitzpatrick_optimizer/models/__init__.py`:

```python
from fitzpatrick_optimizer.models.illumination_unet import IlluminationGuidedUNet
from fitzpatrick_optimizer.models.residual_filter import ParameterConditionedResidualFilter

__all__ = ["IlluminationGuidedUNet", "ParameterConditionedResidualFilter"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version src/fitzpatrick_optimizer tests/test_cli.py
git commit -m "chore: add package skeleton and test tooling"
```

---

### Task 2: Shared Device, Seed, and Image Utilities

**Files:**
- Create: `src/fitzpatrick_optimizer/config.py`
- Create: `src/fitzpatrick_optimizer/randomness.py`
- Create: `src/fitzpatrick_optimizer/imaging.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing utility tests**

Create `tests/test_data.py`:

```python
import numpy as np
import pytest

from fitzpatrick_optimizer.imaging import normalize_fitzpatrick_scale, to_chw_float


def test_normalize_fitzpatrick_scale_maps_1_to_0_and_6_to_1():
    assert normalize_fitzpatrick_scale(1) == 0.0
    assert normalize_fitzpatrick_scale(6) == 1.0
    assert normalize_fitzpatrick_scale(3) == pytest.approx(0.4)


def test_normalize_fitzpatrick_scale_rejects_invalid_values():
    with pytest.raises(ValueError, match="Fitzpatrick scale must be between 1 and 6"):
        normalize_fitzpatrick_scale(0)


def test_to_chw_float_converts_uint8_rgb_to_float_tensor_layout():
    image = np.array(
        [[[0, 128, 255], [255, 128, 0]]],
        dtype=np.uint8,
    )

    result = to_chw_float(image)

    assert result.shape == (3, 1, 2)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == 0.0
    assert result[1, 0, 0] == pytest.approx(128 / 255)
    assert result[2, 0, 0] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_data.py -v
```

Expected: FAIL with missing `fitzpatrick_optimizer.imaging`.

- [ ] **Step 3: Implement utilities**

Create `src/fitzpatrick_optimizer/config.py`:

```python
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Create `src/fitzpatrick_optimizer/randomness.py`:

```python
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
```

Create `src/fitzpatrick_optimizer/imaging.py`:

```python
from pathlib import Path

import cv2
import numpy as np


def normalize_fitzpatrick_scale(value: int | float) -> float:
    scale = float(value)
    if scale < 1.0 or scale > 6.0:
        raise ValueError(f"Fitzpatrick scale must be between 1 and 6, got {value!r}")
    return (scale - 1.0) / 5.0


def read_rgb_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_chw_float(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")
    return image_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0


def write_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise OSError(f"Could not write image: {path}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/config.py src/fitzpatrick_optimizer/randomness.py src/fitzpatrick_optimizer/imaging.py tests/test_data.py
git commit -m "feat: add shared image and runtime utilities"
```

---

### Task 3: Validated Dataset and Deterministic Preprocessing

**Files:**
- Create: `src/fitzpatrick_optimizer/data.py`
- Modify: `data/training_preprocess.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Add failing dataset and preprocessing tests**

Append to `tests/test_data.py`:

```python
from pathlib import Path

import cv2
import pandas as pd
import torch

from fitzpatrick_optimizer.data import (
    FitzpatrickImageDataset,
    SyntheticDegradationConfig,
    degrade_image,
    validate_records,
)


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((8, 8, 3), color, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


def test_validate_records_reports_missing_images(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "training_image": str(tmp_path / "missing-input.jpg"),
                "ground_truth_image": str(tmp_path / "missing-target.jpg"),
                "Fitzpatrick scale": 3,
            }
        ]
    ).to_csv(csv_path, index=False)

    report = validate_records(csv_path)

    assert report.total_rows == 1
    assert report.valid_rows == 0
    assert len(report.errors) == 2
    assert "missing-input.jpg" in report.errors[0]


def test_dataset_raises_when_csv_has_no_valid_pairs(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "training_image": str(tmp_path / "missing-input.jpg"),
                "ground_truth_image": str(tmp_path / "missing-target.jpg"),
                "Fitzpatrick scale": 3,
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No valid image pairs"):
        FitzpatrickImageDataset(csv_path)


def test_dataset_returns_tensor_pair_and_normalized_scale(tmp_path):
    input_path = tmp_path / "input.jpg"
    target_path = tmp_path / "target.jpg"
    _write_rgb(input_path, (20, 40, 60))
    _write_rgb(target_path, (60, 40, 20))
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "training_image": str(input_path),
                "ground_truth_image": str(target_path),
                "Fitzpatrick scale": 6,
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset = FitzpatrickImageDataset(csv_path, image_size=(16, 16))

    input_tensor, target_tensor, scale = dataset[0]
    assert input_tensor.shape == (3, 16, 16)
    assert target_tensor.shape == (3, 16, 16)
    assert scale.shape == (1,)
    assert scale.item() == 1.0
    assert input_tensor.dtype == torch.float32


def test_degrade_image_is_deterministic_with_seed():
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    config = SyntheticDegradationConfig(seed=123)

    first = degrade_image(image, config)
    second = degrade_image(image, config)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_data.py -v
```

Expected: FAIL with missing `fitzpatrick_optimizer.data`.

- [ ] **Step 3: Implement validated dataset and deterministic degradation**

Create `src/fitzpatrick_optimizer/data.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fitzpatrick_optimizer.imaging import (
    normalize_fitzpatrick_scale,
    read_rgb_image,
    resize_rgb,
    to_chw_float,
)


@dataclass(frozen=True)
class ValidationReport:
    total_rows: int
    valid_rows: int
    errors: list[str]


@dataclass(frozen=True)
class SyntheticDegradationConfig:
    gamma_range: tuple[float, float] = (0.4, 2.5)
    color_weight_range: tuple[float, float] = (0.75, 1.25)
    contrast_alpha_range: tuple[float, float] = (0.5, 0.8)
    seed: int = 42


REQUIRED_COLUMNS = ("training_image", "ground_truth_image", "Fitzpatrick scale")


def validate_records(csv_path: str | Path) -> ValidationReport:
    df = pd.read_csv(csv_path)
    errors: list[str] = []
    valid_rows = 0

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            errors.append(f"Missing required column: {column}")

    if errors:
        return ValidationReport(total_rows=len(df), valid_rows=0, errors=errors)

    for row_index, row in df.iterrows():
        row_errors: list[str] = []
        for column in ("training_image", "ground_truth_image"):
            path = Path(str(row[column]))
            if not path.exists():
                row_errors.append(f"row {row_index}: image does not exist: {path}")
        try:
            normalize_fitzpatrick_scale(row["Fitzpatrick scale"])
        except ValueError as exc:
            row_errors.append(f"row {row_index}: {exc}")

        if row_errors:
            errors.extend(row_errors)
        else:
            valid_rows += 1

    return ValidationReport(total_rows=len(df), valid_rows=valid_rows, errors=errors)


def _valid_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    valid_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        input_path = Path(str(row["training_image"]))
        target_path = Path(str(row["ground_truth_image"]))
        try:
            normalize_fitzpatrick_scale(row["Fitzpatrick scale"])
        except ValueError:
            continue
        if input_path.exists() and target_path.exists():
            valid_rows.append(row.to_dict())
    return pd.DataFrame(valid_rows)


class FitzpatrickImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        max_samples: int | None = None,
        image_size: tuple[int, int] = (256, 256),
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.df = _valid_dataframe(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.sample(
                min(max_samples, len(self.df)), random_state=seed
            ).reset_index(drop=True)
        if self.df.empty:
            report = validate_records(csv_path)
            errors = "\n".join(report.errors[:10])
            raise ValueError(f"No valid image pairs found in {csv_path}.\n{errors}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        input_rgb = resize_rgb(read_rgb_image(row["training_image"]), self.image_size)
        target_rgb = resize_rgb(read_rgb_image(row["ground_truth_image"]), self.image_size)
        scale = normalize_fitzpatrick_scale(row["Fitzpatrick scale"])
        return (
            torch.from_numpy(to_chw_float(input_rgb)),
            torch.from_numpy(to_chw_float(target_rgb)),
            torch.tensor([scale], dtype=torch.float32),
        )


def degrade_image(
    image_bgr: np.ndarray,
    config: SyntheticDegradationConfig | None = None,
) -> np.ndarray:
    config = config or SyntheticDegradationConfig()
    rng = np.random.default_rng(config.seed)

    gamma = rng.uniform(*config.gamma_range)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    degraded = cv2.LUT(image_bgr, table)

    weights = rng.uniform(*config.color_weight_range, size=3).astype(np.float32)
    degraded_float = degraded.astype(np.float32)
    degraded_float[:, :, 0] *= weights[0]
    degraded_float[:, :, 1] *= weights[1]
    degraded_float[:, :, 2] *= weights[2]

    alpha = rng.uniform(*config.contrast_alpha_range)
    beta = rng.uniform(0.0, 255.0 * (1.0 - alpha))
    degraded_float = degraded_float * alpha + beta
    return np.clip(degraded_float, 0, 255).astype(np.uint8)
```

Replace `data/training_preprocess.py` with a wrapper:

```python
import argparse
from pathlib import Path

import cv2
import pandas as pd

from fitzpatrick_optimizer.data import SyntheticDegradationConfig, degrade_image


def preprocess_dataset(
    csv_path: str,
    images_dir: str,
    output_images_dir: str,
    output_csv_path: str,
    seed: int = 42,
) -> None:
    df = pd.read_csv(csv_path)
    output_dir = Path(output_images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for row_index, row in df.iterrows():
        image_path = Path(images_dir) / f"{row['md5hash']}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        degraded = degrade_image(
            image,
            SyntheticDegradationConfig(seed=seed + int(row_index)),
        )
        training_path = output_dir / f"train_{row['md5hash']}.jpg"
        cv2.imwrite(str(training_path), degraded)
        records.append(
            {
                "training_image": str(training_path),
                "ground_truth_image": str(image_path),
                "Fitzpatrick scale": int(row["fitzpatrick_scale"]),
            }
        )

    pd.DataFrame(records).to_csv(output_csv_path, index=False)
    print(f"Created {len(records)} synthetic pairs at {output_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs")
    parser.add_argument("--csv_path", default="data/fitzpatrick17k.csv")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--output_images_dir", default="data/training_images")
    parser.add_argument("--output_csv_path", default="data/labels.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess_dataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_images_dir=args.output_images_dir,
        output_csv_path=args.output_csv_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/data.py data/training_preprocess.py tests/test_data.py
git commit -m "feat: add validated dataset and deterministic preprocessing"
```

---

### Task 4: Move and Rename Models Honestly

**Files:**
- Create: `src/fitzpatrick_optimizer/models/residual_filter.py`
- Create: `src/fitzpatrick_optimizer/models/illumination_unet.py`
- Modify: `src/fitzpatrick_optimizer/models/__init__.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_models.py`:

```python
import torch

from fitzpatrick_optimizer.models import (
    IlluminationGuidedUNet,
    ParameterConditionedResidualFilter,
)


def test_parameter_conditioned_residual_filter_output_shape_and_range():
    model = ParameterConditionedResidualFilter(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    s = torch.tensor([[0.0], [1.0]])

    y = model(x, s)

    assert y.shape == x.shape
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_illumination_guided_unet_output_shape_and_illumination_map():
    model = IlluminationGuidedUNet()
    x = torch.rand(2, 3, 64, 64)
    s = torch.tensor([[0.2], [0.8]])

    y, illumination = model(x, s)

    assert y.shape == x.shape
    assert illumination.shape == (2, 1, 64, 64)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL because the model modules do not exist.

- [ ] **Step 3: Implement model modules**

Create `src/fitzpatrick_optimizer/models/residual_filter.py` by moving the DeepLPF code from `src/train_deeplpf.py` and applying these exact public names:

```python
import torch
from torch import nn
import torchvision


def apply_residual_filter(
    x: torch.Tensor,
    p_grad: torch.Tensor,
    p_ellip: torch.Tensor,
    p_poly: torch.Tensor,
) -> torch.Tensor:
    batch_size = x.shape[0]
    grad_scale = torch.sigmoid(p_grad.mean(dim=1).view(batch_size, 1, 1, 1))
    a = p_poly[:, :3].view(batch_size, 3, 1, 1) * 0.1
    b = p_poly[:, 3:6].view(batch_size, 3, 1, 1) * 0.1
    c = p_poly[:, 6:9].view(batch_size, 3, 1, 1) * 0.1
    residual = a * (x**2) + b * x + c
    ellip_shift = torch.tanh(p_ellip.mean(dim=1).view(batch_size, 1, 1, 1)) * 0.1
    return torch.clamp(x + residual * grad_scale + ellip_shift, 0.0, 1.0)


class ParameterConditionedResidualFilter(nn.Module):
    """ResNet-conditioned residual image filter used as a DeepLPF-inspired baseline."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.s_embed = nn.Linear(1, 1)
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)

        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv.weight
            self.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 76))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        s_map = self.s_embed(s).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
        features = torch.cat([x, s_map], dim=1)
        features = self.conv1(features)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features)
        params = self.head(torch.flatten(features, 1))
        return apply_residual_filter(x, params[:, :8], params[:, 8:16], params[:, 16:])
```

Create `src/fitzpatrick_optimizer/models/illumination_unet.py` by moving the Retinex code from `src/train_retinex.py` and applying these exact class names:

```python
import torch
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features * 2),
        )

    def forward(self, feature_map: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(s).chunk(2, dim=1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        return feature_map * gamma + beta


class IlluminationUNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU())
        self.film = FiLMLayer(256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.out_l = nn.Conv2d(32, 1, 1)
        self.out_r = nn.Conv2d(32, 3, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.film(self.bottleneck(e3), s)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_l(d1)), torch.sigmoid(self.out_r(d1))


class SobelTextureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False, groups=3)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        weights = torch.zeros(6, 1, 3, 3)
        for channel in range(3):
            weights[channel * 2, 0] = sobel_x
            weights[channel * 2 + 1, 0] = sobel_y
        self.extractor.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class GaussianMembershipActivation(nn.Module):
    def __init__(self, center: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.center = center
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.pow(x - self.center, 2) / (2 * (self.sigma**2)))


class RefinementCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            GaussianMembershipActivation(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IlluminationGuidedUNet(nn.Module):
    """FiLM-conditioned U-Net with Sobel texture features and refinement CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.illumination_backbone = IlluminationUNetBackbone()
        self.texture_extractor = SobelTextureExtractor()
        self.refinement = RefinementCNN()

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination, reflectance = self.illumination_backbone(x, s)
        texture = self.texture_extractor(reflectance)
        output = self.refinement(torch.cat([reflectance, texture, x], dim=1))
        return output, illumination
```

Update `src/fitzpatrick_optimizer/models/__init__.py`:

```python
from fitzpatrick_optimizer.models.illumination_unet import IlluminationGuidedUNet
from fitzpatrick_optimizer.models.residual_filter import ParameterConditionedResidualFilter

__all__ = ["IlluminationGuidedUNet", "ParameterConditionedResidualFilter"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/models tests/test_models.py
git commit -m "feat: move models into honest package modules"
```

---

### Task 5: Metrics, Baselines, and Deterministic Splits

**Files:**
- Create: `src/fitzpatrick_optimizer/metrics.py`
- Create: `src/fitzpatrick_optimizer/splits.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_splits.py`

- [ ] **Step 1: Write failing metrics and split tests**

Create `tests/test_metrics.py`:

```python
import torch

from fitzpatrick_optimizer.metrics import compute_batch_metrics, grouped_average


def test_compute_batch_metrics_returns_expected_identity_scores():
    target = torch.ones(2, 3, 8, 8)
    prediction = target.clone()

    metrics = compute_batch_metrics(prediction, target)

    assert metrics.l1 == 0.0
    assert metrics.mse == 0.0
    assert metrics.psnr == float("inf")
    assert metrics.ssim > 0.99


def test_grouped_average_aggregates_by_fitzpatrick_scale():
    rows = [
        {"fitzpatrick_scale": 1, "l1": 0.1, "mse": 0.2, "psnr": 20.0, "ssim": 0.9},
        {"fitzpatrick_scale": 1, "l1": 0.3, "mse": 0.4, "psnr": 22.0, "ssim": 0.8},
        {"fitzpatrick_scale": 2, "l1": 0.5, "mse": 0.6, "psnr": 24.0, "ssim": 0.7},
    ]

    result = grouped_average(rows)

    assert result[1]["count"] == 2
    assert result[1]["l1"] == 0.2
    assert result[2]["count"] == 1
    assert result[2]["ssim"] == 0.7
```

Create `tests/test_splits.py`:

```python
import pandas as pd

from fitzpatrick_optimizer.splits import assign_split


def test_assign_split_is_deterministic_and_complete():
    df = pd.DataFrame(
        {
            "training_image": [f"input-{i}.jpg" for i in range(20)],
            "ground_truth_image": [f"target-{i}.jpg" for i in range(20)],
            "Fitzpatrick scale": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4] * 2,
        }
    )

    first = assign_split(df, seed=7)
    second = assign_split(df, seed=7)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}
    assert len(first) == 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_splits.py -v
```

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement metrics and splits**

Create `src/fitzpatrick_optimizer/metrics.py`:

```python
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ImageMetrics:
    l1: float
    mse: float
    psnr: float
    ssim: float


def ssim_score(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=window_size // 2) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2))
    return score.mean()


def compute_psnr(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_batch_metrics(prediction: torch.Tensor, target: torch.Tensor) -> ImageMetrics:
    l1 = torch.mean(torch.abs(prediction - target)).item()
    mse = torch.mean((prediction - target) ** 2).item()
    return ImageMetrics(l1=l1, mse=mse, psnr=compute_psnr(mse), ssim=ssim_score(prediction, target).item())


def grouped_average(rows: list[dict[str, float | int]]) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for row in rows:
        grouped.setdefault(int(row["fitzpatrick_scale"]), []).append(row)

    result: dict[int, dict[str, float | int]] = {}
    for scale, scale_rows in grouped.items():
        result[scale] = {"count": len(scale_rows)}
        for metric in ("l1", "mse", "psnr", "ssim"):
            result[scale][metric] = sum(float(row[metric]) for row in scale_rows) / len(scale_rows)
    return result
```

Create `src/fitzpatrick_optimizer/splits.py`:

```python
import numpy as np
import pandas as pd


def assign_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must leave a positive test fraction")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_end = max(1, int(len(indices) * train_fraction))
    val_end = max(train_end + 1, int(len(indices) * (train_fraction + val_fraction)))
    val_end = min(val_end, len(indices) - 1)

    split_values = np.empty(len(df), dtype=object)
    split_values[indices[:train_end]] = "train"
    split_values[indices[train_end:val_end]] = "val"
    split_values[indices[val_end:]] = "test"

    result = df.copy()
    result["split"] = split_values
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_splits.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/metrics.py src/fitzpatrick_optimizer/splits.py tests/test_metrics.py tests/test_splits.py
git commit -m "feat: add holdout split and grouped metrics utilities"
```

---

### Task 6: Training CLI with Package Imports and Compatibility Wrappers

**Files:**
- Create: `src/fitzpatrick_optimizer/train.py`
- Modify: `src/train_deeplpf.py`
- Modify: `src/train_retinex.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_cli.py`:

```python
import subprocess
import sys


def test_train_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "residual-filter" in result.stdout
    assert "illumination-unet" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py::test_train_cli_help_runs -v
```

Expected: FAIL because `fitzpatrick_optimizer.train` does not exist.

- [ ] **Step 3: Implement training CLI**

Create `src/fitzpatrick_optimizer/train.py`:

```python
import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fitzpatrick_optimizer.config import get_device
from fitzpatrick_optimizer.data import FitzpatrickImageDataset
from fitzpatrick_optimizer.metrics import ssim_score
from fitzpatrick_optimizer.models import IlluminationGuidedUNet, ParameterConditionedResidualFilter
from fitzpatrick_optimizer.randomness import seed_everything


def create_model(model_name: str) -> nn.Module:
    if model_name == "residual-filter":
        return ParameterConditionedResidualFilter(pretrained=True)
    if model_name == "illumination-unet":
        return IlluminationGuidedUNet()
    raise ValueError(f"Unknown model: {model_name}")


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(prediction, target) + (1.0 - ssim_score(prediction, target))


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FitzpatrickImageDataset(
        args.csv_path,
        max_samples=args.max_samples,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = create_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for input_image, target_image, scale in dataloader:
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            scale = scale.to(device)
            optimizer.zero_grad()
            output = model(input_image, scale)
            prediction = output[0] if isinstance(output, tuple) else output
            loss = reconstruction_loss(prediction, target_image)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info("epoch=%s loss=%.6f", epoch, epoch_loss / len(dataloader))
        torch.save(model.state_dict(), output_dir / f"{args.model}-epoch-{epoch}.pth")

    final_path = output_dir / f"{args.model}.pth"
    torch.save(model.state_dict(), final_path)
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train image illumination normalization models")
    parser.add_argument("--model", choices=["residual-filter", "illumination-unet"], required=True)
    parser.add_argument("--csv_path", default="data/labels.csv")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    path = train(args)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
```

Replace `src/train_deeplpf.py` with:

```python
import sys

from fitzpatrick_optimizer.train import main


if __name__ == "__main__":
    argv = ["--model", "residual-filter", *sys.argv[1:]]
    if "--scale_dataset" in argv:
        index = argv.index("--scale_dataset")
        argv[index] = "--max_samples"
    main(argv)
```

Replace `src/train_retinex.py` with:

```python
import sys

from fitzpatrick_optimizer.train import main


if __name__ == "__main__":
    argv = ["--model", "illumination-unet", *sys.argv[1:]]
    if "--scale_dataset" in argv:
        index = argv.index("--scale_dataset")
        argv[index] = "--max_samples"
    main(argv)
```

- [ ] **Step 4: Run the CLI test**

Run:

```bash
uv run pytest tests/test_cli.py::test_train_cli_help_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/train.py src/train_deeplpf.py src/train_retinex.py tests/test_cli.py
git commit -m "feat: add package training cli"
```

---

### Task 7: Evaluation CLI with Holdout, Baselines, and Grouped Metrics

**Files:**
- Create: `src/fitzpatrick_optimizer/evaluate.py`
- Modify: `src/evaluate_deeplpf.py`
- Modify: `src/evaluate_retinex.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing evaluation CLI test**

Append to `tests/test_cli.py`:

```python
def test_evaluate_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.evaluate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "--metrics_json" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py::test_evaluate_cli_help_runs -v
```

Expected: FAIL because `fitzpatrick_optimizer.evaluate` does not exist.

- [ ] **Step 3: Implement evaluation CLI**

Create `src/fitzpatrick_optimizer/evaluate.py`:

```python
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fitzpatrick_optimizer.config import get_device
from fitzpatrick_optimizer.data import FitzpatrickImageDataset
from fitzpatrick_optimizer.metrics import compute_batch_metrics, grouped_average
from fitzpatrick_optimizer.splits import assign_split
from fitzpatrick_optimizer.train import create_model


def _scale_from_normalized(scale: torch.Tensor) -> int:
    return int(round(float(scale.item()) * 5 + 1))


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = get_device(args.device)
    df = pd.read_csv(args.csv_path)
    if "split" not in df.columns:
        df = assign_split(df, seed=args.seed)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    temporary_csv = Path(args.output_dir) / f"eval-{args.split}.csv"
    temporary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temporary_csv, index=False)

    dataset = FitzpatrickImageDataset(temporary_csv, image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    model_rows: list[dict[str, float | int]] = []
    identity_rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for input_image, target_image, scale in dataloader:
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            scale = scale.to(device)
            output = model(input_image, scale)
            prediction = output[0] if isinstance(output, tuple) else output
            model_metrics = compute_batch_metrics(prediction, target_image)
            identity_metrics = compute_batch_metrics(input_image, target_image)
            for item_index in range(input_image.shape[0]):
                fitzpatrick_scale = _scale_from_normalized(scale[item_index])
                model_rows.append({"fitzpatrick_scale": fitzpatrick_scale, **model_metrics.__dict__})
                identity_rows.append({"fitzpatrick_scale": fitzpatrick_scale, **identity_metrics.__dict__})

    report = {
        "model": args.model,
        "split": args.split,
        "count": len(dataset),
        "model_grouped": grouped_average(model_rows),
        "identity_baseline_grouped": grouped_average(identity_rows),
    }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models on deterministic holdout splits")
    parser.add_argument("--model", choices=["residual-filter", "illumination-unet"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--csv_path", default="data/labels.csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--metrics_json", default="results/metrics.json")
    parser.add_argument("--output_dir", default="results/evaluation")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

Replace `src/evaluate_deeplpf.py` with:

```python
import sys

from fitzpatrick_optimizer.evaluate import main


if __name__ == "__main__":
    main(["--model", "residual-filter", *sys.argv[1:]])
```

Replace `src/evaluate_retinex.py` with:

```python
import sys

from fitzpatrick_optimizer.evaluate import main


if __name__ == "__main__":
    main(["--model", "illumination-unet", *sys.argv[1:]])
```

- [ ] **Step 4: Run the CLI test**

Run:

```bash
uv run pytest tests/test_cli.py::test_evaluate_cli_help_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/evaluate.py src/evaluate_deeplpf.py src/evaluate_retinex.py tests/test_cli.py
git commit -m "feat: add grouped holdout evaluation cli"
```

---

### Task 8: Inference CLI and Fresh-Clone Demo Assets

**Files:**
- Create: `src/fitzpatrick_optimizer/infer.py`
- Create: `src/fitzpatrick_optimizer/demo.py`
- Modify: `src/infer_deeplpf.py`
- Modify: `src/infer_retinex.py`
- Create: `tests/test_demo.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing demo and inference tests**

Create `tests/test_demo.py`:

```python
from pathlib import Path

import pandas as pd

from fitzpatrick_optimizer.demo import create_demo_dataset


def test_create_demo_dataset_writes_images_and_labels(tmp_path):
    output_dir = tmp_path / "demo"

    csv_path = create_demo_dataset(output_dir, count=3)

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert set(df.columns) == {"training_image", "ground_truth_image", "Fitzpatrick scale"}
    for path in df["training_image"].tolist() + df["ground_truth_image"].tolist():
        assert Path(path).exists()
```

Append to `tests/test_cli.py`:

```python
def test_infer_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.infer", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--output_dir" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_demo.py tests/test_cli.py::test_infer_cli_help_runs -v
```

Expected: FAIL because demo and infer modules do not exist.

- [ ] **Step 3: Implement generated demo assets and inference CLI**

Create `src/fitzpatrick_optimizer/demo.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fitzpatrick_optimizer.data import SyntheticDegradationConfig, degrade_image


def _synthetic_skin_image(index: int, size: int = 128) -> np.ndarray:
    base_colors = [
        (235, 198, 170),
        (213, 161, 120),
        (181, 122, 82),
        (137, 85, 55),
        (96, 60, 42),
        (62, 40, 30),
    ]
    color = np.array(base_colors[index % len(base_colors)], dtype=np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    center = (size // 2, size // 2)
    cv2.circle(image, center, size // 5, (60, 35, 45), thickness=-1)
    cv2.circle(image, (center[0] - 12, center[1] - 8), size // 12, (120, 70, 80), thickness=2)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def create_demo_dataset(output_dir: str | Path = "demo_assets", count: int = 6) -> Path:
    output_dir = Path(output_dir)
    input_dir = output_dir / "inputs"
    target_dir = output_dir / "targets"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index in range(count):
        target_bgr = _synthetic_skin_image(index)
        input_bgr = degrade_image(target_bgr, SyntheticDegradationConfig(seed=100 + index))
        target_path = target_dir / f"sample-{index}.jpg"
        input_path = input_dir / f"sample-{index}.jpg"
        cv2.imwrite(str(target_path), target_bgr)
        cv2.imwrite(str(input_path), input_bgr)
        rows.append(
            {
                "training_image": str(input_path),
                "ground_truth_image": str(target_path),
                "Fitzpatrick scale": (index % 6) + 1,
            }
        )

    csv_path = output_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
```

Create `src/fitzpatrick_optimizer/infer.py`:

```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fitzpatrick_optimizer.config import get_device
from fitzpatrick_optimizer.imaging import normalize_fitzpatrick_scale, read_rgb_image, resize_rgb, to_chw_float, write_rgb_image
from fitzpatrick_optimizer.train import create_model


def run_inference(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    model = create_model(args.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(args.csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with torch.no_grad():
        for _, row in df.iterrows():
            image_path = Path(str(row[args.image_col]))
            image = read_rgb_image(image_path)
            resized = resize_rgb(image, (args.image_size, args.image_size))
            image_tensor = torch.from_numpy(to_chw_float(resized)).unsqueeze(0).to(device)
            scale = torch.tensor(
                [[normalize_fitzpatrick_scale(row[args.scale_col])]],
                dtype=torch.float32,
                device=device,
            )
            output = model(image_tensor, scale)
            prediction = output[0] if isinstance(output, tuple) else output
            output_rgb = prediction.squeeze(0).cpu().numpy()
            output_rgb = np.clip(output_rgb.transpose((1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
            write_rgb_image(output_dir / image_path.name, output_rgb)
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model inference over a CSV of images")
    parser.add_argument("--model", choices=["residual-filter", "illumination-unet"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--csv_path", default="data/labels.csv")
    parser.add_argument("--image_col", default="training_image")
    parser.add_argument("--scale_col", default="Fitzpatrick scale")
    parser.add_argument("--output_dir", default="results/inference")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = run_inference(args)
    print(f"Wrote {written} images to {args.output_dir}")


if __name__ == "__main__":
    main()
```

Replace `src/infer_deeplpf.py` with:

```python
import sys

from fitzpatrick_optimizer.infer import main


if __name__ == "__main__":
    main(["--model", "residual-filter", *sys.argv[1:]])
```

Replace `src/infer_retinex.py` with:

```python
import sys

from fitzpatrick_optimizer.infer import main


if __name__ == "__main__":
    main(["--model", "illumination-unet", *sys.argv[1:]])
```

Add to `.gitignore`:

```text
demo_assets/
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_demo.py tests/test_cli.py::test_infer_cli_help_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/demo.py src/fitzpatrick_optimizer/infer.py src/infer_deeplpf.py src/infer_retinex.py tests/test_demo.py tests/test_cli.py .gitignore
git commit -m "feat: add inference cli and generated demo assets"
```

---

### Task 9: Unified CLI Entrypoint

**Files:**
- Create: `src/fitzpatrick_optimizer/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing unified CLI tests**

Append to `tests/test_cli.py`:

```python
def test_unified_cli_help_lists_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "fitzpatrick_optimizer.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "infer" in result.stdout
    assert "create-demo-data" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py::test_unified_cli_help_lists_subcommands -v
```

Expected: FAIL because `fitzpatrick_optimizer.cli` does not exist.

- [ ] **Step 3: Implement unified CLI**

Create `src/fitzpatrick_optimizer/cli.py`:

```python
import argparse

from fitzpatrick_optimizer.demo import create_demo_dataset
from fitzpatrick_optimizer.evaluate import main as evaluate_main
from fitzpatrick_optimizer.infer import main as infer_main
from fitzpatrick_optimizer.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fitzpatrick image optimizer portfolio demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train a model")
    subparsers.add_parser("evaluate", help="Evaluate a model")
    subparsers.add_parser("infer", help="Run inference")
    demo_parser = subparsers.add_parser("create-demo-data", help="Create generated demo images")
    demo_parser.add_argument("--output_dir", default="demo_assets")
    demo_parser.add_argument("--count", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "train":
        train_main(remaining)
    elif args.command == "evaluate":
        evaluate_main(remaining)
    elif args.command == "infer":
        infer_main(remaining)
    elif args.command == "create-demo-data":
        csv_path = create_demo_dataset(args.output_dir, args.count)
        print(f"Created demo dataset: {csv_path}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_cli.py::test_unified_cli_help_lists_subcommands -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitzpatrick_optimizer/cli.py tests/test_cli.py
git commit -m "feat: add unified portfolio demo cli"
```

---

### Task 10: Rewrite Public Documentation and Model Card

**Files:**
- Modify: `README.md`
- Create: `docs/model-card.md`
- Create: `docs/reproducibility.md`
- Create: `docs/results-schema.md`

- [ ] **Step 1: Replace README with honest portfolio framing**

Use this README structure and preserve `comparison_grid.png` as the visual showcase:

```markdown
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
```

- [ ] **Step 2: Add model card**

Create `docs/model-card.md`:

```markdown
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
```

- [ ] **Step 3: Add reproducibility and results schema docs**

Create `docs/reproducibility.md`:

```markdown
# Reproducibility

## Environment

Use Python 3.11 or newer.

```bash
uv sync --extra dev
```

## Demo Run

```bash
uv run fitzopt create-demo-data --output_dir demo_assets --count 6
uv run fitzopt train --model residual-filter --csv_path demo_assets/labels.csv --max_samples 6 --epochs 1 --batch_size 2 --output_dir models/demo
uv run fitzopt evaluate --model residual-filter --model_path models/demo/residual-filter.pth --csv_path demo_assets/labels.csv --split test --metrics_json results/demo-metrics.json --max_samples 2
```

## Full Dataset Run

1. Download Fitzpatrick17k images according to the dataset source terms.
2. Store images as `data/images/<md5hash>.jpg`.
3. Generate synthetic pairs:

```bash
uv run python data/training_preprocess.py --csv_path data/fitzpatrick17k.csv --images_dir data/images --output_images_dir data/training_images --output_csv_path data/labels.csv --seed 42
```

4. Train and evaluate on deterministic splits:

```bash
uv run fitzopt train --model residual-filter --csv_path data/labels.csv --epochs 50 --batch_size 32 --output_dir models
uv run fitzopt evaluate --model residual-filter --model_path models/residual-filter.pth --csv_path data/labels.csv --split test --metrics_json results/residual-filter-test.json
```
```

Create `docs/results-schema.md`:

```markdown
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
```

- [ ] **Step 4: Run documentation sanity checks**

Run:

```bash
rg -n "diagnostic|diagnosis|elimination of|perfectly preserve|optimal diagnostic|Add your description" README.md docs pyproject.toml
```

Expected: no unsupported claims except the explicit out-of-scope phrases in `docs/model-card.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/model-card.md docs/reproducibility.md docs/results-schema.md
git commit -m "docs: reframe project as honest portfolio demo"
```

---

### Task 11: CI and Final Verification

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Add CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run pytest
      - run: uv run python -m compileall src data
```

- [ ] **Step 2: Run final local verification**

Run:

```bash
uv run ruff check .
uv run pytest
uv run python -m compileall src data
uv run fitzopt create-demo-data --output_dir demo_assets --count 6
```

Expected:

```text
All checks passed!
... passed
Listing 'src'...
Created demo dataset: demo_assets/labels.csv
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add portfolio demo verification workflow"
```

---

## Self-Review

Spec coverage:

- README overclaims: covered in Task 10.
- DeepLPF mismatch: covered in Task 4 and Task 10 by renaming to `residual-filter` and documenting it as DeepLPF-inspired.
- Weak evaluation: covered in Task 5 and Task 7 with deterministic splits, identity baseline, and grouped metrics.
- Missing images as black images: covered in Task 3 by validated dataset and fail-fast behavior.
- Retinex/CDH overnaming: covered in Task 4 and Task 10 by using `IlluminationGuidedUNet` and `SobelTextureExtractor`.
- Script-oriented repo: covered in Tasks 1, 6, 7, 8, and 9.
- Public demo path: covered in Task 8 and Task 10.
- Metadata/tooling: covered in Task 1 and Task 11.
- Git commits after implementation steps: every task ends with an explicit commit command.

Placeholder scan:

- Checked for red-flag placeholder language and removed it from implementation steps.

Type consistency:

- Model names are `residual-filter` and `illumination-unet` in CLIs.
- Python class names are `ParameterConditionedResidualFilter` and `IlluminationGuidedUNet`.
- Dataset CSV columns remain `training_image`, `ground_truth_image`, and `Fitzpatrick scale`.
- Split values are `train`, `val`, and `test`.
