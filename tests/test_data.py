from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from fitzpatrick_optimizer.data import (
    FitzpatrickImageDataset,
    SyntheticDegradationConfig,
    degrade_image,
    validate_records,
)
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
