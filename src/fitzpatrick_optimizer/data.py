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
        target_rgb = resize_rgb(
            read_rgb_image(row["ground_truth_image"]), self.image_size
        )
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
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
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
