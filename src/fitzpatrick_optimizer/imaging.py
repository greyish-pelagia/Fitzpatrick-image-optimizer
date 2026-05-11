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
