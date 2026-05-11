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
    cv2.circle(
        image,
        (center[0] - 12, center[1] - 8),
        size // 12,
        (120, 70, 80),
        thickness=2,
    )
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
        input_bgr = degrade_image(
            target_bgr,
            SyntheticDegradationConfig(seed=100 + index),
        )
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
