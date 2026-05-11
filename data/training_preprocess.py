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
