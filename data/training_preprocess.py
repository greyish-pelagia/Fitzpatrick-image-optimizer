import os
import cv2
import random
import numpy as np
import pandas as pd
from typing import Tuple


def random_gamma_shift(
    image: np.ndarray, gamma_range: Tuple[float, float] = (0.4, 2.5)
) -> np.ndarray:
    """
    Apply a power-law transformation (Gamma Shift).
    X = Y^gamma
    """
    gamma = random.uniform(gamma_range[0], gamma_range[1])

    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def random_color_cast(
    image: np.ndarray, weight_range: Tuple[float, float] = (0.75, 1.25)
) -> np.ndarray:
    """
    Multiply individual RGB channels by independent random scalars.
    """

    b_weight = random.uniform(weight_range[0], weight_range[1])
    g_weight = random.uniform(weight_range[0], weight_range[1])
    r_weight = random.uniform(weight_range[0], weight_range[1])

    img_float = image.astype(np.float32)
    img_float[:, :, 0] *= b_weight
    img_float[:, :, 1] *= g_weight
    img_float[:, :, 2] *= r_weight

    return np.clip(img_float, 0, 255).astype(np.uint8)


def histogram_denormalization(
    image: np.ndarray, alpha_range: Tuple[float, float] = (0.5, 0.8)
) -> np.ndarray:
    """
    Compress the dynamic range to simulate low contrast.
    X = Y * alpha + beta
    """
    alpha = random.uniform(alpha_range[0], alpha_range[1])

    max_beta = 255 * (1.0 - alpha)
    beta = random.uniform(0, max_beta)

    img_float = image.astype(np.float32)
    img_float = img_float * alpha + beta

    return np.clip(img_float, 0, 255).astype(np.uint8)


def degrade_image(image: np.ndarray) -> np.ndarray:
    """
    Pass the well-lit Ground Truth image through the randomized degradation function.
    """
    img = random_gamma_shift(image)
    img = random_color_cast(img)
    img = histogram_denormalization(img)
    return img


def preprocess_dataset(
    csv_path: str, images_dir: str, output_images_dir: str, output_csv_path: str
):
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    os.makedirs(output_images_dir, exist_ok=True)

    records = []
    total = len(df)

    print(f"Found {total} records. Starting preprocessing...")

    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{total} records...")

        md5hash = row["md5hash"]
        fitzpatrick_scale = row["fitzpatrick_scale"]

        gt_filename = f"{md5hash}.jpg"
        gt_image_path = os.path.join(images_dir, gt_filename)

        if not os.path.exists(gt_image_path):
            continue

        img = cv2.imread(gt_image_path)
        if img is None:
            continue

        degraded_img = degrade_image(img)

        training_filename = f"train_{md5hash}.jpg"
        training_image_path = os.path.join(output_images_dir, training_filename)
        cv2.imwrite(training_image_path, degraded_img)

        records.append(
            {
                "training_image": training_image_path,
                "ground_truth_image": gt_image_path,
                "Fitzpatrick scale": fitzpatrick_scale,
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Done! Preprocessed {len(records)} images in total.")
    print(f"Created new labels file at: {output_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Fitzpatrick17k Images Pipeline"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/fitzpatrick17k.csv",
        help="Original CSV path",
    )
    parser.add_argument(
        "--images_dir", type=str, default="data/images", help="Original images dir"
    )
    parser.add_argument(
        "--output_images_dir",
        type=str,
        default="data/training_images",
        help="Output images dir",
    )
    parser.add_argument(
        "--output_csv_path", type=str, default="data/labels.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    preprocess_dataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_images_dir=args.output_images_dir,
        output_csv_path=args.output_csv_path,
    )
