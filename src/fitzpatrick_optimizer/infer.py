import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fitzpatrick_optimizer.config import get_device
from fitzpatrick_optimizer.imaging import (
    normalize_fitzpatrick_scale,
    read_rgb_image,
    resize_rgb,
    to_chw_float,
    write_rgb_image,
)
from fitzpatrick_optimizer.train import create_model


def run_inference(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    model = create_model(args.model).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
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
            image_tensor = (
                torch.from_numpy(to_chw_float(resized)).unsqueeze(0).to(device)
            )
            scale = torch.tensor(
                [[normalize_fitzpatrick_scale(row[args.scale_col])]],
                dtype=torch.float32,
                device=device,
            )
            output = model(image_tensor, scale)
            prediction = output[0] if isinstance(output, tuple) else output
            output_rgb = prediction.squeeze(0).cpu().numpy()
            output_rgb = np.clip(
                output_rgb.transpose((1, 2, 0)) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            write_rgb_image(output_dir / image_path.name, output_rgb)
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model inference over a CSV of images"
    )
    parser.add_argument(
        "--model",
        choices=["residual-filter", "illumination-unet"],
        required=True,
    )
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
