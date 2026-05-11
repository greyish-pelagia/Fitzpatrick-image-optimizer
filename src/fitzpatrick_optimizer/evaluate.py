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

    dataset = FitzpatrickImageDataset(
        temporary_csv,
        image_size=(args.image_size, args.image_size),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model).to(device)
    if args.model != "baseline":
        model.load_state_dict(
            torch.load(args.model_path, map_location=device, weights_only=True)
        )
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
                model_rows.append(
                    {"fitzpatrick_scale": fitzpatrick_scale, **model_metrics.__dict__}
                )
                identity_rows.append(
                    {
                        "fitzpatrick_scale": fitzpatrick_scale,
                        **identity_metrics.__dict__,
                    }
                )

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
    parser = argparse.ArgumentParser(
        description="Evaluate models on deterministic holdout splits"
    )
    parser.add_argument(
        "--model",
        choices=["residual-filter", "illumination-unet", "baseline"],
        required=True,
    )
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
