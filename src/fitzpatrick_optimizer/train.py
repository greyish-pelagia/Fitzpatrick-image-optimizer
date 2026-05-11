import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fitzpatrick_optimizer.config import get_device
from fitzpatrick_optimizer.data import FitzpatrickImageDataset
from fitzpatrick_optimizer.metrics import ssim_score
from fitzpatrick_optimizer.models import (
    IlluminationGuidedUNet,
    ParameterConditionedResidualFilter,
)
from fitzpatrick_optimizer.randomness import seed_everything


def create_model(model_name: str) -> nn.Module:
    if model_name == "residual-filter":
        return ParameterConditionedResidualFilter(pretrained=True)
    if model_name == "illumination-unet":
        return IlluminationGuidedUNet()
    raise ValueError(f"Unknown model: {model_name}")


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.l1_loss(prediction, target) + (
        1.0 - ssim_score(prediction, target)
    )


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
    parser = argparse.ArgumentParser(
        description="Train image illumination normalization models"
    )
    parser.add_argument(
        "--model",
        choices=["residual-filter", "illumination-unet"],
        required=True,
    )
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    path = train(args)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
