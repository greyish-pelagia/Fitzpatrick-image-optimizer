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


def ssim_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    sigma_x = (
        F.avg_pool2d(
            prediction * prediction,
            window_size,
            stride=1,
            padding=window_size // 2,
        )
        - mu_x.pow(2)
    )
    sigma_y = (
        F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2)
        - mu_y.pow(2)
    )
    sigma_xy = (
        F.avg_pool2d(
            prediction * target,
            window_size,
            stride=1,
            padding=window_size // 2,
        )
        - mu_x * mu_y
    )
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def compute_psnr(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> ImageMetrics:
    l1 = torch.mean(torch.abs(prediction - target)).item()
    mse = torch.mean((prediction - target) ** 2).item()
    return ImageMetrics(
        l1=l1,
        mse=mse,
        psnr=compute_psnr(mse),
        ssim=ssim_score(prediction, target).item(),
    )


def grouped_average(
    rows: list[dict[str, float | int]],
) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for row in rows:
        grouped.setdefault(int(row["fitzpatrick_scale"]), []).append(row)

    result: dict[int, dict[str, float | int]] = {}
    for scale, scale_rows in grouped.items():
        result[scale] = {"count": len(scale_rows)}
        for metric in ("l1", "mse", "psnr", "ssim"):
            result[scale][metric] = sum(
                float(row[metric]) for row in scale_rows
            ) / len(scale_rows)
    return result
