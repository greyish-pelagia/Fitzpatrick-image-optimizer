import torch

from fitzpatrick_optimizer.metrics import compute_batch_metrics, grouped_average


def test_compute_batch_metrics_returns_expected_identity_scores():
    target = torch.ones(2, 3, 8, 8)
    prediction = target.clone()

    metrics = compute_batch_metrics(prediction, target)

    assert metrics.l1 == 0.0
    assert metrics.mse == 0.0
    assert metrics.psnr == float("inf")
    assert metrics.ssim > 0.99


def test_grouped_average_aggregates_by_fitzpatrick_scale():
    rows = [
        {"fitzpatrick_scale": 1, "l1": 0.1, "mse": 0.2, "psnr": 20.0, "ssim": 0.9},
        {"fitzpatrick_scale": 1, "l1": 0.3, "mse": 0.4, "psnr": 22.0, "ssim": 0.8},
        {"fitzpatrick_scale": 2, "l1": 0.5, "mse": 0.6, "psnr": 24.0, "ssim": 0.7},
    ]

    result = grouped_average(rows)

    assert result[1]["count"] == 2
    assert result[1]["l1"] == 0.2
    assert result[2]["count"] == 1
    assert result[2]["ssim"] == 0.7
