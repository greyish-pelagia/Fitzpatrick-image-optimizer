import argparse
import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_deeplpf import DeepLPFModel, FitzpatrickDataset, SSIMLoss

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def compute_psnr(mse):
    """Compute Peak Signal-to-Noise Ratio."""
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def evaluate_model(model_path, csv_path, num_samples, batch_size):
    print(f"Loading pretrained model from: {model_path}")
    print(f"Target hardware device: {device}")

    model = DeepLPFModel()

    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Please train and save the model first."
        )
        return

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    if num_samples > 0 and num_samples < len(df):
        print(f"Sampling {num_samples} random images for evaluation...")
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    tmp_csv = "data/tmp_eval_labels.csv"
    os.makedirs(os.path.dirname(tmp_csv), exist_ok=True)
    df.to_csv(tmp_csv, index=False)

    dataset = FitzpatrickDataset(tmp_csv, max_samples=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    l1_loss_fn = nn.L1Loss(reduction="mean")
    mse_loss_fn = nn.MSELoss(reduction="mean")
    ssim_loss_fn = SSIMLoss().to(device)

    total_l1 = 0.0
    total_mse = 0.0
    total_ssim_err = 0.0
    total_psnr = 0.0
    count = 0

    print("\n---------------------------------------------------------")
    print("Initiating Image Reconstruction Diagnostics")
    print("---------------------------------------------------------")

    with torch.no_grad():
        for batch_idx, (t_img, g_img, s_val) in enumerate(dataloader):
            t_img = t_img.to(device)
            g_img = g_img.to(device)
            s_val = s_val.to(device)

            output = model(t_img, s_val)

            l1 = l1_loss_fn(output, g_img).item()
            mse = mse_loss_fn(output, g_img).item()
            ssim_err = ssim_loss_fn(output, g_img).item()

            batch_size_actual = t_img.size(0)

            psnr = compute_psnr(mse)

            total_l1 += l1 * batch_size_actual
            total_mse += mse * batch_size_actual
            total_ssim_err += ssim_err * batch_size_actual
            total_psnr += psnr * batch_size_actual
            count += batch_size_actual

            print(f"Evaluated {count}/{len(dataset)} images...", end="\r")

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    if count == 0:
        print("\nNo valid samples found for evaluation.")
        return

    avg_l1 = total_l1 / count
    avg_mse = total_mse / count
    avg_ssim = 1.0 - (total_ssim_err / count)
    avg_psnr = total_psnr / count

    print("\n\n" + "=" * 55)
    print("               EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Total Test Samples Validated : {count}")
    print(f"Mean Absolute Error (L1)     : {avg_l1:.5f} (lower is better)")
    print(f"Mean Squared Error (MSE)     : {avg_mse:.5f} (lower is better)")
    print(f"Peak Signal-Noise (PSNR)     : {avg_psnr:.2f} dB (higher is better)")
    print(f"Structural Similarity (SSIM) : {avg_ssim:.4f} (max 1.0, higher is better)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PyTorch DeepLPF Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/deeplpf.pth",
        help="Path accurately to model .pth weights file",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/labels.csv",
        help="Original preprocessed dataset tracked",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Random sample subset size for verifying quickly",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Hardware compute allocation parameter",
    )

    args = parser.parse_args()
    evaluate_model(args.model_path, args.csv_path, args.num_samples, args.batch_size)
