import os
import math
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class SSIMLoss(nn.Module):
    """
    Simplified pure PyTorch implementation of SSIM (Structural Similarity Index) Loss.
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):

        mu1 = F.avg_pool2d(
            img1, self.window_size, stride=1, padding=self.window_size // 2
        )
        mu2 = F.avg_pool2d(
            img2, self.window_size, stride=1, padding=self.window_size // 2
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.avg_pool2d(
                img1 * img1, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.avg_pool2d(
                img2 * img2, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu2_sq
        )
        sigma12 = (
            F.avg_pool2d(
                img1 * img2, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class FitzpatrickDataset(Dataset):
    def __init__(self, csv_path, max_samples=None, img_size=(256, 256)):
        self.img_size = img_size
        self.df = pd.read_csv(csv_path)

        if max_samples is not None and max_samples > 0:
            self.df = self.df.sample(
                min(max_samples, len(self.df)), random_state=42
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        t_img_path = row["training_image"]
        g_img_path = row["ground_truth_image"]
        f_scale = row["Fitzpatrick scale"]

        t_img = cv2.imread(t_img_path)
        g_img = cv2.imread(g_img_path)

        if t_img is None or g_img is None:
            t_img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            g_img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        t_img = cv2.resize(t_img, self.img_size)
        g_img = cv2.resize(g_img, self.img_size)

        t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)

        t_img = t_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        g_img = g_img.transpose((2, 0, 1)).astype(np.float32) / 255.0

        s_norm = (f_scale - 1.0) / 5.0
        s_norm = max(0.0, min(1.0, s_norm))

        return (
            torch.tensor(t_img),
            torch.tensor(g_img),
            torch.tensor([s_norm], dtype=torch.float32),
        )


def apply_deeplpf_filters(x, p_grad, p_ellip, p_poly):
    """
    Custom PyTorch tensor operations representing the filters described in Pipeline 1.
    Since we don't have the exact complex mathematical functions for all 76 features,
    this calculates them via differentiable PyTorch proxies maintaining the computational graph.
    """
    B, C, H, W = x.shape

    grad_scale = torch.sigmoid(p_grad.mean(dim=1).view(B, 1, 1, 1))

    a = p_poly[:, :3].view(B, 3, 1, 1) * 0.1
    b = p_poly[:, 3:6].view(B, 3, 1, 1) * 0.1
    c = p_poly[:, 6:9].view(B, 3, 1, 1) * 0.1

    out = a * (x**2) + b * x + c

    ellip_shift = torch.tanh(p_ellip.mean(dim=1).view(B, 1, 1, 1)) * 0.1

    out = out * grad_scale + ellip_shift

    out = x + out

    return torch.clamp(out, 0.0, 1.0)


class DeepLPFModel(nn.Module):
    def __init__(self):
        super(DeepLPFModel, self).__init__()

        self.s_embed = nn.Linear(1, 1)

        backbone = torchvision.models.resnet50(pretrained=True)

        orig_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(
            4,
            orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=orig_conv1.bias is not None,
        )

        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = orig_conv1.weight
            self.conv1.weight[:, 3, :, :] = orig_conv1.weight.mean(dim=1)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.mlp = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 76))

    def forward(self, x, s):
        B, C, H, W = x.shape

        s_emb = self.s_embed(s)
        s_expand = s_emb.view(B, 1, 1, 1).expand(B, 1, H, W)

        x_fusion = torch.cat([x, s_expand], dim=1)

        f = self.conv1(x_fusion)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)

        f = self.avgpool(f)
        f = torch.flatten(f, 1)

        params = self.mlp(f)

        p_grad = params[:, :8]
        p_ellip = params[:, 8:16]
        p_poly = params[:, 16:]

        return apply_deeplpf_filters(x, p_grad, p_ellip, p_poly)


def train_deeplpf(csv_path, max_samples=None, epochs=10, batch_size=8, lr=1e-4):
    os.makedirs("models", exist_ok=True)
    print(f"Loading dataset from {csv_path} with max_samples={max_samples}")
    dataset = FitzpatrickDataset(csv_path, max_samples=max_samples)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    print(f"Initializing Context -> Using device: {device}")
    model = DeepLPFModel().to(device)

    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = SSIMLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("==================================================================")
    print("Initiating Pipeline 1: Parameter-Driven DeepLPF (Simple) Training")
    print("==================================================================")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (t_img, g_img, s_val) in enumerate(dataloader):
            t_img = t_img.to(device)
            g_img = g_img.to(device)
            s_val = s_val.to(device)

            optimizer.zero_grad()

            output = model(t_img, s_val)

            loss_l1 = l1_loss_fn(output, g_img)
            loss_ssim = ssim_loss_fn(output, g_img)

            loss = loss_l1 + loss_ssim

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs - 1}] Batch [{batch_idx}/{len(dataloader) - 1}] Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch} Average End-to-End Loss: {avg_loss:.4f} ---")

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), f"models/deeplpf_epoch_{epoch}.pth")

    model_save_path = "models/deeplpf.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model Training Successfully Concluded. Model saved to {model_save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pipeline 1: DeepLPF Models")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/labels.csv",
        help="Transformed CSV file tracking matching datasets.",
    )
    parser.add_argument(
        "--scale_dataset",
        type=int,
        default=100,
        help="Max number of samples to limit (use 0 for entirely full dataset training)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning Rate configuration"
    )

    args = parser.parse_args()

    samples = None if args.scale_dataset <= 0 else args.scale_dataset

    train_deeplpf(
        csv_path=args.csv_path,
        max_samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
