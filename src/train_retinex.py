import argparse
import os
import time
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_deeplpf import FitzpatrickDataset, SSIMLoss

from utils import setup_logger

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class FiLMLayer(nn.Module):
    """
    Featurewise Linear Modulation (FiLM) layer logic explicitly injected at bottlenecks.
    Transforms structural spatial components via scale mapping parameters (S).
    """

    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features * 2),
        )

    def forward(self, feature_map, s):
        """
        Calculates conditional manipulation across embedded logic sequences naturally.
        """
        params = self.mlp(s)
        gamma, beta = params.chunk(2, dim=1)

        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        return feature_map * gamma + beta

class RetinexUNet(nn.Module):
    """
    Stage 1 Configuration: RetinexDIP architectural U-Net mapped functionally securely.
    Synthesizes mapping isolated context to capture Illumination alongside normalized Base properties safely.
    """

    def __init__(self):
        super(RetinexUNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU()
        )
        self.film = FiLMLayer(256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.out_L = nn.Conv2d(32, 1, 1)
        self.out_R = nn.Conv2d(32, 3, 1)

    def forward(self, x, s):
        """
        Runs complete sequence mapping outputs into Reflectance alongside illumination safely.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)
        b = self.film(b, s)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        L = torch.sigmoid(self.out_L(d1))
        R = torch.sigmoid(self.out_R(d1))

        return L, R

class DeterministicCDH(nn.Module):
    """
    Stage 2 Configuration: Operates safely mathematically isolating frozen continuous mappings natively seamlessly.
    Injects completely parameter-free Convolutional blocks dynamically producing features correctly configured over image.
    """

    def __init__(self):
        super(DeterministicCDH, self).__init__()

        self.extractor = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False, groups=3)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )

        weights = torch.zeros(6, 1, 3, 3)

        weights[0, 0, :, :] = sobel_x
        weights[1, 0, :, :] = sobel_y
        weights[2, 0, :, :] = sobel_x
        weights[3, 0, :, :] = sobel_y
        weights[4, 0, :, :] = sobel_x
        weights[5, 0, :, :] = sobel_y

        self.extractor.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        """
        Runs mapping safely producing identical derivative results logically statically.
        """
        return self.extractor(x)

class FuzzyActivation(nn.Module):
    """
    Implements a custom PyTorch parameter-free mathematical configuration producing Fuzzy Logic Curves smoothly.
    Matches the Gaussian membership definition organically resolving logic structure smoothly correctly natively.
    """

    def __init__(self, c=0.0, sigma=1.0):
        super(FuzzyActivation, self).__init__()
        self.c = float(c)
        self.sigma = float(sigma)

    def forward(self, x):
        """
        Processes standard tensor inputs logically generating isolated output ranges precisely.
        """
        return torch.exp(-torch.pow(x - self.c, 2) / (2 * (self.sigma**2)))

class FuzzyCNN(nn.Module):
    """
    Stage 3 Configuration: Custom refinement block tracking directly representations configured cleanly optimally safely.
    Handles extraction artifacts accurately matching input characteristics producing image output cleanly rationally dynamically.
    """

    def __init__(self):
        super(FuzzyCNN, self).__init__()

        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.fuzzy_act = FuzzyActivation(c=0.0, sigma=1.0)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x_concat):
        """
        Calculates cleanly against features combining stage mappings identically predictably.
        """
        f = self.conv1(x_concat)
        f = self.fuzzy_act(f)

        f = self.conv2(f)
        f = self.relu2(f)

        f = self.out_conv(f)
        return torch.sigmoid(f)

class HybridRetinexFuzzyModel(nn.Module):
    """
    Advanced Pipeline Native Configuration mathematically tracking constraints statically matching design documentation.
    Chains cleanly RetinexDIP architectural blocks mapping cleanly to Deterministic tracking onto output CNNs securely dynamically.
    """

    def __init__(self):
        super(HybridRetinexFuzzyModel, self).__init__()
        self.stage1 = RetinexUNet()
        self.stage2 = DeterministicCDH()
        self.stage3 = FuzzyCNN()

    def forward(self, x, s):
        """
        Runs output execution properly matching identically natively fully configured sequence cleanly efficiently predictably.
        """
        L, R_adj = self.stage1(x, s)
        T_maps = self.stage2(R_adj)
        combined_features = torch.cat([R_adj, T_maps, x], dim=1)
        output = self.stage3(combined_features)

        return output, L

def tv_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    )

def train_hybrid_pipeline(csv_path, max_samples=None, epochs=10, batch_size=4, lr=1e-4):
    """
    Initiates isolated learning infrastructure naturally targeting Pipeline 2 logical operations robustly securely appropriately.
    """
    log_file = "logs/training_retinex.log"
    setup_logger(log_file)
    logging.info(f"Starting training with device: {device}")
    os.makedirs("models", exist_ok=True)
    logging.info(f"Loading dataset from {csv_path} with max_samples={max_samples}")
    dataset = FitzpatrickDataset(csv_path, max_samples=max_samples)

    if len(dataset) == 0:
        logging.error("Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    logging.info(f"Initializing Context -> Using device: {device}")
    model = HybridRetinexFuzzyModel().to(device)

    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = SSIMLoss().to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=epochs
    )

    logging.info(f"Initiating Training. Device: {device}")
    
    total_train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (t_img, g_img, s_val) in enumerate(dataloader):
            t_img = t_img.to(device)
            g_img = g_img.to(device)
            s_val = s_val.to(device)

            optimizer.zero_grad()

            output, L_map = model(t_img, s_val)
            loss_tv = tv_loss(L_map)

            loss_l1 = l1_loss_fn(output, g_img)
            loss_ssim = ssim_loss_fn(output, g_img)

            loss = loss_l1 + (5.0 * loss_ssim) + (0.1 * loss_tv)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch [{epoch}/{epochs - 1}] Batch [{batch_idx}/{len(dataloader) - 1}] Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logging.info(f"--- Epoch {epoch} Average End-to-End Loss: {avg_loss:.4f} ---")

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), f"models/retinex_epoch_{epoch}.pth")

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        logging.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

    total_train_end = time.time()
    total_train_duration = total_train_end - total_train_start
    logging.info(f"Total training completed in {total_train_duration:.2f} seconds")

    model_save_path = "models/retinex.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model Training Successfully Concluded. Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Pipeline 2: Hybrid Retinex-Fuzzy-CNN Architecture"
    )

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

    train_hybrid_pipeline(
        csv_path=args.csv_path,
        max_samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
