import torch
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features * 2),
        )

    def forward(self, feature_map: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(s).chunk(2, dim=1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        return feature_map * gamma + beta


class IlluminationUNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.out_l = nn.Conv2d(32, 1, 1)
        self.out_r = nn.Conv2d(32, 3, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.film(self.bottleneck(e3), s)
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_l(d1)), torch.sigmoid(self.out_r(d1))


class SobelTextureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False, groups=3)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )
        weights = torch.zeros(6, 1, 3, 3)
        for channel in range(3):
            weights[channel * 2, 0] = sobel_x
            weights[channel * 2 + 1, 0] = sobel_y
        self.extractor.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class GaussianMembershipActivation(nn.Module):
    def __init__(self, center: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.center = center
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.pow(x - self.center, 2) / (2 * (self.sigma**2)))


class RefinementCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            GaussianMembershipActivation(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IlluminationGuidedUNet(nn.Module):
    """FiLM-conditioned U-Net with Sobel texture features and refinement CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.illumination_backbone = IlluminationUNetBackbone()
        self.texture_extractor = SobelTextureExtractor()
        self.refinement = RefinementCNN()

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination, reflectance = self.illumination_backbone(x, s)
        texture = self.texture_extractor(reflectance)
        output = self.refinement(torch.cat([reflectance, texture, x], dim=1))
        return output, illumination
