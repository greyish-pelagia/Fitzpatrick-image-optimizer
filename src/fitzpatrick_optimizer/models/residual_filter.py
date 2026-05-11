import torch
import torchvision
from torch import nn


def apply_residual_filter(
    x: torch.Tensor,
    p_grad: torch.Tensor,
    p_ellip: torch.Tensor,
    p_poly: torch.Tensor,
) -> torch.Tensor:
    batch_size = x.shape[0]
    grad_scale = torch.sigmoid(p_grad.mean(dim=1).view(batch_size, 1, 1, 1))
    a = p_poly[:, :3].view(batch_size, 3, 1, 1) * 0.1
    b = p_poly[:, 3:6].view(batch_size, 3, 1, 1) * 0.1
    c = p_poly[:, 6:9].view(batch_size, 3, 1, 1) * 0.1
    residual = a * (x**2) + b * x + c
    ellip_shift = torch.tanh(p_ellip.mean(dim=1).view(batch_size, 1, 1, 1)) * 0.1
    return torch.clamp(x + residual * grad_scale + ellip_shift, 0.0, 1.0)


class ParameterConditionedResidualFilter(nn.Module):
    """ResNet-conditioned residual image filter used as a DeepLPF-inspired baseline."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.s_embed = nn.Linear(1, 1)
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)

        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv.weight
            self.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 76))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        s_map = (
            self.s_embed(s)
            .view(batch_size, 1, 1, 1)
            .expand(batch_size, 1, height, width)
        )
        features = torch.cat([x, s_map], dim=1)
        features = self.conv1(features)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features)
        params = self.head(torch.flatten(features, 1))
        return apply_residual_filter(x, params[:, :8], params[:, 8:16], params[:, 16:])
