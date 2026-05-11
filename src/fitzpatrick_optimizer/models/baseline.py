import torch
from torch import nn


class IdentityBaseline(nn.Module):
    """No-op image restoration baseline.

    Returns the input image unchanged.
    """

    def forward(
        self,
        x: torch.Tensor,
        fitzpatrick_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x