import torch

from fitzpatrick_optimizer.models import (
    IlluminationGuidedUNet,
    ParameterConditionedResidualFilter,
)


def test_parameter_conditioned_residual_filter_output_shape_and_range():
    model = ParameterConditionedResidualFilter(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    s = torch.tensor([[0.0], [1.0]])

    y = model(x, s)

    assert y.shape == x.shape
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_illumination_guided_unet_output_shape_and_illumination_map():
    model = IlluminationGuidedUNet()
    x = torch.rand(2, 3, 64, 64)
    s = torch.tensor([[0.2], [0.8]])

    y, illumination = model(x, s)

    assert y.shape == x.shape
    assert illumination.shape == (2, 1, 64, 64)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
