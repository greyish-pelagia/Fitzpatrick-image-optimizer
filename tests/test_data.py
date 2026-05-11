import numpy as np
import pytest

from fitzpatrick_optimizer.imaging import normalize_fitzpatrick_scale, to_chw_float


def test_normalize_fitzpatrick_scale_maps_1_to_0_and_6_to_1():
    assert normalize_fitzpatrick_scale(1) == 0.0
    assert normalize_fitzpatrick_scale(6) == 1.0
    assert normalize_fitzpatrick_scale(3) == pytest.approx(0.4)


def test_normalize_fitzpatrick_scale_rejects_invalid_values():
    with pytest.raises(ValueError, match="Fitzpatrick scale must be between 1 and 6"):
        normalize_fitzpatrick_scale(0)


def test_to_chw_float_converts_uint8_rgb_to_float_tensor_layout():
    image = np.array(
        [[[0, 128, 255], [255, 128, 0]]],
        dtype=np.uint8,
    )

    result = to_chw_float(image)

    assert result.shape == (3, 1, 2)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == 0.0
    assert result[1, 0, 0] == pytest.approx(128 / 255)
    assert result[2, 0, 0] == 1.0
