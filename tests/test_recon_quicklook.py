from __future__ import annotations

import numpy as np
import pytest

from tomojax.recon.quicklook import extract_central_slice, scale_to_uint8


def test_extract_central_slice_uses_central_z_slice_and_display_orientation() -> None:
    volume = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)

    image = extract_central_slice(volume)

    np.testing.assert_array_equal(image, volume[:, :, 2].T)
    assert image.shape == (3, 2)


def test_extract_central_slice_accepts_2d_images() -> None:
    image_xy = np.arange(6, dtype=np.float32).reshape(2, 3)

    image = extract_central_slice(image_xy)

    np.testing.assert_array_equal(image, image_xy.T)
    assert image.shape == (3, 2)


def test_extract_central_slice_rejects_unsupported_dimensions() -> None:
    with pytest.raises(ValueError, match="quicklook expects a 2D image or 3D volume"):
        extract_central_slice(np.zeros((2, 3, 4, 5), dtype=np.float32))


def test_scale_to_uint8_percentile_scales_clips_and_maps_nonfinite_to_zero() -> None:
    image = np.asarray(
        [
            [-100.0, 0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0, 100.0],
            [np.nan, np.inf, -np.inf, 2.5],
        ],
        dtype=np.float32,
    )

    scaled = scale_to_uint8(image, lower_percentile=10.0, upper_percentile=90.0)

    assert scaled.dtype == np.uint8
    assert scaled.shape == image.shape
    assert scaled[0, 0] == 0
    assert scaled[1, 3] == 255
    assert scaled[2, 0] == 0
    assert scaled[2, 1] == 0
    assert scaled[2, 2] == 0
    assert 0 < scaled[0, 2] < 255


def test_scale_to_uint8_constant_and_all_nonfinite_inputs_are_black() -> None:
    constant = np.full((2, 3), 7.0, dtype=np.float32)
    all_nonfinite = np.asarray([[np.nan, np.inf]], dtype=np.float32)

    assert np.array_equal(scale_to_uint8(constant), np.zeros((2, 3), dtype=np.uint8))
    assert np.array_equal(scale_to_uint8(all_nonfinite), np.zeros((1, 2), dtype=np.uint8))


def test_scale_to_uint8_rejects_invalid_percentile_bounds() -> None:
    with pytest.raises(ValueError, match="percentile bounds"):
        scale_to_uint8(np.zeros((2, 2), dtype=np.float32), lower_percentile=99.0)
