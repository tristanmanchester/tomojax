from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main
from tomojax.recon.quicklook import extract_central_slice, save_quicklook_png, scale_to_uint8


class _VolumeOnlySlices:
    """A resident/lazy volume whose full host mirror exceeds available storage."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.selections = []

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("Quicklook downloaded the whole volume")

    def __getitem__(self, selection):
        self.selections.append(selection)
        return self.array[selection]


@pytest.mark.parametrize("depth", [1, 4, 5])
def test_quicklook_downloads_only_the_selected_plane(depth):
    array = np.arange(7 * 6 * depth, dtype=np.float32).reshape(7, 6, depth) / 100
    volume = _VolumeOnlySlices(array)
    actual = extract_central_slice(volume)
    np.testing.assert_array_equal(actual, array[:, :, depth // 2].T)
    assert volume.selections == [(slice(None), slice(None), depth // 2)]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.uint16])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_quicklook_preserves_numpy_dtype_alias_and_orientation(dtype, layout):
    volume = np.arange(7 * 6 * 5, dtype=dtype).reshape(7, 6, 5)
    if layout == "F":
        volume = np.asfortranarray(volume)
    elif layout == "strided":
        volume = volume[::-1, ::2, :]
    elif layout == "readonly":
        volume.setflags(write=False)
    actual = extract_central_slice(volume)
    assert actual.dtype == volume.dtype
    assert np.shares_memory(actual, volume)
    assert actual.flags.writeable == volume.flags.writeable
    np.testing.assert_array_equal(actual, volume[:, :, 2].T)


@pytest.mark.parametrize("shape", [(7, 6), (7, 6, 4), (7, 6, 5)])
def test_quicklook_keeps_existing_array_like_inputs(shape):
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = volume.T if volume.ndim == 2 else volume[:, :, shape[2] // 2].T
    np.testing.assert_array_equal(extract_central_slice(volume.tolist()), expected)


@pytest.mark.parametrize("shape", [(), (7,), (2, 3, 4, 5)])
def test_quicklook_keeps_invalid_dimension_errors(shape):
    with pytest.raises(ValueError, match="quicklook expects a 2D image or 3D volume"):
        extract_central_slice(np.zeros(shape))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [1, 4, 5])
def test_quicklook_jax_plane_is_bit_exact(dtype, depth):
    array = np.arange(7 * 6 * depth).astype(dtype).reshape(7, 6, depth) / 100
    with jax.enable_x64():
        volume = jnp.asarray(array)
        actual = extract_central_slice(volume)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, array[:, :, depth // 2].T)


def test_quicklook_png_is_the_same_complete_image(tmp_path: Path):
    array = np.random.default_rng(124).normal(0.2, 0.01, (7, 6, 5)).astype(np.float32)
    volume = _VolumeOnlySlices(array)
    path = tmp_path / "preview" / "slice.png"
    assert save_quicklook_png(path, volume) == path
    np.testing.assert_array_equal(iio.imread(path), scale_to_uint8(array[:, :, 2].T))
    assert volume.selections == [(slice(None), slice(None), 2)]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.uint16])
@pytest.mark.parametrize("layout", ["C", "transposed", "strided"])
@pytest.mark.parametrize(("lower", "upper"), [(1.0, 99.0), (0.0, 100.0), (22.33, 65.1)])
def test_quicklook_scaling_preserves_pixels_and_readonly_input(dtype, layout, lower, upper):
    rng = np.random.default_rng(253)
    if np.issubdtype(dtype, np.floating):
        image = rng.normal(0.1, 0.001, (31, 29)).astype(dtype)
        image.ravel()[::19] = np.nan
        image.ravel()[::23] = np.inf
    else:
        image = rng.integers(0, 1024, (31, 29)).astype(dtype)
    if layout == "transposed":
        image = image.T
    elif layout == "strided":
        image = image[::-1, ::2]
    original = image.copy()
    image.setflags(write=False)
    # Independent expression for the existing FP32 percentile/rounding policy.
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    values = arr[finite]
    lo, hi = map(float, np.percentile(values, [lower, upper]))
    expected = np.zeros(arr.shape, np.uint8)
    expected[finite] = np.round(np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    actual = scale_to_uint8(image, lower_percentile=lower, upper_percentile=upper)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("value", [0.2, np.nan, np.inf])
def test_quicklook_constant_or_invalid_pixels_remain_black(value):
    image = np.full((3, 4), value, dtype=np.float32)
    image.setflags(write=False)
    np.testing.assert_array_equal(scale_to_uint8(image), np.zeros((3, 4), dtype=np.uint8))


def test_quicklook_scaling_preserves_empty_image():
    np.testing.assert_array_equal(scale_to_uint8(np.zeros((0, 0))), np.zeros((0, 0), np.uint8))


def test_recon_cli_keeps_device_volume_until_quicklook_selects_a_plane(monkeypatch, tmp_path):
    grid = tj.Grid(7, 6, 5, 1.0, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 5, 1.0, 1.0), [0.0, 60.0, 120.0])
    scan = tj.Scan(np.ones((3, 5, 7), np.float32), geometry)
    array = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5) / 100
    volume = _VolumeOnlySlices(array)
    result = tj.Reconstruction(volume, scan, "fbp", {})
    monkeypatch.setattr(tj, "load", lambda *args, **kwargs: scan)
    monkeypatch.setattr(tj, "reconstruct", lambda *args, **kwargs: result)
    saved = []
    monkeypatch.setattr(tj, "save", lambda path, item: saved.append(item))
    path = tmp_path / "scan.nxs"
    path.touch()
    preview = tmp_path / "preview.png"
    assert (
        main(
            [
                "recon",
                str(path),
                "-o",
                str(tmp_path / "result.nxs"),
                "--roi",
                "off",
                "--preview",
                str(preview),
            ]
        )
        == 0
    )
    assert saved == [result]
    assert volume.selections == [(slice(None), slice(None), 2)]
    np.testing.assert_array_equal(iio.imread(preview), scale_to_uint8(array[:, :, 2].T))
