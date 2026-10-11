"""The eager affine boundary preserves physical coordinates and zero extension."""

from __future__ import annotations

import importlib.util

import jax
import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.transform import Rotation

from tomojax.core import resampling
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.resampling import affine_resample

pytestmark = pytest.mark.numerical


def _matrix(shape, motion):
    spacing = np.asarray([0.7, 1.3, 0.9])
    origin = np.asarray([0.23, -0.17, 0.31]) - (np.asarray(shape) - 1) * spacing / 2
    rotation = Rotation.from_rotvec([0.02, -0.013, 0.007] if motion == "rigid" else [0, 0, 0])
    rotation = rotation.as_matrix()
    shift = {
        "identity": [0, 0, 0],
        "sliver": [1e-4, -1e-4, 0],
        "rigid": [0.37, -0.21, 0.13],
        "outside": [100, -100, 100],
    }[motion]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * spacing[None, :] / spacing[:, None]
    matrix[:3, 3] = (rotation @ origin + shift - origin) / spacing
    return matrix


def _eight_taps(volume, matrix):
    """Independent FP64 trilinear sum, including taps outside the grid."""
    indices = np.indices(volume.shape).reshape(3, -1)
    coords = matrix[:3, :3] @ indices + matrix[:3, 3, None]
    lower = np.floor(coords).astype(int)
    fraction = coords - lower
    result = np.zeros(indices.shape[1])
    for corner in np.ndindex(2, 2, 2):
        tap = lower + np.asarray(corner)[:, None]
        valid = np.all((tap >= 0) & (tap < np.asarray(volume.shape)[:, None]), axis=0)
        weight = np.prod(np.where(np.asarray(corner)[:, None], fraction, 1 - fraction), axis=0)
        result[valid] += weight[valid] * volume[tuple(tap[:, valid])]
    return result.reshape(volume.shape).astype(np.float32)


@pytest.fixture(params=["cpu", "cuda"])
def backend(request, monkeypatch):
    if request.param == "cuda":
        if not cuda_backend() or importlib.util.find_spec("cupy") is None:
            pytest.skip("CUDA/CuPy resampling requires an NVIDIA GPU")
        monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    else:
        monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "0")
    return request.param


@pytest.mark.parametrize("shape", [(13, 12, 11), (1, 9, 7), (7, 1, 1)])
@pytest.mark.parametrize("motion", ["identity", "sliver", "rigid", "outside"])
@pytest.mark.parametrize("device_input", [False, True])
def test_physical_affines_match_independent_zero_extended_interpolation(
    backend, shape, motion, device_input
):
    volume = np.random.default_rng(751).uniform(-0.1, 1.1, shape).astype(np.float32)
    matrix = _matrix(shape, motion)
    expected = _eight_taps(volume, matrix)
    actual = affine_resample(jax.device_put(volume) if device_input else volume, matrix)
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    assert actual.shape == volume.shape
    np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=4e-7)
    np.testing.assert_allclose(
        actual,
        ndimage.affine_transform(volume, matrix, order=1, mode="grid-constant"),
        atol=4e-7,
        rtol=4e-7,
    )


def test_affine_parameter_bits_survive_disabled_jax_x64_and_noncontiguous_storage(backend):
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    # Cancellation makes this row's sub-FP32 coefficient bits observable.
    # Encoding coordinates must not depend on jax_enable_x64.
    matrix = np.eye(4)
    matrix[0, 0], matrix[0, 3] = 1e8 + 0.125, -2e8
    matrix = np.asfortranarray(matrix)
    with jax.enable_x64(False):
        actual = affine_resample(volume, matrix)
    np.testing.assert_allclose(actual, _eight_taps(volume, matrix), atol=2e-6, rtol=1e-7)


def test_explicit_cpu_array_retains_scipy_path_on_a_cuda_default_backend(monkeypatch):
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    cpu = jax.devices("cpu")[0]
    volume = np.random.default_rng(83).normal(size=(13, 12, 11)).astype(np.float32)
    matrix = _matrix(volume.shape, "rigid")
    expected = ndimage.affine_transform(volume, matrix, order=1, mode="grid-constant")
    actual = affine_resample(jax.device_put(volume, cpu), matrix)
    np.testing.assert_array_equal(actual, expected)
    with jax.default_device(cpu):
        np.testing.assert_array_equal(affine_resample(volume, matrix), expected)


def test_missing_optional_cupy_retains_scipy(monkeypatch):
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    monkeypatch.setattr(resampling, "cuda_backend", lambda: True)
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "cupy" else find_spec(name)
    )
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    matrix = _matrix(volume.shape, "rigid")
    expected = ndimage.affine_transform(volume, matrix, order=1, mode="grid-constant")
    np.testing.assert_array_equal(affine_resample(volume, matrix), expected)
