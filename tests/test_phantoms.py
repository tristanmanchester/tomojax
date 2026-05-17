import inspect

import numpy as np

import tomojax.datasets as datasets_api
from tomojax.datasets import (
    blobs,
    cube,
    lamino_disk,
    lamino_disk_legacy,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
import tomojax.datasets.api as datasets_full_api


def test_public_phantom_api_is_reexported_from_package_root():
    for name in (
        "LaminoGeometryMeta",
        "SimMetadata",
        "apply_simulation_artefacts",
        "blobs",
        "cube",
        "lamino_disk",
        "lamino_disk_legacy",
        "random_cubes_spheres",
        "rotated_centered_cube",
        "shepp_logan_3d",
        "sphere",
    ):
        assert name in datasets_api.__all__
        assert getattr(datasets_api, name) is getattr(datasets_full_api, name)


def test_public_phantom_generators_return_float32_volumes():
    phantoms = [
        sphere(16, 16, 16),
        cube(16, 16, 16),
        rotated_centered_cube(16, 16, 16, seed=1),
        blobs(16, 16, 16, seed=1),
        shepp_logan_3d(16, 16, 16),
        random_cubes_spheres(16, 16, 16, n_cubes=2, n_spheres=2, min_size=3, max_size=6),
        lamino_disk(16, 16, 16, seed=1, min_size=4, max_size=8),
    ]

    for volume in phantoms:
        assert volume.shape == (16, 16, 16)
        assert volume.dtype == np.float32
        assert np.isfinite(volume).all()


def test_lamino_disk_thickness_centered():
    nx, ny, nz = 32, 64, 32
    ratio = 0.2
    vol = lamino_disk(
        nx,
        ny,
        nz,
        thickness_ratio=ratio,
        seed=0,
        min_size=4,
        max_size=12,
    )

    assert vol.shape == (nx, ny, nz)
    assert np.isclose(vol.max(), 1.0, atol=1e-6)

    # In sample frame, slab is orthogonal to +z, confined to central z slices
    zs = np.argwhere(vol.max(axis=(0, 1)) > 1e-4).ravel()
    assert zs.size > 0
    cz = (nz - 1) / 2.0
    dist = np.abs(zs - cz)
    expected = max(1, int(round(ratio * nz))) * 0.5 + 0.75
    assert dist.max() <= expected


def test_lamino_disk_primary_api_has_no_ignored_tilt_args():
    signature = inspect.signature(lamino_disk)

    assert "tilt_deg" not in signature.parameters
    assert "tilt_about" not in signature.parameters


def test_lamino_disk_legacy_accepts_ignored_tilt_args():
    base = lamino_disk(16, 16, 16, seed=2, min_size=4, max_size=8)
    legacy = lamino_disk_legacy(
        16,
        16,
        16,
        seed=2,
        min_size=4,
        max_size=8,
        tilt_deg=55.0,
        tilt_about="z",
    )

    np.testing.assert_allclose(legacy, base)
