import numpy as np

from tomojax.data.phantoms import lamino_disk


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
    zs = np.argwhere(vol.max(axis=(0,1)) > 1e-4).ravel()
    assert zs.size > 0
    cz = (nz - 1) / 2.0
    dist = np.abs(zs - cz)
    expected = max(1, int(round(ratio * nz))) * 0.5 + 0.75
    assert dist.max() <= expected
