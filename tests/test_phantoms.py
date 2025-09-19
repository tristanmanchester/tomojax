import numpy as np

from tomojax.data.phantoms import lamino_disk
from tomojax.core.geometry.lamino import laminography_axis_unit


def test_lamino_disk_thickness_centered():
    nx, ny, nz = 32, 64, 32
    ratio = 0.2
    tilt_deg = 35.0
    vol = lamino_disk(
        nx,
        ny,
        nz,
        thickness_ratio=ratio,
        seed=0,
        min_size=4,
        max_size=12,
        tilt_deg=tilt_deg,
        tilt_about="x",
    )

    assert vol.shape == (nx, ny, nz)
    assert np.isclose(vol.max(), 1.0, atol=1e-6)

    axis = laminography_axis_unit(tilt_deg, "x").astype(np.float32)
    coords = np.argwhere(vol > 1e-4).astype(np.float32)
    assert coords.size > 0

    center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0], dtype=np.float32)
    rel = coords - center
    dist = rel @ axis
    expected = max(1, int(round(ratio * ny))) * 0.5 + 0.75
    assert dist.max() <= expected
    assert dist.min() >= -expected
