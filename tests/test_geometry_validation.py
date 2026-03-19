import pytest

from tomojax.core.geometry import Detector, Grid


def test_grid_validates_optional_volume_tuples():
    with pytest.raises(ValueError, match="vol_origin must have length 3"):
        Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0, vol_origin=(0.0, 0.0))

    with pytest.raises(ValueError, match="vol_center must have length 3"):
        Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0, vol_center=(0.0, 0.0))


def test_detector_validates_centre_tuple():
    with pytest.raises(ValueError, match="det_center must have length 2"):
        Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0,))
