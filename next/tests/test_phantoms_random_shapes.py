import sys
import numpy as np
import pytest

from tomojax_next.data.phantoms import random_cubes_spheres


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def test_random_shapes_deterministic_and_bounds():
    nx = ny = nz = 32
    vol1 = random_cubes_spheres(nx, ny, nz, n_cubes=4, n_spheres=4, min_size=3, max_size=8, min_value=0.2, max_value=0.9, seed=123)
    vol2 = random_cubes_spheres(nx, ny, nz, n_cubes=4, n_spheres=4, min_size=3, max_size=8, min_value=0.2, max_value=0.9, seed=123)
    assert np.allclose(vol1, vol2)
    assert vol1.shape == (nx, ny, nz)
    assert np.min(vol1) >= 0.0
    assert np.max(vol1) <= 1.0

