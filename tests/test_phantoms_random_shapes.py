import hashlib
import os

import numpy as np
import pytest

from tomojax.data.phantoms import random_cubes_spheres


_SMALL_CASE = dict(
    nx=32,
    ny=32,
    nz=32,
    n_cubes=4,
    n_spheres=4,
    min_size=4,
    max_size=12,
    min_value=0.03,
    max_value=0.9,
    seed=123,
)

_MEDIUM_CASE = dict(
    nx=128,
    ny=128,
    nz=128,
    n_cubes=24,
    n_spheres=24,
    min_size=4,
    max_size=24,
    min_value=0.03,
    max_value=0.9,
    seed=17,
)

_HEAVY_CASE = dict(
    nx=192,
    ny=192,
    nz=144,
    n_cubes=48,
    n_spheres=48,
    min_size=4,
    max_size=28,
    min_value=0.05,
    max_value=1.0,
    seed=81,
)

_EXPECTED_SHA256 = {
    "small": "57a2092790a55833144169ed56cbd8d4845d83700ea52afeb57f3d4784ec9889",
    "medium": "656c3669ec605e683db51822d668b78a9a5da8e4c46f72cd5c385fb4b6fbfa77",
    "heavy": "059afc0074ab5401ac6abefea15f1cb7c728fee7644f6972a3b00d2c25b9737f",
}


def _hash_volume(vol: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vol).tobytes()).hexdigest()


@pytest.mark.parametrize("case", [_SMALL_CASE, _MEDIUM_CASE])
def test_random_shapes_deterministic_and_bounds(case):
    vol1 = random_cubes_spheres(**case, use_inscribed_fov=True)
    vol2 = random_cubes_spheres(**case, use_inscribed_fov=True)
    assert np.array_equal(vol1, vol2)
    assert vol1.shape == (case["nx"], case["ny"], case["nz"])
    assert np.min(vol1) >= 0.0
    assert np.max(vol1) <= 1.0


@pytest.mark.parametrize(
    ("name", "case"),
    [("small", _SMALL_CASE), ("medium", _MEDIUM_CASE)],
)
def test_random_shapes_matches_expected_hash(name, case):
    vol = random_cubes_spheres(**case, use_inscribed_fov=True)
    assert _hash_volume(vol) == _EXPECTED_SHA256[name]


@pytest.mark.skipif(
    not bool(int(os.environ.get("TOMOJAX_RUN_HEAVY_PHANTOM_TESTS", "0"))),
    reason="set TOMOJAX_RUN_HEAVY_PHANTOM_TESTS=1 to run heavy phantom regression checks",
)
def test_random_shapes_heavy_matches_expected_hash():
    vol = random_cubes_spheres(**_HEAVY_CASE, use_inscribed_fov=True)
    assert _hash_volume(vol) == _EXPECTED_SHA256["heavy"]
