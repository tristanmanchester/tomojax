"""Procedural documentation phantoms build quickly, deterministically and in physical units."""

from collections.abc import Callable
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))

Builder = Callable[..., tuple[np.ndarray, dict[str, Any]]]
NAMES = ["porous_rock", "battery", "lattice_part"]
pytestmark = pytest.mark.slow


def _builder(name: str) -> Builder:
    return getattr(importlib.import_module(name), name)


@pytest.mark.parametrize("name", NAMES)
def test_phantom_is_deterministic_bounded_and_scale_free(name: str) -> None:
    build = _builder(name)
    volume, meta = build(24, seed=3)
    again, _ = build(24, seed=3)
    coarse, coarse_meta = build(32, seed=3)

    assert volume.shape == (24, 24, 24)
    assert volume.dtype == np.float32
    np.testing.assert_array_equal(volume, again)
    assert meta["seed"] == 3
    assert meta["voxel_mm"] == pytest.approx(meta["fov_mm"] / 24)
    mu = np.array(list(meta["materials"].values()))
    assert np.all(np.isfinite(volume))
    assert volume.min() >= 0
    assert volume.max() <= mu.max() * (1 + 1e-5)
    # Several materials appear, so the volume is not a single-valued mask.
    assert np.count_nonzero(np.histogram(volume[volume > 0], bins=64)[0]) > 8
    # The object is fixed in mm: a coarser grid integrates to the same total.
    total = volume.sum(dtype=np.float64) * meta["voxel_mm"] ** 3
    coarse_total = coarse.sum(dtype=np.float64) * coarse_meta["voxel_mm"] ** 3
    assert coarse_total == pytest.approx(total, rel=0.03)
