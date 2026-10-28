"""The mesh voxeliser gives each voxel its exact share of a closed mesh."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))
from mesh_voxels import occupancy


def _box(lo, hi):
    """A closed, outward-facing triangle mesh of an axis-aligned box."""
    c = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([c[[a, b, k]] for a, b, k, _ in quads] + [c[[a, k, e]] for a, _, k, e in quads])


def test_box_fills_each_voxel_by_its_overlap():
    lo, hi = np.array([0.3, 0.55, 0.2]), np.array([2.7, 1.8, 3.35])
    filled = occupancy(_box(lo, hi), (4, 4, 4), (1.0,) * 3, (0.0,) * 3, supersample=4)
    # Four sub-columns a voxel across x and y; the overlap along z is exact.
    centres = (np.arange(16) + 0.5) / 4 - 0.5
    across = [((centres >= lo[a]) & (centres <= hi[a])).reshape(4, 4).mean(1) for a in (0, 1)]
    edges = np.arange(5) - 0.5
    along = np.clip(np.minimum(hi[2], edges[1:]) - np.maximum(lo[2], edges[:-1]), 0, None)
    np.testing.assert_allclose(filled, np.einsum("i,j,k->ijk", *across, along), atol=1e-12)


def test_open_mesh_is_refused():
    bottomless = np.delete(_box([0.2] * 3, [1.4] * 3), 4, axis=0)  # half the base
    with pytest.raises(ValueError, match="not closed"):
        occupancy(bottomless, (2, 2, 2), (1.0,) * 3, (0.0,) * 3)
