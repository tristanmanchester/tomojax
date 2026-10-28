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


@pytest.mark.parametrize("shape", [(4, 4, 4), (5, 4, 3)])
def test_box_fills_each_voxel_by_its_overlap(shape):
    lo, hi = np.array([0.3, 0.55, 0.2]), np.array([2.7, 1.8, 3.35])
    filled = occupancy(_box(lo, hi), shape, (1.0,) * 3, (0.0,) * 3, supersample=4)
    # Four sub-columns a voxel across x and y; the overlap along z is exact.
    across = []
    for a, length in enumerate(shape[:0:-1]):
        centres = (np.arange(4 * length) + 0.5) / 4 - 0.5
        across.append(((centres >= lo[a]) & (centres <= hi[a])).reshape(length, 4).mean(1))
    edges = np.arange(shape[0] + 1) - 0.5
    along = np.clip(np.minimum(hi[2], edges[1:]) - np.maximum(lo[2], edges[:-1]), 0, None)
    np.testing.assert_allclose(filled, np.einsum("i,j,k->kji", *across, along), atol=1e-12)


def test_open_mesh_is_refused():
    bottomless = np.delete(_box([0.2] * 3, [1.4] * 3), 4, axis=0)  # half the base
    with pytest.raises(ValueError, match="not closed"):
        occupancy(bottomless, (2, 2, 2), (1.0,) * 3, (0.0,) * 3)


def test_watch_truth_uses_physical_xyz_origin(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import mesh_voxels

    import _watch

    observed = {}

    def occupy(_mesh, shape, _spacing, origin, **_kwargs):
        observed.update(shape=shape, origin=origin)
        return np.zeros(shape)

    monkeypatch.setattr(_watch, "MU", {"test": 1.0})
    monkeypatch.setattr(mesh_voxels, "read_stl", lambda _: None)
    monkeypatch.setattr(mesh_voxels, "occupancy", occupy)
    actual = _watch.truth(Path("."), (2, 3, 5), 0.5, (1.0, 2.0, 3.0))
    assert actual.shape == observed["shape"] == (2, 3, 5)
    assert observed["origin"] == (0.0, 1.5, 2.75)


def test_volume_renderer_samples_zyx_in_physical_coordinates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import jax.numpy as jnp

    import _volume_render as vr

    volume = np.broadcast_to(np.arange(5, dtype=np.float32), (2, 3, 5))
    data = vr.fields(volume, 1.0, np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), 4.0)
    np.testing.assert_allclose(data["lit"], 1.25)
    rays = tuple(jnp.asarray(v) for v in ([[3.5, 1.5, 0.0]], [[0.0, 0.0, 1.0]], [0.0]))
    table = tuple(jnp.asarray(v) for v in ([0.0, 4.0], [[1.0, 0, 0], [1.0, 0, 0]], [0.0, 4.0]))
    actual = vr._march(data, rays, table, (jnp.zeros(3), 1.0, 1.0, 0.0), 1)
    np.testing.assert_allclose(actual, [[1.25 * (1 - np.exp(-3)), 0, 0, np.exp(-3)]], atol=1e-6)
