"""Optional independent corpus meshing conventions and binary saddle regression."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))
pytest.importorskip("trimesh")


def test_material_mesh_preserves_xyz_centres_and_hollow_orientation(tmp_path):
    from mesh_voxels import check_closed, occupancy, read_stl
    from rock_mesh import material_mesh

    labels = np.zeros((12, 16, 20), np.uint8)
    labels[2:10, 3:13, 5:19] = 1
    labels[4:8, 6:10, 8:16] = 0
    mesh = material_mesh(labels, 1, 0.1)
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [0.2, 0, 0], atol=1e-7)
    mesh.export(tmp_path / "shell.stl")
    triangles = read_stl(tmp_path / "shell.stl")
    assert check_closed(triangles)["open_edges"] == 0
    truth = occupancy(triangles, labels.shape, (0.1,) * 3, (-0.95, -0.75, -0.55))
    assert truth[5, 7, 10] == 0 and truth[5, 4, 10] == 1
    assert np.sum(truth) * 0.1**3 == pytest.approx(mesh.volume, rel=0.01)


def test_material_mesh_resolves_binary_saddle_contacts():
    from rock_mesh import material_mesh

    labels = (np.random.default_rng(0).random((30, 30, 30)) > 0.8).astype(np.uint8)
    mesh = material_mesh(labels, 1, 0.01)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0


def test_multimaterial_tetrahedra_are_disjoint():
    from mesh_voxels import occupancy
    from rock_mesh import material_mesh

    labels = np.random.default_rng(8).integers(0, 4, (8, 9, 10), dtype=np.uint8)
    fractions = np.zeros_like(labels, dtype=float)
    for label in range(1, 4):
        mesh = material_mesh(labels, label, 0.1)
        fractions += occupancy(
            mesh.triangles, labels.shape, (0.1,) * 3, (-0.45, -0.4, -0.35), supersample=3
        )
    assert fractions.max() <= 1 + 1e-6
