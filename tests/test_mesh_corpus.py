"""Independent corpus metadata, noise, motion and physical mesh partition contracts."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))
from corpus import moved_poses, pack, prepare
from mesh_voxels import occupancy, read_stl


def cube_mesh(path):
    vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    faces = np.array(
        [
            [0, 3, 1],
            [0, 2, 3],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ]
    )
    records = np.zeros(12, dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    records["v"] = vertices[faces]
    path.write_bytes(bytes(80) + np.array([12], dtype="<u4").tobytes() + records.tobytes())


def test_motion_is_rigid_deterministic_and_physical():
    nominal = np.tile(np.eye(4), (17, 1, 1))
    moved = moved_poses(nominal, 0.3)
    np.testing.assert_array_equal(moved, moved_poses(nominal, 0.3))
    np.testing.assert_allclose(
        moved[:, :3, :3] @ moved[:, :3, :3].transpose(0, 2, 1), nominal[:, :3, :3], atol=1e-15
    )
    np.testing.assert_allclose(np.linalg.det(moved[:, :3, :3]), 1)
    np.testing.assert_allclose(moved_poses(nominal, 0.6)[:, :3, 3], 2 * moved[:, :3, 3])
    np.testing.assert_array_equal(nominal, np.tile(np.eye(4), (17, 1, 1)))
    assert np.ptp(moved[:, 0, 3]) > 0.3 and np.all(moved[:, 1, 3] == 0)


@pytest.mark.parametrize("kind", ["parallel", "laminography", "cone"])
def test_prepare_and_pack_preserve_truth_geometry_and_noise(tmp_path, kind):
    import tomojax as tj

    cube_mesh(tmp_path / "box.stl")
    grid = tj.Grid(4, 4, 4, 0.5, 0.5, 0.5)
    detector = tj.Detector(3, 2, 0.5, 0.5, center=(0.2, -0.3))
    config = dict(
        geometry=dict(
            kind=kind,
            grid=grid.to_dict(),
            detector=detector.to_dict(),
            angles_deg=[0, 60, 120],
            tilt_deg=-35,
        ),
        meshes=str(tmp_path),
        interior_meshes=str(tmp_path),
        exterior_meshes=str(tmp_path),
        materials=[dict(name="box", mu_mm=0.2)],
        reference_energy_keV=120,
        seed=7,
    )
    if kind == "cone":
        config["geometry"]["beam"] = dict(source_to_axis=10, source_to_detector=20)
    out = tmp_path / "case"
    prepare(config, out)
    np.testing.assert_allclose(np.load(out / "truth.npy"), 0.2)
    plan = np.load(out / "clean-plan.npz")
    np.testing.assert_array_equal(plan["shape"], [2, 3])
    np.testing.assert_allclose(plan["centre_mm"], [0.2, -0.3])
    assert json.loads((out / "config.json").read_text())["held_out_indices"] == [2]
    for name in ("clean", "clean-motion", "exterior", "motion", "exterior-only", "moving-exterior"):
        tr = np.full((3, 2, 3), 0.8, dtype=np.float32)
        if name == "clean":
            tr[:, 0, 0] = 0
        np.savez(out / f"{name}-render.npz", transmission=tr, gvxr="test simulator")
    pack(out, [100])
    with np.load(out / "clean-100.npz") as data:
        assert np.isfinite(data["projections"]).all()
        np.testing.assert_allclose(data["projections"][:, 0, 0], -np.log(0.5 / 100))
        assert not data["exterior_projection"].any()
        first = data["projections"].copy()
    pack(out, [100])
    np.testing.assert_array_equal(first, np.load(out / "clean-100.npz")["projections"])
    with np.load(out / "motion-100.npz") as data:
        np.testing.assert_allclose(data["exterior_projection"], -np.log(0.8))
        assert not np.array_equal(data["poses"], data["nominal_poses"])


def test_blender_partition_preserves_physical_mass(tmp_path):
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender is optional benchmark software")
    source = tmp_path / "meshes"
    source.mkdir()
    cube_mesh(source / "box.stl")
    script = Path(__file__).resolve().parents[1] / "bench/phantoms/partition_meshes_blender.py"
    out = tmp_path / "partition"
    subprocess.run(
        [
            blender,
            "-b",
            "--factory-startup",
            "-P",
            str(script),
            "--",
            str(source),
            str(out),
            "--centre",
            "0.5",
            "0",
            "0",
            "--extent",
            "2",
            "2",
            "2",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    volumes = [
        occupancy(read_stl(out / side / "box.stl"), (4, 4, 4), (0.5,) * 3, (-0.75,) * 3)
        for side in ("interior", "exterior")
    ]
    np.testing.assert_allclose(volumes[0] + volumes[1], 1, atol=1e-12)
    assert volumes[0].sum() == pytest.approx(48)
    assert volumes[1].sum() == pytest.approx(16)
