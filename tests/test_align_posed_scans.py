"""Aligning scans that already carry poses, multi-orbit scans among them."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj

# check-public-imports: allow-private
from tomojax._data.geometry_meta import composed_poses
from tomojax.alignment.api import apply_pose_updates
from tomojax.core.cone import cone_model
from tomojax.geometry import stack_view_poses


def _phantom(n: int) -> np.ndarray:
    c = (np.arange(n) - (n - 1) / 2) / (n / 6)
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.exp(-((x - 0.6) ** 2 + (y + 0.3) ** 2 + (z - 0.4) ** 2) * 1.5)
    volume += 0.8 * ((x**2 + y**2 + (z + 0.8) ** 2) < 0.3)
    volume += 0.5 * (np.abs(z - 1.2) < 0.15) * ((x**2 + y**2) < 2)
    return volume.astype(np.float32)


def _orbit_vectors(views: int, source_height: float, detector_height: float) -> np.ndarray:
    """ASTRA cone_vec rows of one circular orbit (source 70, detector 50 from the axis)."""
    a = np.linspace(0, 2 * np.pi, views, endpoint=False)
    s, c, zero = np.sin(a), np.cos(a), np.zeros_like(a)
    return np.stack(
        [
            s * 70, -c * 70, zero + source_height,
            -s * 50, c * 50, zero + detector_height,
            c * 1.2, s * 1.2, zero,
            zero, zero, zero + 1.2,
        ],
        axis=1,
    )  # fmt: skip


def _scan(vectors: np.ndarray, data: np.ndarray, n: int) -> tj.Scan:
    proj_geom = {"type": "cone_vec", "DetectorRowCount": 48, "DetectorColCount": 64}
    window = {f"WindowMin{a}": -n / 2 for a in "XYZ"} | {f"WindowMax{a}": n / 2 for a in "XYZ"}
    vol_geom = {"GridColCount": n, "GridRowCount": n, "GridSliceCount": n, "option": window}
    return tj.Scan.from_astra(data, proj_geom | {"Vectors": vectors}, vol_geom)


@pytest.mark.parametrize("frame", ["detector", "object"])
def test_corrections_compose_with_a_scans_own_poses(frame):
    rng = np.random.default_rng(0)
    vectors = _orbit_vectors(12, 3.0, -2.0)
    for row in vectors:  # the arrangement wobbles rigidly: per-view poses
        turn = Rotation.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix()
        shift = rng.normal(0, 0.3, 3)
        for k in range(4):
            row[3 * k : 3 * k + 3] = turn @ row[3 * k : 3 * k + 3] + (shift if k < 2 else 0)
    scan = _scan(vectors, np.zeros((48, 12, 64), np.float32), 24)
    assert scan.poses is not None
    corrections = np.concatenate(
        [rng.normal(0, 0.01, (12, 3)), rng.normal(0, 0.5, (12, 3))], axis=1
    ).astype(np.float32)

    composed = composed_poses(scan.geometry, corrections, frame)

    expected = apply_pose_updates(
        stack_view_poses(scan.geometry, 12), corrections, translation_frame=frame
    )
    # check-public-imports: allow-private
    from tomojax._data.geometry_meta import AugmentedGeometry

    nominal = scan.geometry.base  # pyright: ignore[reportAttributeAccessIssue]
    actual = stack_view_poses(AugmentedGeometry(nominal, composed, "detector"), 12)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


def test_segment_detectors_keep_their_offsets_when_binned():
    low = tj.ConeGeometry(
        tj.Grid(8, 8, 8, 1.0, 1.0, 1.0),
        tj.Detector(17, 15, 1.0, 1.0, (0.5, -1.0)),
        [0.0, 90.0],
        tj.ConeBeam(30.0, 50.0),
    )
    high = tj.ConeGeometry(
        low.grid, tj.Detector(17, 15, 1.0, 1.0, (0.5, 3.0)), [0.0, 90.0], tj.ConeBeam(32.0, 50.0)
    )
    segments = tj.geometry.ConeSegments((low, high))
    binned = tj.Detector(8, 7, 2.0, 2.0, (0.0, -1.5))

    model = cone_model(segments, binned)

    assert model is not None
    assert [part[2].center for part in model.parts] == [(0.0, -1.5), (0.0, 2.5)]
    assert [part[0] for part in model.parts] == [2, 2]
    assert model.frames(4).shape == (4, 4, 3)
    assert cone_model(tj.ParallelGeometry(low.grid, low.detector, [0.0]), low.detector) is None


def test_segments_refuse_detectors_of_another_pitch_when_made():
    grid = tj.Grid(8, 8, 8, 1.0, 1.0, 1.0)
    one = tj.ConeGeometry(grid, tj.Detector(16, 16, 1.0, 1.0), [0.0], tj.ConeBeam(30.0, 50.0))
    other = tj.ConeGeometry(grid, tj.Detector(16, 16, 0.5, 1.0), [0.0], tj.ConeBeam(30.0, 50.0))
    with pytest.raises(ValueError, match="same pixel pitch"):
        tj.geometry.ConeSegments((one, other))


def test_fdk_of_repeated_turns_matches_one_turn():
    grid = tj.Grid(16, 16, 16, 1.0, 1.0, 1.0)
    detector = tj.Detector(24, 24, 1.0, 1.0)
    angles = np.linspace(0, 360, 30, endpoint=False)
    beam = tj.ConeBeam(48.0, 72.0)
    once = tj.ConeGeometry(grid, detector, angles, beam)
    twice = tj.ConeGeometry(grid, detector, np.concatenate([angles, angles + 360]), beam)
    data = np.asarray(tj.project(once, _phantom(16)))

    single = np.asarray(tj.reconstruct(tj.Scan(data, once), "fbp").volume)
    double = np.asarray(tj.reconstruct(tj.Scan(np.concatenate([data, data]), twice), "fbp").volume)

    np.testing.assert_allclose(double, single, atol=1e-4 * np.abs(single).max())


@pytest.mark.gpu
def test_align_recovers_the_misregistration_of_a_multi_orbit_scan():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    n, views = 48, 60
    # Three orbits whose detector moves twice as far as the source between them.
    heights = [(-8.0, 16.0), (0.0, 0.0), (8.0, -16.0)]
    truth = np.concatenate([_orbit_vectors(views, s, d) for s, d in heights])
    error = np.repeat([0.0, 1.5, 3.0], views)[:, None]  # voxels the record is off by
    recorded = truth.copy()
    recorded[:, 2] -= error[:, 0]
    recorded[:, 5] -= error[:, 0]
    empty = np.zeros((48, 3 * views, 64), np.float32)
    true_scan = _scan(truth, empty, n)
    assert isinstance(true_scan.geometry, tj.geometry.ConeSegments)
    data = np.asarray(tj.project(true_scan.geometry, _phantom(n)))
    scan = tj.Scan(data, _scan(recorded, empty, n).geometry)

    result = tj.align(scan, levels=(2, 1))

    shift = result.scan.poses[:, 4] - np.asarray(true_scan.poses)[:, 4]  # pyright: ignore[reportOptionalSubscript]
    per_orbit = shift.reshape(3, views).mean(axis=1)
    # Only the orbits' relative heights are observable: a common shift moves the volume.
    np.testing.assert_allclose(per_orbit - per_orbit[0], 0.0, atol=0.3)
    misregistered = np.asarray(scan.poses)[:, 4] - np.asarray(true_scan.poses)[:, 4]  # pyright: ignore[reportOptionalSubscript]
    assert np.ptp(misregistered.reshape(3, views).mean(axis=1)) > 2.9


def _fine_level_does_not_fit(monkeypatch, n):
    """Make the joint update at grid size ``n`` report that it needs 16 GiB; count checks."""
    # check-public-imports: allow-private
    import tomojax.alignment._pose._coupled_objective as coupled

    checked = coupled._check_device_memory
    sizes = []

    def check(arrays, spec):
        sizes.append(spec.grid.nx)
        if spec.grid.nx == n:
            raise coupled.AlignmentMemoryError(2**34, 2**30)
        return checked(arrays, spec)

    monkeypatch.setattr(coupled, "_check_device_memory", check)
    return sizes


def test_a_level_that_does_not_fit_in_device_memory_ends_alignment_at_the_one_before(
    monkeypatch,
):
    n = 16
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, tj.Detector(n, n, 1.0, 1.0), np.linspace(0, 180, 20))
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom(n))), geometry)
    _fine_level_does_not_fit(monkeypatch, n)

    with pytest.warns(UserWarning, match=r"stops at factor 2, skipping factors \[1\]") as caught:
        result = tj.align(scan, levels=(2, 1))

    (skip,) = [w for w in caught if "skipping factors" in str(w.message)]
    assert skip.filename == __file__  # the caller's line, not the library's
    assert result.info["factors"] == [2]
    assert result.info["factors_skipped"] == [1]
    assert np.asarray(result.volume).shape == (n, n, n)
    with pytest.raises(MemoryError, match="needs 16.0 GiB"):
        tj.align(scan, levels=(1,))


def test_a_run_ended_by_a_level_that_does_not_fit_is_complete(monkeypatch):
    from tomojax.alignment import align_multires, alignment_plan

    n = 16
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = tj.Detector(n, n, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 20))
    data = tj.project(geometry, _phantom(n))
    config = alignment_plan("pose", grid, levels=(2, 1)).config
    sizes = _fine_level_does_not_fit(monkeypatch, n)
    checkpoints = []

    with pytest.warns(UserWarning, match="skipping factors"):
        volume, params, _ = align_multires(
            geometry,
            grid,
            detector,
            data,
            factors=(2, 1),
            config=config,
            checkpoint_callback=checkpoints.append,
        )

    assert checkpoints[-1].run_complete
    assert sizes == [n // 2, n]
    resumed_volume, resumed_params, info = align_multires(
        geometry, grid, detector, data, factors=(2, 1), config=config, resume_state=checkpoints[-1]
    )
    assert sizes == [n // 2, n]  # the skipped level is not retried
    assert info["factors"] == [2]
    assert info["factors_skipped"] == [1]
    np.testing.assert_allclose(resumed_volume, volume, atol=1e-6)
    np.testing.assert_allclose(resumed_params, params, atol=1e-6)
