"""Interactive products compared with independently invoked full public workflows."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RadialDistortion, RejectViews, Stripes, Zingers
from tomojax.geometry import ConeBeam, ConeGeometry
from tomojax.interactive import centre_sweep, correction_preview, reconstruct_slice
from tomojax.recon import CGLSConfig, FBPConfig


@pytest.mark.parametrize("geometry", ["axial", "posed", "laminography", "cone"])
@pytest.mark.parametrize(("axis", "index"), [("z", 0), ("z", 4), ("x", 3)])
def test_plane_is_the_requested_full_result(geometry, axis, index):
    grid = tj.Grid(9, 6, 5, 0.8, 1.2, 1, vol_origin=(-2.6, -3.5, -2))
    detector = tj.Detector(13, 5, 1, 1, (0.35, 0))
    angles = np.linspace(0, 360 if geometry == "cone" else 180, 11, endpoint=False)
    if geometry == "cone":
        geom = ConeGeometry(grid, detector, angles, ConeBeam(15, 28))
    elif geometry == "laminography":
        geom = tj.LaminographyGeometry(grid, detector, angles, tilt=-23)
    else:
        geom = tj.ParallelGeometry(grid, detector, angles)
    rows = np.random.default_rng(835).normal(size=(11, 5, 13)).astype(np.float32)
    scan = tj.Scan(rows, geom)
    if geometry == "posed":
        poses = np.zeros((11, 6))
        poses[:, 4] = np.linspace(-0.8, 0.6, 11)
        scan = scan.with_poses(poses)
    reference = tj.reconstruct(scan, "fbp").volume
    result = reconstruct_slice(scan, axis, index, method="fbp")
    np.testing.assert_array_equal(
        result, np.take(np.asarray(reference), index, axis="zyx".index(axis))
    )


def test_iterative_slice_retains_global_stopping_and_rejects_unsupported_geometry():
    grid = tj.Grid(6, 4, 3, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(8, 3, 1, 1), np.arange(7) * 23.0)
    data = np.random.default_rng(983).uniform(0.1, 2, (7, 3, 8)).astype(np.float32)
    scan = tj.Scan(data, geom)
    config = CGLSConfig(backend="jax", iterations=9, tolerance=0.1)
    full = tj.reconstruct(scan, "cgls", config=config)
    import jax

    plane = reconstruct_slice(scan, "z", 1, method="cgls", config=config)
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(plane, full.volume[1])
    else:
        # Repeated full reconstructions on archived main differ by 1.2e-7 on this fixture.
        np.testing.assert_allclose(plane, full.volume[1], rtol=0, atol=4e-7)
    invalid = tj.Scan(data, ConeGeometry(grid, geom.detector, geom.angles, ConeBeam(15, 28)))
    with pytest.raises(ValueError) as reference:
        tj.reconstruct(invalid, "fourier")
    with pytest.raises(type(reference.value)) as actual:
        reconstruct_slice(invalid, "z", 1, method="fourier")
    assert str(actual.value) == str(reference.value)


def test_sweep_fixes_input_region_for_every_signed_offset_and_duplicate():
    grid = tj.Grid(25, 25, 5, 1, 1, 1)
    det = tj.Detector(25, 5, 1, 1)
    geom = tj.ParallelGeometry(grid, det, np.arange(0, 180, 15))
    data = np.random.default_rng(42).uniform(0.2, 1, (12, 5, 25)).astype(np.float32)
    scan = tj.Scan(data, geom)
    fixed = tj.reconstruct(scan, "fbp", region="bbox").grid
    expected = []
    for c in [0, 3, 0]:
        shifted = tj.Scan(data, replace(geom, detector=replace(det, center=(c * det.du, 0))))
        if c:
            assert tj.reconstruct(shifted, "fbp", region="bbox").grid.shape == (5, 22, 21)
        expected.append(tj.reconstruct(shifted, "fbp", grid=fixed).volume[2])
    result = centre_sweep(scan, "z", 2, [0, 3, 0], method="fbp", region="bbox")
    assert result.shape == (3, 25, 25) and not np.array_equal(expected[0], expected[1])
    np.testing.assert_array_equal(result, np.stack(expected))


@pytest.mark.parametrize("rows", [(0, 1), (2, 4), (5, 6)])
@pytest.mark.parametrize("reject", [True, False])
def test_correction_band_preserves_complete_blocks_halos_and_view_ids(rows, reject):
    n, nv, nu = 13, 13, 17
    grid = tj.Grid(9, 8, 5, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(nu, nv, 0.8, 1.1, (0.2, 0.3)), np.arange(n) * 13.0)
    rng = np.random.default_rng(932)
    counts = rng.uniform(43, 55, (n, nv, nu)).astype("f")
    counts[4, 0, 4] = 123
    counts[8, 2:, :] *= 0.25
    darks = np.full((2, nv, nu), 7.0, np.float32)
    flats = np.full((2, nv, nu), 139.0, np.float32)
    flats[:, :, 4] = 7
    frames = tj.Frames(counts, geom, flats=flats, darks=darks, view_positions=np.arange(n) + 0.5)
    steps = [
        Zingers(size=3, threshold=1.3),
        RadialDistortion(center=(6.25, 7.75), coefficients=(0.99, 0.008)),
        Stripes(3),
        BeamHardening(),
    ]
    if reject:
        steps.insert(3, RejectViews())
    full = frames.corrected(*steps, binning=2)
    band = correction_preview(frames, rows, *steps, binning=2)
    lo, hi = rows
    np.testing.assert_array_equal(band.before, counts[:, lo * 2 : hi * 2])
    np.testing.assert_array_equal(band.after, full.projections[:, lo:hi])
    expected_ids = (
        np.arange(n) if full.view_positions is None else (full.view_positions - 0.5).astype(int)
    )
    np.testing.assert_array_equal(band.views, expected_ids)
    assert (8 not in band.views) == reject
    assert band.detector == full.detector


@pytest.mark.gpu
def test_fdk_preview_inherits_full_filtered_batch_peak():
    grid = tj.Grid(13, 11, 7, 1, 1, 1)
    det = tj.Detector(23, 29, 1, 1)
    geom = ConeGeometry(grid, det, np.arange(0, 360, 15), ConeBeam(25, 42))
    data = np.random.default_rng(321).random((24, 29, 23), dtype=np.float32)
    data[:, 0, 6::2] = 1e3
    cfg = FBPConfig(backend="cuda")

    def full(rows):
        return np.asarray(tj.reconstruct(tj.Scan(rows, geom), "fbp", config=cfg).volume)[3]

    reference = full(data)
    quiet = data.copy()
    quiet[:, 0, :] = 0
    assert np.any(reference != full(quiet))
    np.testing.assert_array_equal(
        reconstruct_slice(tj.Scan(data, geom), "z", 3, method="fbp", config=cfg), reference
    )


def test_region_overlay_registers_estimated_voxel_centres_and_edges():
    from tomojax.interactive import region_preview

    grid = tj.Grid(11, 9, 5, 1, 1, 1)
    det = tj.Detector(11, 9, 1, 1)
    geom = tj.LaminographyGeometry(grid, det, np.arange(0, 360, 15), tilt=36)
    scan = tj.Scan(np.random.default_rng(256).uniform(0, 1, (24, 9, 11)).astype("f"), geom)
    overlay = region_preview(scan)
    expected = tj.estimate_region(scan)
    assert overlay.estimate.grid == expected.grid
    grid = expected.grid
    full = tj.reconstruct(expected.scan, "fbp").volume
    for dim, axis in enumerate("zyx"):
        plane = overlay.images[axis]
        n = full.shape[dim]
        spacing = (grid.vz, grid.vy, grid.vx)[dim]
        center = grid.vol_center[2 - dim]
        assert overlay.bounds[2 - dim] == pytest.approx(
            (center - n * spacing / 2, center + n * spacing / 2)
        )
        (fixed,) = plane.fixed
        assert (fixed.name, fixed.index) == (axis, n // 2)
        assert fixed.coordinate == pytest.approx(center + (n // 2 - (n - 1) / 2) * spacing)
        np.testing.assert_array_equal(plane.data, np.take(np.asarray(full), n // 2, axis=dim))
    with pytest.raises(ValueError) as reference:
        tj.estimate_region(replace(scan, geometry=tj.ParallelGeometry(scan.grid, det, geom.angles)))
    with pytest.raises(type(reference.value)) as observed:
        region_preview(replace(scan, geometry=tj.ParallelGeometry(scan.grid, det, geom.angles)))
    assert str(observed.value) == str(reference.value)


def test_sweep_preserves_calibration_and_view_poses():
    grid = tj.Grid(9, 7, 5, 1, 0.8, 1.1)
    det = tj.Detector(13, 9, 0.7, 1.1, (-0.35, 0.8))
    geom = tj.ParallelGeometry(grid, det, np.arange(0, 180, 15))
    rows = np.random.default_rng(87).uniform(0.1, 3, (12, 9, 13)).astype("f")
    poses = np.zeros((12, 6))
    poses[:, 4] = np.linspace(-0.7, 0.8, 12)
    scan = tj.Scan(rows, geom).with_poses(poses)
    centres = [-2.5, 1.5]
    actual = centre_sweep(scan, "x", 2, centres, method="fbp")
    for i, c in enumerate(centres):
        shifted = replace(geom, detector=replace(det, center=(c * det.du, 0.8)))
        ref = tj.reconstruct(tj.Scan(rows, shifted).with_poses(poses), "fbp", grid=grid)
        np.testing.assert_array_equal(actual[i], ref.volume[:, :, 2])


@pytest.mark.parametrize(
    ("step", "expected"), [(Zingers(size=5, threshold=1.3), (4, 10)), (Stripes(3), (6, 8))]
)
def test_raw_preview_reads_only_required_native_rows(step, expected):
    data = np.random.default_rng(512).uniform(2, 6, (9, 15, 13)).astype("f")
    windows = []

    class Source:
        shape = data.shape
        dtype = data.dtype

        def window(self, rows, cols):
            windows.append((rows.start, rows.stop))
            return data[:, rows, cols]

        def __getitem__(self, key):
            pytest.fail(f"unbounded access {key}")

    grid = tj.Grid(7, 7, 15, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(13, 15, 1, 1), np.arange(9) * 20.0)
    frames = tj.Frames(Source(), geom, white_level=7)
    band = correction_preview(frames, (3, 4), step, binning=2)
    full = replace(frames, counts=data).corrected(step, binning=2)
    np.testing.assert_array_equal(band.after, full.projections[:, 3:4])
    assert windows == [expected, (6, 8)]


def test_nonlocal_transmission_uses_distant_detector_rows():
    from tomojax.corrections import Paganin

    grid = tj.Grid(11, 11, 13, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(17, 13, 1, 1), np.arange(5) * 37.0)
    counts = np.full((5, 13, 17), 80, np.float32)
    counts[:, :2, :] = 25
    frames = tj.Frames(counts, geom, white_level=100)
    step = Paganin(delta_beta=100, distance=0.1, energy_kev=20, pixel_size=1e-6)
    band = correction_preview(frames, (3, 4), step, binning=2)
    full = frames.corrected(step, binning=2)
    np.testing.assert_array_equal(band.after, full.projections[:, 3:4])
    assert np.max(abs(band.after + np.log(0.8))) > 0.01


def test_nonbinary_pitch_preserves_fractional_row_in_the_full_jax_kernel():
    pitch = 0.33
    grid = tj.Grid(9, 7, 8, 1, 1, pitch)
    det = tj.Detector(15, 8, 1, pitch)
    geom = tj.ParallelGeometry(grid, det, np.arange(0, 180, 20))
    scan = tj.Scan(np.random.default_rng(5).uniform(size=(9, 8, 15)).astype("f"), geom)
    cfg = FBPConfig(backend="jax")
    np.testing.assert_array_equal(
        reconstruct_slice(scan, "z", 6, method="fbp", config=cfg),
        tj.reconstruct(scan, "fbp", config=cfg).volume[6],
    )


def test_sweep_keeps_input_cylinder_support():
    grid = tj.Grid(17, 17, 3, 1, 1, 1)
    det = tj.Detector(17, 3, 1, 1)
    geom = tj.ParallelGeometry(grid, det, np.arange(0, 180, 15))
    scan = tj.Scan(np.random.default_rng(564).uniform(0.1, 1, (12, 3, 17)).astype("f"), geom)
    cyl = tj.reconstruct(scan, "fbp", region="cyl")
    support = np.asarray(cyl.volume)[1] != 0
    assert support.any() and not support.all()
    result = centre_sweep(scan, "z", 1, [0, 3], method="fbp", region="cyl")
    shifted = replace(scan, geometry=replace(geom, detector=replace(det, center=(3, 0))))
    full = tj.reconstruct(shifted, "fbp", grid=cyl.grid)
    np.testing.assert_array_equal(result[1], np.where(support, full.volume[1], 0))


def test_plane_does_not_retain_the_full_mapped_reconstruction(monkeypatch, tmp_path):
    import gc
    import weakref

    from tomojax import interactive

    grid = tj.Grid(9, 7, 5, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(11, 5, 1, 1), np.arange(0, 180, 15))
    scan = tj.Scan(np.random.default_rng(8).uniform(0, 1, (12, 5, 11)).astype("f"), geom)
    full = tj.reconstruct(scan, "fourier").volume
    reconstruct = interactive.reconstruct_resolved
    refs = []

    def mapped(plan, **options):
        out = np.lib.format.open_memmap(
            tmp_path / "volume.npy", mode="w+", dtype="f", shape=grid.shape
        )
        result = reconstruct(plan, out=out, **options)
        refs.append(weakref.ref(result.volume))
        return result

    monkeypatch.setattr(interactive, "reconstruct_resolved", mapped)
    plane = reconstruct_slice(scan, "x", 3, method="fourier")
    gc.collect()
    np.testing.assert_array_equal(plane, full[:, :, 3])
    assert refs[0]() is None


def test_dark_preview_band_uses_original_detector_illumination():
    grid = tj.Grid(5, 5, 7, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(9, 7, 1, 1), np.arange(5) * 31.0)
    counts = np.arange(315, dtype=np.float32).reshape(5, 7, 9) + 30
    flats = np.full((1, 7, 9), 500, np.float32)
    flats[:, :2] = 10
    frames = tj.Frames(counts, geom, flats=flats, darks=np.full_like(flats, 10))
    expected = frames.corrected(binning=2).projections[:, :1]
    result = correction_preview(frames, (0, 1), binning=2)
    np.testing.assert_array_equal(result.after, expected)
    with pytest.raises(ValueError, match="No detector pixels have resolved illumination"):
        correction_preview(replace(frames, flats=np.full_like(flats, 10)), (0, 1), binning=2)
