"""Scans whose object extends outside the field of view, through the public API.

A uniform cylinder about the rotation axis, wider than the detector, has exact
line integrals 2 mu sqrt(R^2 - s^2) at distance s from the axis, and the analytic
methods continue rows as such a cylinder: its attenuation inside the field of view
is the analytic answer for each of them.
"""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FBPConfig, FourierConfig

MU, RADIUS = 0.03, 1.6  # attenuation per length; cylinder radius over the field-of-view radius
NU, VIEWS = 64, 96


def _columns(nu: int = NU) -> np.ndarray:
    """Detector coordinates spanning the field of view [-1, 1]."""
    return (np.arange(nu) - (nu - 1) / 2) * 2.0 / nu


def _radius() -> np.ndarray:
    x = _columns()
    return np.hypot(x[None, :], x[:, None])


def _parallel(radius: float = RADIUS, *, rows: int = 2) -> tj.Scan:
    pitch = 2.0 / NU
    grid = tj.Grid(NU, NU, rows, pitch, pitch, pitch)
    detector = tj.Detector(NU, rows, pitch, pitch)
    angles = np.linspace(0.0, 180.0, VIEWS, endpoint=False)
    profile = 2 * MU * np.sqrt(np.maximum(radius**2 - _columns() ** 2, 0))
    data = np.broadcast_to(profile, (VIEWS, rows, NU)).astype(np.float32)
    return tj.Scan(data, tj.ParallelGeometry(grid, detector, angles))


def _cone(radius: float = RADIUS) -> tj.Scan:
    """One detector row in the source's plane: each ray's distance from the axis is exact."""
    source_to_axis, magnification = 12.0, 1.5
    pitch = 2.0 / NU
    grid = tj.Grid(NU, NU, 1, pitch, pitch, pitch)
    detector = tj.Detector(NU, 1, pitch * magnification, pitch * magnification)
    angles = np.linspace(0.0, 360.0, 2 * VIEWS, endpoint=False)
    beam = tj.ConeBeam(source_to_axis, source_to_axis * magnification)
    fan = np.arctan(_columns() * magnification / beam.source_to_detector)
    distance = source_to_axis * np.sin(fan)
    profile = 2 * MU * np.sqrt(np.maximum(radius**2 - distance**2, 0))
    data = np.broadcast_to(profile, (len(angles), 1, NU)).astype(np.float32)
    return tj.Scan(data, tj.ConeGeometry(grid, detector, angles, beam))


def _level_and_bowl(volume: np.ndarray) -> tuple[float, float]:
    """Mean over the inner half of the field of view, and the outer ring's excess over the center."""
    r = _radius()
    plane = np.asarray(volume)[np.asarray(volume).shape[0] // 2]
    return float(plane[r < 0.5].mean()), float(
        plane[(r > 0.8) & (r < 0.92)].mean() - plane[r < 0.3].mean()
    )


@pytest.mark.parametrize(
    ("method", "config"),
    [
        ("fbp", FBPConfig(exterior="cylinder")),
        ("fourier", FourierConfig(backend="numpy", exterior="cylinder")),
        ("fourier", FourierConfig(backend="numpy", filter="ramp", exterior="cylinder")),
    ],
)
def test_parallel_methods_recover_a_cylinder_wider_than_the_detector(method, config) -> None:
    # Zero beyond the detector would raise the level by a quarter here and bend it by more.
    level, bowl = _level_and_bowl(tj.reconstruct(_parallel(), method, config=config).volume)
    assert level == pytest.approx(MU, rel=0.03)
    assert abs(bowl) < 0.03 * MU


@pytest.mark.parametrize("scan_factory", [_parallel, _cone])
@pytest.mark.parametrize("method", ["cgls", "sirt", "fista", "spdhg", "mlem"])
def test_iterative_fit_recovers_the_attenuation_and_flat_interior(scan_factory, method) -> None:
    scan = scan_factory()
    original = np.asarray(scan.projections).copy()
    rec = tj.reconstruct(scan, method)
    level, bowl = _level_and_bowl(np.asarray(rec.volume))
    assert level == pytest.approx(MU, rel=0.04)
    assert abs(bowl) < 0.04 * MU
    expected = "additive_poisson" if method == "mlem" else "subtracted"
    assert rec.info["truncation"]["background_treatment"] == expected
    np.testing.assert_array_equal(scan.projections, original)


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_view_sharing_uses_the_same_exterior_on_each_device(method) -> None:
    import jax

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two devices")
    scan = _parallel(rows=1)
    # Accumulate the same 48-view blocks sequentially on one CPU and in parallel
    # on two. Regrouping a float32 transpose perturbs the fifth CGLS direction.
    from tomojax.recon import CGLSConfig, FISTAConfig

    config = (
        CGLSConfig(iterations=5, views_per_batch=VIEWS // 2)
        if method == "cgls"
        else FISTAConfig(iterations=5, tv_weight=0.0)
    )
    single = tj.reconstruct(scan, method, devices=devices[:1], config=config)
    shared = tj.reconstruct(scan, method, devices=devices[:2], config=config)
    assert shared.info["truncation"] == single.info["truncation"]
    assert shared.info["truncation"]["outside_grid"] > 0
    np.testing.assert_allclose(shared.volume, single.volume, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["cgls", "sirt", "mlem"])
def test_truncated_cone_uses_the_same_model_in_host_output_and_resident_fits(method) -> None:
    scan = _cone()
    options = {"warm_start": True, "iterations": 2}
    resident = tj.reconstruct(scan, method, **options)
    target = np.empty(scan.grid.shape, np.float32)
    streamed = tj.reconstruct(scan, method, out=target, **options)
    assert streamed.volume is target
    assert streamed.info["truncation"] == resident.info["truncation"]
    assert resident.info["truncation"]["outside_grid"] > 0
    np.testing.assert_allclose(target, resident.volume, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("method", ["sirt", "mlem"])
def test_raw_slab_uses_the_same_exterior_as_the_whole_scan(method) -> None:
    scan = _parallel(rows=9)
    frames = tj.Frames(1000 * np.exp(-scan.projections), scan.geometry, white_level=1000)
    from tomojax.recon import MLEMConfig, SIRTConfig

    config = SIRTConfig(iterations=3) if method == "sirt" else MLEMConfig(iterations=3)
    options = {"warm_start": True, "config": config}
    resident = tj.reconstruct(frames.corrected(), method, **options)
    whole = {}
    with tj.reconstruct_slabs(frames, method, info=whole, **options) as slabs:
        sections = list(slabs)
    assert whole["truncation"] == pytest.approx(resident.info["truncation"])
    assert len(sections) > 1
    for section, result in sections:
        assert result.info["truncation"]["outside_grid"] > 0
        assert result.info["truncation"] == pytest.approx(resident.info["truncation"])
        np.testing.assert_allclose(result.volume, resident.volume[section], atol=2e-6, rtol=3e-5)


def test_checkpoint_preserves_the_exterior_during_the_final_fit(tmp_path) -> None:
    from tomojax import progress
    from tomojax.errors import Canceled

    scan = _parallel()
    path = tmp_path / "exterior"
    plain = tj.reconstruct(scan, "sirt", iterations=12)
    control = progress.Control()
    tasks = set()

    def cancel(event):
        if isinstance(event, progress.Started) and event.label == "sirt":
            tasks.add(event.task)
        if isinstance(event, progress.Advanced) and event.task in tasks and event.done >= 2:
            control.cancel()

    with progress.watch(cancel, control=control), pytest.raises(Canceled):
        tj.reconstruct(scan, "sirt", iterations=12, checkpoint=path)
    saved = tj.inspect(path)["checkpoint"]["entries"]["projection-model-0"]
    assert saved["background_treatment"] == "subtracted"
    assert saved["outside_grid"] == plain.info["truncation"]["outside_grid"]
    resumed = tj.reconstruct(scan, "sirt", iterations=12, checkpoint=path)
    np.testing.assert_array_equal(resumed.volume, plain.volume)
    assert resumed.info["truncation"] == plain.info["truncation"]


def test_fdk_recovers_a_cylinder_wider_than_the_detector() -> None:
    rec = tj.reconstruct(_cone(), "fbp", config=FBPConfig(exterior="cylinder"))
    level, bowl = _level_and_bowl(rec.volume)
    assert level == pytest.approx(MU, rel=0.04)
    assert abs(bowl) < 0.04 * MU


@pytest.mark.parametrize("method", ["fbp", "fourier"])
def test_a_contained_object_reconstructs_as_on_a_wider_detector_of_zeros(method) -> None:
    # Rows that end at zero continue as zero: more zero columns change nothing.
    scan = _parallel(0.7, rows=1)
    wide = NU + 32
    pitch = scan.detector.du
    padded = np.pad(np.asarray(scan.projections), ((0, 0), (0, 0), (16, 16)))
    geometry = tj.ParallelGeometry(scan.grid, tj.Detector(wide, 1, pitch, pitch), scan.angles)
    config = FourierConfig(backend="numpy") if method == "fourier" else None

    narrow = np.asarray(tj.reconstruct(scan, method, config=config).volume)
    wider = np.asarray(tj.reconstruct(tj.Scan(padded, geometry), method, config=config).volume)

    assert np.linalg.norm(narrow - wider) < 2e-3 * np.linalg.norm(wider)
    level, _ = _level_and_bowl(narrow)
    assert level == pytest.approx(MU, rel=0.01)


def test_the_record_says_how_much_lies_beyond_the_detector() -> None:
    record = tj.reconstruct(_parallel(), "fbp").info["truncation"]
    edge = (NU - 1) / NU  # distance of the outermost column's center from the axis
    s = np.linspace(-RADIUS, RADIUS, 200_001)
    profile = 2 * MU * np.sqrt(np.maximum(RADIUS**2 - s**2, 0))
    inside = np.abs(s) <= 1.0
    step = s[1] - s[0]
    # Independent of the library: numerical integrals of the cylinder's exact projection.
    beyond = profile[~inside].sum() / profile.sum()
    on_axis = -(profile[~inside] / s[~inside] ** 2).sum() * step / (2 * np.pi)
    at_edge = 2 * MU * np.sqrt(RADIUS**2 - edge**2)

    assert record["radius"] == pytest.approx(RADIUS, rel=1e-3)
    assert record["field_of_view_radius"] == pytest.approx(edge)
    assert record["outside_fraction"] == pytest.approx(beyond, rel=0.02)
    assert record["interior_shift"] == pytest.approx(on_axis, rel=0.03)
    assert record["edge_ratio"] == pytest.approx(at_edge / profile[inside].mean(), rel=0.02)


def test_the_record_of_a_contained_object_is_zero() -> None:
    record = tj.reconstruct(_parallel(0.7), "fbp").info["truncation"]
    assert record["edge_ratio"] == 0 and record["outside_fraction"] == 0
    assert record["interior_shift"] == 0


def test_every_method_carries_the_record() -> None:
    # The original measurements determine all five analytic fields before the solve.
    scan = _parallel()
    expected = tj.reconstruct(scan, "fbp").info["truncation"]
    record = tj.reconstruct(scan, "cgls", iterations=2).info["truncation"]
    assert {key: record[key] for key in expected} == expected
    assert record["data_model"] == "continued_exterior" and record["outside_grid"] > 0
    assert tj.reconstruct(_cone(), "fbp").info["truncation"]["radius"] == pytest.approx(
        RADIUS, rel=1e-3
    )


@pytest.mark.parametrize("method", ["fbp", "cgls", "mlem"])
@pytest.mark.parametrize(("radius", "beyond"), [(RADIUS, True), (0.7, False)])
def test_the_command_line_and_the_mcp_summary_say_what_lies_beyond_the_detector(
    method, radius, beyond, tmp_path, capsys
) -> None:
    from tomojax.cli.main import main
    from tomojax.cli.mcp.tools import summary_text

    tj.save(tmp_path / "scan.nxs", _parallel(radius))
    arguments = ["reconstruct", str(tmp_path / "scan.nxs"), "-o", str(tmp_path / "out.nxs")]
    assert main([*arguments, "--method", method]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = tj.load_reconstruction(tmp_path / "out.nxs").info["truncation"]
    summary = summary_text({"kind": "volume", "truncation": record, "wall_seconds": 1.0})

    share = f"{record['outside_fraction']:.0%} of attenuation beyond the detector"
    assert (share in line) is beyond
    assert (share in summary) is beyond
    assert ("beyond the detector" in line + summary) is beyond
    assert ("modeled exterior" in line and "modeled exterior" in summary) == (
        method in {"cgls", "mlem"} and beyond
    )


@pytest.mark.parametrize(
    ("scan", "config"),
    [
        (_parallel, FBPConfig(exterior="zero")),
        (_parallel, FourierConfig(exterior="zero")),
        (_cone, FBPConfig(exterior="zero")),
    ],
)
def test_zero_exterior_keeps_the_truncated_cylinders_bright_rim(scan, config) -> None:
    rec = tj.reconstruct(scan(), config.method, config=config)
    level, bowl = _level_and_bowl(np.asarray(rec.volume))
    assert level > 1.2 * MU
    assert bowl > 0.2 * MU
    assert rec.info["config"].exterior == "zero"


@pytest.mark.parametrize("method", ["cgls", "sirt", "fista", "spdhg", "mlem"])
def test_zero_exterior_fits_measured_rays_from_the_zero_padded_warm_start(method) -> None:
    from tomojax import recon

    scan = _cone()
    config = recon.api.method_config(method, exterior="zero", iterations=2)
    actual = tj.reconstruct(scan, method, config=config, warm_start=True)
    initial = tj.reconstruct(scan, "fbp", config=FBPConfig(exterior="zero")).volume
    if method == "mlem":
        initial = np.maximum(initial, 0)
    expected, _ = getattr(recon, method)(
        scan.geometry, scan.grid, scan.detector, scan.projections, config=config, init_x=initial
    )
    np.testing.assert_array_equal(actual.volume, expected)
    assert actual.info["config"].exterior == "zero"
    record = actual.info["truncation"]
    assert record["data_model"] == "measured"
    assert record["background_treatment"] == "none"
    assert record["outside_grid"] == 0


@pytest.mark.gpu
def test_device_detector_means_preserve_host_rounding_for_every_leaf_width():
    # check-public-imports: allow-private
    from tomojax.recon._exterior_cuda import host_detector_block

    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(28)
    columns = 9
    for length in [*range(columns, 130 * columns, columns), 31]:
        data = rng.standard_normal((3, 4, length), dtype=np.float32)
        data[:, 0] = -0.0
        device = cp.asarray(data)
        for sample in (slice(None), slice(1, 3), slice(None, None, -1)):
            expected = np.ascontiguousarray(data[:, sample])
            width, remainder = divmod(length, columns)
            if not remainder and 2 <= width <= 128:
                expected = expected.reshape(*expected.shape[:-1], columns, width).mean(axis=-1)
            actual = host_detector_block(device[:, sample], columns)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_private_exterior_storage_keeps_unmodeled_rows_zero(monkeypatch, preallocated):
    import jax

    # check-public-imports: allow-private
    from tomojax.recon._exterior_cuda import DeviceRows

    cp = pytest.importorskip("cupy")
    data = jax.device_put(np.ones((3, 4, 5), np.float32))
    rows = DeviceRows.on_selected_device(data)
    assert rows is not None
    if preallocated:

        def outside_pool_full(allocator):
            raise cp.cuda.runtime.CUDARuntimeError(2)  # cudaErrorMemoryAllocation

        monkeypatch.setattr(cp.cuda, "using_allocator", outside_pool_full)
    rows.write(slice(None), slice(1, 2), cp.full((3, 1, 5), 2.0, dtype=np.float32))
    background, share = rows.result()
    expected = np.zeros(data.shape, np.float32)
    expected[:, 1] = 2.0
    np.testing.assert_array_equal(background, expected)
    np.testing.assert_array_equal(data, np.ones_like(expected))
    assert share == 0.5


@pytest.mark.gpu
def test_binned_detector_has_the_same_exterior_on_host_and_device():
    import jax

    columns, views = 640, 48
    pitch = 2.0 / columns
    grid = tj.Grid(columns, columns, 1, pitch, pitch, pitch)
    detector = tj.Detector(columns, 1, pitch, pitch)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    row = 2 * MU * np.sqrt(RADIUS**2 - _columns(columns) ** 2)
    data = np.broadcast_to(row.astype(np.float32), (views, 1, columns))
    host = tj.reconstruct(tj.Scan(data, geometry), "sirt", iterations=1)
    device = tj.reconstruct(tj.Scan(jax.device_put(data), geometry), "sirt", iterations=1)
    assert host.info["truncation"]["outside_grid"] > 0
    assert device.info["truncation"]["outside_grid"] == pytest.approx(
        host.info["truncation"]["outside_grid"], abs=1e-13, rel=0
    )
    np.testing.assert_array_equal(device.volume, host.volume)
