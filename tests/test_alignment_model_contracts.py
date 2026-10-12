from __future__ import annotations

from dataclasses import replace
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._geometry import _vo_gpu as gpu, initializers as init

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_blocks import _theta_span_from_geometry

# check-public-imports: allow-private
from tomojax.alignment._stages._reconstruction_stage import _is_oom_error_message
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage, SetupGeometryState
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import fista_tv

_vo_mask = init._vo_mask
_vo_metrics = init._vo_metrics
projection_pair_det_u_seed = init.projection_pair_det_u_seed
sinogram_det_u_seed = init.sinogram_det_u_seed


@pytest.mark.parametrize("optimizer", ["adam", "validation_lm"])
def test_fixed_volume_stages_reject_unsupported_optimizers(optimizer: str) -> None:
    schedule = AlignmentSchedule(
        name="bad",
        stages=(
            AlignmentStage(
                name="bad",
                active_dofs=("dx",),
                objective_kind="fixed_volume",
                optimizer=optimizer,  # type: ignore[arg-type]
            ),
        ),
    )

    with pytest.raises(ValueError, match="fixed_volume.*'gd', 'gn', or 'lbfgs'"):
        schedule.validate()


def test_axis_unit_lab_matches_projection_axis_unit_with_tilt() -> None:
    setup = SetupGeometryState.from_degrees(tilt_deg=5.0)

    axis = setup.axis_unit_lab()

    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert axis[1] == pytest.approx(np.sin(np.deg2rad(5.0)))
    assert axis[2] == pytest.approx(np.cos(np.deg2rad(5.0)))


def test_theta_span_uses_sampled_geometric_coverage() -> None:
    geometry = ParallelGeometry(
        grid=Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0),
        detector=Detector(nu=4, nv=4, du=1.0, dv=1.0),
        angles=np.linspace(0.0, 270.0, 12, endpoint=False, dtype=np.float32),
    )

    assert _theta_span_from_geometry(geometry) == pytest.approx(247.5)


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED: allocator failed",
        "resource_exhausted: allocator failed",
        "out of memory while allocating buffer",
        "Allocator could not reserve memory",
    ],
)
def test_alignment_reconstruction_oom_detection_is_case_insensitive(message: str) -> None:
    assert _is_oom_error_message(message)


def test_projection_pair_detector_center_seed_uses_tomojax_sign_convention() -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=32, nv=16, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 180.0], dtype=np.float32),
    )
    u = np.arange(detector.nu, dtype=np.float32)
    base_profile = np.exp(-0.5 * ((u - 13.0) / 2.5) ** 2)
    base_profile += 0.4 * np.exp(-0.5 * ((u - 22.0) / 1.8) ** 2)
    first = np.tile(base_profile[None, :], (detector.nv, 1))
    mirrored_second = np.roll(first, shift=6, axis=1)
    second = np.flip(mirrored_second, axis=1)
    projections = np.stack([first, second], axis=0).astype(np.float32)

    seed = projection_pair_det_u_seed(projections, geometry)

    assert seed.status == "ok_pairs=1"
    assert seed.det_u_px == pytest.approx(3.0, abs=0.25)


def test_projection_pair_detector_center_seed_rejects_constant_projections() -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=32, nv=16, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 180.0], dtype=np.float32),
    )
    projections = np.ones((2, detector.nv, detector.nu), dtype=np.float32)

    seed = projection_pair_det_u_seed(projections, geometry)

    assert seed.status == "no_usable_opposite_angle_pairs"
    assert seed.det_u_px == pytest.approx(0.0)


def _analytic_shifted_sinogram(angles, columns):
    """Independent infinite-line integrals of a broad Gaussian mixture."""
    theta = np.deg2rad(angles)
    centre = 4.6
    u = np.arange(columns) - (columns - 1) / 2 + centre
    rng = np.random.default_rng(4)
    centres = rng.uniform(-0.59 * columns, 0.59 * columns, (2, 12))
    radii = rng.uniform(2, 6, 12) * columns / 64
    profile = np.zeros((len(angles), columns))
    for (cx, cy), radius in zip(centres.T, radii, strict=True):
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        profile += (
            np.sqrt(2 * np.pi)
            * radius
            * np.exp(-0.5 * ((u[None] - projected[:, None]) / radius) ** 2)
        )
    return np.repeat(profile[:, None], 3, axis=1).astype(np.float32), centre


@pytest.mark.numerical
@pytest.mark.parametrize("turn", [180, 360])
@pytest.mark.parametrize("labels", ["wrapped", "individual_turns", "shuffled"])
def test_sinogram_centre_seed_ignores_angle_labels_and_order(turn, labels):
    angles = (350.35 + np.linspace(0, turn, 150, endpoint=False)).astype(np.float32)
    data, centre = _analytic_shifted_sinogram(angles, 64)
    expected = sinogram_det_u_seed(data, angles)
    assert expected is not None
    assert abs(expected.det_u_px - centre) <= 0.3
    relabelled = angles % 360
    if labels == "individual_turns":
        relabelled += 360 * np.random.default_rng(7).integers(-2, 3, len(angles))
    order = np.arange(len(angles))
    if labels == "shuffled":
        order = np.random.default_rng(8).permutation(order)
    actual = sinogram_det_u_seed(data[order], relabelled[order])
    assert actual is not None
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("blank", [False, True])
def test_sinogram_seed_still_declines_missing_half_turn_evidence(blank):
    angles = 350 + np.linspace(0, 130 if not blank else 180, 120, endpoint=False)
    data, _ = _analytic_shifted_sinogram(angles, 64)
    if blank:
        data[:] = 0
    assert sinogram_det_u_seed(data, angles) is None
    assert sinogram_det_u_seed(data, angles % 360) is None


@pytest.mark.numerical
@pytest.mark.parametrize("sampling", ["nonuniform", "repeated"])
def test_sinogram_centre_seed_uses_actual_angular_spacing(sampling):
    angles = np.linspace(0, 179, 90)
    if sampling == "nonuniform":
        angles = np.r_[angles, np.random.default_rng(1).uniform(20, 50, 150)]
    else:
        angles = np.r_[angles, np.repeat(angles[10:25], 10)]
    # Keep the acquisition order arbitrary; the Fourier rows must represent
    # equally spaced angles, not equally spaced indices or repeated exposures.
    angles = angles[np.random.default_rng(2).permutation(len(angles))]
    data, centre = _analytic_shifted_sinogram(angles, 128)
    seed = sinogram_det_u_seed(data, angles, rows=1)
    assert seed is not None
    assert abs(seed.det_u_px - centre) <= 0.3


def test_sinogram_centre_seed_materializes_only_the_sampled_rows():
    angles = np.linspace(0, 180, 120, endpoint=False)
    data, _ = _analytic_shifted_sinogram(angles, 64)
    data = np.tile(data, (1, 7, 1))
    expected = sinogram_det_u_seed(data, angles)

    class RowReadable:
        shape = data.shape

        def __init__(self):
            self.rows = []

        def __array__(self, *args, **kwargs):
            raise AssertionError("centre finding must not materialize the complete scan")

        def __getitem__(self, index):
            views, rows, cols = index
            self.rows.extend(rows)
            return data[views, rows, cols]

    source = RowReadable()
    actual = sinogram_det_u_seed(source, angles)
    assert actual == expected
    assert len(source.rows) == 3
    assert len(set(source.rows)) == 3
    assert sinogram_det_u_seed(jnp.asarray(data), angles) == expected


@pytest.mark.parametrize("columns", [31, 32, 33, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_sinogram_real_fourier_metric_matches_full_spectrum(columns, dtype, layout):
    sino = np.random.default_rng(3).normal(size=(37, columns)).astype(dtype)
    if layout == "F":
        sino = np.asfortranarray(sino)
    elif layout == "reverse":
        sino = sino[:, ::-1]
    original = sino.copy()
    sino.flags.writeable = False
    offsets = np.array([-4, -1.75, 0, 0.25, 3.5])
    mask = _vo_mask(2 * len(sino), columns, 0.25 * columns, 3)
    expected = []
    for offset in offsets:
        shift = 2 * offset
        phase = np.exp(-2j * np.pi * np.fft.fftfreq(columns) * shift)
        moved = np.fft.ifft(np.fft.fft(sino[:, ::-1], axis=1) * phase, axis=1).real
        edge = int(np.ceil(abs(shift)))
        if shift > 0:
            moved[:, :edge] = sino[::-1, :edge]
        elif shift < 0:
            moved[:, -edge:] = sino[::-1, -edge:]
        spectrum = np.fft.fftshift(np.fft.fft2(np.vstack([sino, moved])))
        expected.append(np.mean(np.abs(spectrum) * mask))
    np.testing.assert_allclose(_vo_metrics(sino, offsets, 0.5, 3), expected, rtol=2e-6)
    np.testing.assert_array_equal(sino, original)


def test_sinogram_frequency_cache_is_bounded_immutable_and_keyed():
    initializers = importlib.import_module("tomojax.alignment._geometry.initializers")
    cached = initializers._vo_folded_mask
    cached.cache_clear()
    try:
        fine = cached(37, 33, 0.5, 3)
        expected = fine.copy()
        assert not fine.flags.writeable
        assert cached(37, 33, 0.5, 3) is fine
        coarse = cached(18, 16, 0.5, 3)
        assert cached(37, 33, 0.5, 3) is fine
        assert cached(18, 16, 0.5, 3) is coarse
        for key in [(37, 33, 0.4, 3), (37, 33, 0.4, 1), (38, 33, 0.4, 1)]:
            actual = cached(*key)
            # A different geometry or wedge must never reuse the old weights.
            assert actual is not fine
            assert cached.cache_info().currsize == 2
        restored = cached(37, 33, 0.5, 3)
        assert restored is not fine
        np.testing.assert_array_equal(restored, expected)
    finally:
        cached.cache_clear()


def test_sinogram_metrics_reuse_weights_without_caching_measurements():
    initializers = importlib.import_module("tomojax.alignment._geometry.initializers")
    cached = initializers._vo_folded_mask
    cached.cache_clear()
    try:
        offsets = np.array([-3, -0.5, 0, 1.25, 4])
        sinogram = np.random.default_rng(15).normal(size=(37, 33)).astype(np.float32)
        first = _vo_metrics(sinogram, offsets, 0.5, 3)
        second = _vo_metrics(2 * sinogram, offsets, 0.5, 3)
        np.testing.assert_array_equal(second, 2 * first)
        assert cached.cache_info().misses == 1
        assert cached.cache_info().hits == 1
    finally:
        cached.cache_clear()


@pytest.mark.parametrize("columns", [31, 32, 63, 64])
def test_sinogram_integral_mirror_shifts_do_not_interpolate(columns, monkeypatch):
    initializers = importlib.import_module("tomojax.alignment._geometry.initializers")
    sino = np.random.default_rng(5).normal(size=(37, columns)).astype(np.float32)
    sino.flags.writeable = False
    offsets = np.array([-4, -1.5, 0, 2.5, 4])
    expected = []
    mask = initializers._vo_mask(74, columns, 0.25 * columns, 3)
    for offset in offsets:
        shift = int(2 * offset)
        moved = np.roll(sino[:, ::-1], shift, axis=1)
        if shift > 0:
            moved[:, :shift] = sino[::-1, :shift]
        elif shift < 0:
            moved[:, shift:] = sino[::-1, shift:]
        spectrum = np.fft.fftshift(np.fft.fft2(np.vstack([sino, moved])))
        expected.append(np.mean(np.abs(spectrum) * mask))

    def forbidden(*args, **kwargs):
        raise AssertionError("integral shifts need no inverse FFT")

    monkeypatch.setattr(initializers.fft, "irfft", forbidden)
    np.testing.assert_allclose(initializers._vo_metrics(sino, offsets, 0.5, 3), expected, rtol=2e-6)


def test_sinogram_fractional_mirror_shifts_still_use_fourier_interpolation(monkeypatch):
    initializers = importlib.import_module("tomojax.alignment._geometry.initializers")
    original = initializers.fft.irfft
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(initializers.fft, "irfft", counted)
    sino = np.random.default_rng(6).normal(size=(37, 32)).astype(np.float32)
    initializers._vo_metrics(sino, np.array([-4, -1.75, 0, 0.25, 2]), 0.5, 3)
    assert len(calls) == 2


def test_sinogram_fine_metrics_are_reused_only_within_each_sampled_row(monkeypatch):
    initializers = importlib.import_module("tomojax.alignment._geometry.initializers")
    calls = []

    def metric(sino, offsets, ratio, drop):
        calls.append((sino.shape, float(sino[0, 0]), offsets.copy()))
        return offsets**2

    monkeypatch.setattr(initializers, "_vo_metrics", metric)
    angles = np.arange(180, dtype=np.float32)
    data = np.broadcast_to(np.arange(64, dtype=np.float32), (180, 3, 64)).copy()
    data += np.arange(3, dtype=np.float32)[None, :, None]
    for factor in (1, 2):
        seed = initializers.sinogram_det_u_seed(factor * data, angles)
        assert seed is not None and seed.det_u_px == 0
    fine = [(value, offsets) for shape, value, offsets in calls if shape == (180, 64)]
    assert len(fine) == 12  # Two refinements for each of three rows on both calls.
    for i in range(0, len(fine), 2):
        whole, fractional = fine[i : i + 2]
        assert whole[0] == fractional[0]
        assert len(whole[1]) == 13 and len(fractional[1]) == 6
        assert not set(whole[1]) & set(fractional[1])
    assert [value for value, _ in fine[::2]] == [0, 1, 2, 0, 2, 4]


_COR_CONFIG = AlignConfig(
    outer_iterations=1, iterations=1, tv_weight=0, ray_integrator="joseph", projector_backend="jax"
)


@pytest.mark.numerical
@pytest.mark.parametrize("columns", [31, 32, 63, 64])
@pytest.mark.parametrize("seed", [101, 108])
def test_sinogram_coarse_search_preserves_narrow_detector_evidence(columns, seed):
    angles = np.linspace(0, 180, 180, endpoint=False)
    theta = np.deg2rad(angles)
    rng = np.random.default_rng(seed)
    centre = rng.uniform(-0.2 * columns, 0.2 * columns)
    u = np.arange(columns) - (columns - 1) / 2 + centre
    sino = np.zeros((len(angles), columns))
    # Independent line integrals of an off-axis, truncated Gaussian mixture.
    for _ in range(12):
        cx, cy = rng.uniform(-0.6 * columns, 0.6 * columns, 2)
        radius = rng.uniform(0.03 * columns, 0.1 * columns)
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        sino += (
            rng.uniform(0.5, 2)
            * radius
            * np.exp(-0.5 * ((u[None] - projected[:, None]) / radius) ** 2)
        )
    actual = sinogram_det_u_seed(sino[:, None, :].astype(np.float32), angles, rows=1)
    assert actual is not None
    assert abs(actual.det_u_px - centre) < 0.4


@pytest.mark.numerical
@pytest.mark.parametrize("ray_integrator", ["joseph", "exact"])
def test_public_cor_keeps_wrapped_half_turn_evidence(monkeypatch, ray_integrator):
    final_integrators = []

    def record_final_reconstruction(*args, config, **kwargs):
        final_integrators.append(config.ray_integrator)
        return fista_tv(*args, config=config, **kwargs)

    monkeypatch.setattr(
        "tomojax.alignment._stages._setup_stage.fista_tv", record_final_reconstruction
    )
    config = replace(_COR_CONFIG, ray_integrator=ray_integrator)
    angles = 350 + np.linspace(0, 180, 120, endpoint=False)
    data, centre = _analytic_shifted_sinogram(angles, 48)
    grid = Grid(72, 72, 3, 1, 1, 1)
    detector = Detector(48, 3, 1, 1)
    original = tj.Scan(data, ParallelGeometry(grid, detector, angles))
    expected = tj.align(original, mode="cor", levels=(1,), config=config)
    relabelled = angles % 360 + 360 * np.random.default_rng(7).integers(-2, 3, len(angles))
    scan = tj.Scan(data, ParallelGeometry(grid, detector, relabelled))
    actual = tj.align(scan, mode="cor", levels=(1,), config=config)
    assert final_integrators == [ray_integrator, ray_integrator]
    first = actual.info["outer_stats"][0]
    assert first["detector_center_seed_method"] == "sinogram_vo_2014"
    assert abs(first["detector_center_seed_det_u_px"] - centre) <= 0.3
    assert abs(actual.scan.detector.center[0] - centre) <= 0.35
    assert abs(actual.scan.detector.center[0] - expected.scan.detector.center[0]) < 1e-3
    reference = np.asarray(expected.volume)
    assert np.linalg.norm(np.asarray(actual.volume) - reference) / np.linalg.norm(reference) < 1e-4


@pytest.mark.numerical
@pytest.mark.parametrize("factor", [1, 2])
def test_public_cor_seed_is_relative_to_the_existing_detector_centre(factor):
    prior_px = -2.3
    angles = 350 + np.linspace(0, 180, 120, endpoint=False)
    data, centre = _analytic_shifted_sinogram(angles, 48)
    pitch = 0.8
    grid = Grid(72, 72, 3, pitch, pitch, pitch)
    config = _COR_CONFIG
    detector = Detector(48, 3, pitch, pitch)
    reference = tj.align(
        tj.Scan(data, ParallelGeometry(grid, detector, angles)),
        mode="cor",
        levels=(factor,),
        config=config,
    )
    detector = Detector(48, 3, pitch, pitch, (prior_px * pitch, 0))
    actual = tj.align(
        tj.Scan(data, ParallelGeometry(grid, detector, angles)),
        mode="cor",
        levels=(factor,),
        config=config,
    )
    first = actual.info["outer_stats"][0]
    assert first["detector_center_seed_method"] == "sinogram_vo_2014"
    assert abs(first["detector_center_seed_det_u_px"] + prior_px - centre) <= 0.3
    assert abs(actual.scan.detector.center[0] - centre * pitch) <= 0.35
    assert abs(actual.scan.detector.center[0] - reference.scan.detector.center[0]) < 1e-5
    image = np.asarray(reference.volume)
    assert np.linalg.norm(np.asarray(actual.volume) - image) / np.linalg.norm(image) < 1e-4


@pytest.mark.numerical
def test_public_cor_fallback_seed_remains_an_incremental_offset(monkeypatch):
    # A short arc declines the absolute sinogram estimator. The residual
    # search already measures an offset from the supplied detector geometry.
    angles = np.linspace(0, 130, 32)
    data, _ = _analytic_shifted_sinogram(angles, 48)
    monkeypatch.setattr(
        "tomojax.alignment._stages._setup_stage.sinogram_det_u_seed", lambda *args: None
    )
    monkeypatch.setattr(
        "tomojax.alignment._stages._setup_stage.reprojection_det_u_seed",
        lambda *args: SimpleNamespace(det_u_px=1.25, amplitude_px=0.2, status="ok_reprojection"),
    )
    grid, detector = Grid(72, 72, 3, 1, 1, 1), Detector(48, 3, 1, 1, (2.3, 0))
    out = tj.align(
        tj.Scan(data, ParallelGeometry(grid, detector, angles)),
        mode="cor",
        levels=(1,),
        config=_COR_CONFIG,
    )
    first = out.info["outer_stats"][0]
    assert first["detector_center_seed_method"] == "fbp_reprojection_residual_search"
    assert first["detector_center_seed_det_u_px"] == pytest.approx(1.25)


@pytest.mark.numerical
@pytest.mark.parametrize("batch", [0, 7])
@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_public_cor_preserves_reconstruction_settings(monkeypatch, batch, backend):
    batches = []
    iterations = []
    backends = []

    def record_reconstruction(*args, config, **kwargs):
        batches.append(config.views_per_batch)
        iterations.append(config.iterations)
        backends.append(config.projector_backend)
        return fista_tv(*args, config=config, **kwargs)

    monkeypatch.setattr("tomojax.alignment._stages._setup_stage.fista_tv", record_reconstruction)
    monkeypatch.setattr("tomojax.alignment._objectives.fold_recon.fista_tv", record_reconstruction)
    angles = np.linspace(0, 180, 120, endpoint=False)
    data, centre = _analytic_shifted_sinogram(angles, 48)
    geometry = ParallelGeometry(Grid(72, 72, 3, 1, 1, 1), Detector(48, 3, 1, 1), angles)
    result = tj.align(
        tj.Scan(data, geometry),
        mode="cor",
        levels=(1,),
        config=replace(_COR_CONFIG, views_per_batch=batch, projector_backend=backend),
    )
    assert len(batches) == 3  # Two folds and the returned-volume reconstruction.
    assert batches == [batch or None] * 3
    # Explicit JAX must never be upgraded to CUDA. The accelerator preference
    # still falls back on CPU and for unsupported integration/grid models.
    assert backends == ["jax" if backend == "jax" else "auto"] * 3
    assert iterations == [2 * _COR_CONFIG.iterations] * 3
    assert result.info["outer_stats"][0]["quality_tier"] == "reference"
    assert abs(result.scan.detector.center[0] - centre) <= 0.35
    assert np.isfinite(result.volume).all()


@pytest.mark.numerical
@pytest.mark.parametrize("columns", [128, 511])
@pytest.mark.parametrize("views", [901, 1801])
def test_sinogram_coarse_angular_binning_preserves_sharp_feature_centre(columns, views):
    angles = np.linspace(0, 180, views, endpoint=False)
    theta = np.deg2rad(angles)
    rng = np.random.default_rng(968)
    centre = rng.uniform(-0.2 * columns, 0.2 * columns)
    u = np.arange(columns) - (columns - 1) / 2 + centre
    sino = np.zeros((views, columns))
    for _ in range(12):
        cx, cy = rng.uniform(-0.6 * columns, 0.6 * columns, 2)
        radius = rng.uniform(0.001 * columns, 0.02 * columns)
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        sino += (
            rng.uniform(0.5, 2)
            * radius
            * np.exp(-0.5 * ((u[None] - projected[:, None]) / radius) ** 2)
        )
    actual = sinogram_det_u_seed(sino[:, None, :].astype(np.float32), angles, rows=1)
    assert actual is not None
    assert abs(actual.det_u_px - centre) < 0.3


@pytest.mark.parametrize("cols", [31, 32, 64])
@pytest.mark.parametrize("layout", ["c", "f", "reversed"])
def test_vo_gpu_metric_matches_cpu(cols, layout):
    sino = np.random.default_rng(21).normal(size=(37, cols)).astype(np.float32)
    if layout == "f":
        sino = np.asfortranarray(sino)
    elif layout == "reversed":
        sino = sino[::-1, ::-1]
    sino.flags.writeable = False
    saved = sino.copy()
    offsets = np.array([-40, -4, -1.75, 0, 0.25, 3.5, 40], dtype=np.float32)
    padded = np.zeros(13, np.float32)
    padded[: len(offsets)] = offsets
    got = np.asarray(
        gpu._metrics(
            sino,
            padded,
            init._vo_folded_mask(*sino.shape, 0.5, 3),
            np.fft.rfftfreq(cols).astype(np.float32),
            np.int32(len(offsets)),
        )
    )[: len(offsets)]
    np.testing.assert_allclose(got, init._vo_metrics(sino, offsets, 0.5, 3), rtol=2e-6)
    np.testing.assert_array_equal(sino, saved)


def _fake_execution(monkeypatch, *, free=100, workspace=True, error=None):
    calls = []
    device = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(
        gpu,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_disable_jit=False, jax_default_device=device)),
    )

    def memory(*, device):
        calls.append(("memory", device))
        return free

    def run(*args):
        calls.append(("run", args[-1]))
        if error is not None:
            raise error
        return np.arange(13, dtype=np.float32)

    compiled = SimpleNamespace(
        memory_analysis=lambda: SimpleNamespace(
            temp_size_in_bytes=20, argument_size_in_bytes=20, output_size_in_bytes=10
        )
        if workspace
        else None
    )

    class Executable:
        memory_analysis = staticmethod(compiled.memory_analysis)

        def __call__(self, *args):
            return run(*args)

    def lower(*args):
        calls.append(("lower", len(args[1])))
        return SimpleNamespace(compile=lambda: Executable())

    monkeypatch.setattr(gpu, "device_free_memory_bytes", memory)
    monkeypatch.setattr(gpu, "_metrics", SimpleNamespace(lower=lower))
    return calls, device


@pytest.mark.parametrize("free", [None, 0, 99, 100])
def test_vo_gpu_budget_uses_exact_selected_device(monkeypatch, free):
    calls, device = _fake_execution(monkeypatch, free=free)
    result = gpu.try_metrics(np.ones((16, 32), np.float32), np.arange(6), np.ones((32, 17)))
    assert all(value is device for kind, value in calls if kind == "memory")
    assert any(kind == "run" for kind, _ in calls) == (free == 100)
    if free == 100:
        np.testing.assert_array_equal(result, np.arange(6))
    else:
        assert result is None
    if free is None:
        assert not any(kind == "lower" for kind, _ in calls)


def test_vo_gpu_unknown_workspace_keeps_cpu(monkeypatch):
    calls, _ = _fake_execution(monkeypatch, workspace=False)
    assert gpu.try_metrics(np.ones((16, 32)), np.arange(6), np.ones((32, 17))) is None
    assert not any(kind == "run" for kind, _ in calls)


@pytest.mark.parametrize("disabled", [False, True])
def test_vo_gpu_skips_cpu_and_disabled_jit(monkeypatch, disabled):
    calls, _ = _fake_execution(monkeypatch)
    gpu.jax.config.jax_disable_jit = disabled
    gpu.jax.config.jax_default_device = SimpleNamespace(platform="gpu" if disabled else "cpu")
    assert gpu.try_metrics(np.ones((16, 32)), np.arange(6), np.ones((32, 17))) is None
    assert calls == []


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED: allocation failed",
        "INTERNAL: fft_plan != nullptr Failed to create cuFFT batched plan with scratch allocator.",
        "INTERNAL: unrelated kernel failure",
    ],
)
def test_vo_gpu_only_falls_back_for_allocation_failure(monkeypatch, message):
    _fake_execution(monkeypatch, error=RuntimeError(message))
    args = (np.ones((16, 32)), np.arange(6), np.ones((32, 17)))
    if "unrelated" in message:
        with pytest.raises(RuntimeError, match="unrelated"):
            gpu.try_metrics(*args)
    else:
        assert gpu.try_metrics(*args) is None


def test_vo_gpu_fallback_is_per_estimate(monkeypatch):
    calls = []

    def miss(*args):
        calls.append(True)

    monkeypatch.setattr(gpu, "try_metrics", miss)
    monkeypatch.setattr(init, "_vo_metrics", lambda sino, offsets, ratio, drop: offsets**2)
    data = np.broadcast_to(np.arange(640, dtype=np.float32), (1800, 3, 640))
    angles = np.arange(1800) * 0.1
    for expected_calls in (1, 2):
        seed = init.sinogram_det_u_seed(data, angles)
        assert seed.det_u_px == 0
        assert len(calls) == expected_calls
    init.sinogram_det_u_seed(data[:, :1], angles)
    assert len(calls) == 2


def test_vo_gpu_independent_gaussian_axis(monkeypatch):
    n = 128
    angles = np.arange(180, dtype=np.float32)
    t = np.deg2rad(angles)
    axis = -7.25
    u = np.arange(n) - (n - 1) / 2 + axis
    sino = np.zeros((len(angles), n))
    for cx, cy, radius, amplitude in [(10, 4, 6, 1), (-12, -6, 4, 0.7), (0, 18, 2.4, 0.4)]:
        distance = u[None] - np.cos(t)[:, None] * cx + np.sin(t)[:, None] * cy
        sino += amplitude * np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * (distance / radius) ** 2)
    data = sino.astype(np.float32)[:, None]
    cpu = init.sinogram_det_u_seed(data, angles)

    def metrics(sino, offsets, ratio, drop):
        return np.asarray(
            gpu._metrics(
                sino,
                np.asarray(offsets, np.float32),
                init._vo_folded_mask(*sino.shape, ratio, drop),
                np.fft.rfftfreq(sino.shape[1]).astype(np.float32),
                np.int32(len(offsets)),
            )
        )

    monkeypatch.setattr(init, "_vo_metrics", metrics)
    accelerated = init.sinogram_det_u_seed(data, angles)
    assert accelerated == cpu
    assert abs(accelerated.det_u_px - axis) < 0.3


@pytest.mark.gpu
@pytest.mark.numerical
def test_vo_gpu_public_large_scan_preserves_physical_density(monkeypatch):
    # Cross the actual public dispatch threshold, not a forced small FFT helper.
    n = 512
    angles = np.linspace(0, 180, 2048, endpoint=False, dtype=np.float32)
    theta = np.deg2rad(angles.astype(np.float64))
    axis = -32.25
    u = np.arange(n) - (n - 1) / 2 + axis
    x, y = np.meshgrid((np.arange(256) - 127.5) * 2, (np.arange(256) - 127.5) * 2, indexing="ij")
    sino = np.zeros((len(angles), n))
    truth = np.zeros((256, 256))
    for cx, cy, radius, amplitude in [
        (80, 60, 24, 0.005),
        (-110, -60, 18, 0.004),
        (15, -105, 12, 0.003),
    ]:
        distance = u[None] - np.cos(theta)[:, None] * cx + np.sin(theta)[:, None] * cy
        sino += amplitude * np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * (distance / radius) ** 2)
        truth += amplitude * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / radius**2)
    data = np.broadcast_to(sino.astype(np.float32)[:, None], (len(angles), 3, n))
    scan = tj.Scan(data, ParallelGeometry(Grid(256, 256, 3, 2, 2, 1), Detector(n, 3, 1, 1), angles))
    metrics = gpu.try_metrics
    accelerated = []

    def measured(*args):
        result = metrics(*args)
        accelerated.append(result is not None)
        return result

    monkeypatch.setattr(gpu, "try_metrics", measured)
    centred = tj.estimate_center(scan)
    assert accelerated == [True] * 6
    assert abs(centred.detector.center[0] - axis) <= 0.25
    assert centred.projections is scan.projections
    volume = np.asarray(tj.reconstruct(centred).volume)
    expected = np.broadcast_to(truth[:, :, None], volume.shape)
    assert np.linalg.norm(volume - expected) / np.linalg.norm(expected) < 0.006
