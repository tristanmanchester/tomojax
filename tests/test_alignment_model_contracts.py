from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._geometry import initializers as init

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_blocks import _theta_span_from_geometry

# check-public-imports: allow-private
from tomojax.alignment._stages._reconstruction_stage import _is_oom_error_message
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage, SetupGeometryState
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import fista_tv

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
    angles = angles[np.random.default_rng(2).permutation(len(angles))]
    data, centre = _analytic_shifted_sinogram(angles, 128)
    seed = sinogram_det_u_seed(data, angles, rows=1)
    assert seed is not None
    assert abs(seed.det_u_px - centre) <= 0.3


def _random_gaussian_sinogram(views, columns, seed, radius):
    """Line integrals of off-axis Gaussians whose rotation axis is ``centre`` pixels off."""
    angles = np.linspace(0, 180, views, endpoint=False)
    theta = np.deg2rad(angles)
    rng = np.random.default_rng(seed)
    centre = rng.uniform(-0.2 * columns, 0.2 * columns)
    u = np.arange(columns) - (columns - 1) / 2 + centre
    sino = np.zeros((views, columns))
    for _ in range(12):
        cx, cy = rng.uniform(-0.6 * columns, 0.6 * columns, 2)
        r = rng.uniform(*radius) * columns
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        sino += rng.uniform(0.5, 2) * r * np.exp(-0.5 * ((u[None] - projected[:, None]) / r) ** 2)
    return sino[:, None, :].astype(np.float32), angles, centre


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("views", "columns", "radius"), [(180, 31, (0.03, 0.1)), (901, 128, (0.001, 0.02))]
)
def test_sinogram_coarse_search_keeps_narrow_and_sharp_evidence(views, columns, radius):
    data, angles, centre = _random_gaussian_sinogram(views, columns, 101, radius)
    seed = sinogram_det_u_seed(data, angles, rows=1)
    assert seed is not None
    assert abs(seed.det_u_px - centre) < 0.4


def test_sinogram_centre_seed_reads_only_the_sampled_rows():
    angles = np.linspace(0, 180, 120, endpoint=False)
    data, _ = _analytic_shifted_sinogram(angles, 64)
    data = np.tile(data, (1, 7, 1))

    class RowReadable:
        shape = data.shape

        def __array__(self, *args, **kwargs):
            raise AssertionError("centre finding must not copy the complete scan")

        def __getitem__(self, index):
            return data[index]

    assert sinogram_det_u_seed(RowReadable(), angles) == sinogram_det_u_seed(data, angles)
    assert sinogram_det_u_seed(jnp.asarray(data), angles) == sinogram_det_u_seed(data, angles)


@pytest.mark.parametrize("columns", [32, 33])
def test_sinogram_real_spectrum_metric_matches_the_full_spectrum(columns):
    sino = np.random.default_rng(3).normal(size=(37, columns)).astype(np.float32)
    offsets = np.array([-4, -1.75, 0, 0.25, 3.5])
    mask = init._vo_mask(2 * len(sino), columns, 0.25 * columns, 3)
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
    np.testing.assert_allclose(init._vo_metrics(sino, offsets, 0.5, 3), expected, rtol=2e-6)


_COR_CONFIG = AlignConfig(
    outer_iterations=1, iterations=1, tv_weight=0, ray_integrator="joseph", projector_backend="jax"
)


@pytest.mark.numerical
def test_public_cor_keeps_wrapped_half_turn_evidence():
    angles = 350 + np.linspace(0, 180, 120, endpoint=False)
    data, centre = _analytic_shifted_sinogram(angles, 48)
    grid = Grid(72, 72, 3, 1, 1, 1)
    detector = Detector(48, 3, 1, 1)
    original = tj.Scan(data, ParallelGeometry(grid, detector, angles))
    expected = tj.align(original, mode="cor", levels=(1,), config=_COR_CONFIG)
    relabelled = angles % 360 + 360 * np.random.default_rng(7).integers(-2, 3, len(angles))
    scan = tj.Scan(data, ParallelGeometry(grid, detector, relabelled))
    actual = tj.align(scan, mode="cor", levels=(1,), config=_COR_CONFIG)
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
@pytest.mark.parametrize("bounded", [False, True])
def test_public_cor_fallback_seed_remains_an_incremental_offset(monkeypatch, bounded):
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
        config=replace(_COR_CONFIG, bounds=(("det_u_px", -0.5, 0.5),) if bounded else ()),
    )
    first = out.info["outer_stats"][0]
    assert first["detector_center_seed_method"] == "fbp_reprojection_residual_search"
    assert first["detector_center_seed_det_u_px"] == pytest.approx(1.25)

    assert first["setup_seed_clipped_to_bounds"] is bounded
    if bounded:
        assert abs(out.scan.detector.center[0] - detector.center[0]) <= 0.5


@pytest.mark.numerical
def test_public_cor_reconstructs_with_the_configured_integrator_and_batches(monkeypatch):
    seen = []

    def record(*args, config, **kwargs):
        seen.append((config.ray_integrator, config.views_per_batch, config.projector_backend))
        return fista_tv(*args, config=config, **kwargs)

    monkeypatch.setattr("tomojax.alignment._stages._setup_stage.fista_tv", record)
    monkeypatch.setattr("tomojax.alignment._objectives.fold_recon.fista_tv", record)
    angles = np.linspace(0, 180, 120, endpoint=False)
    data, centre = _analytic_shifted_sinogram(angles, 48)
    geometry = ParallelGeometry(Grid(72, 72, 3, 1, 1, 1), Detector(48, 3, 1, 1), angles)
    config = replace(_COR_CONFIG, ray_integrator="exact", views_per_batch=7)
    result = tj.align(tj.Scan(data, geometry), mode="cor", levels=(1,), config=config)
    # Two folds and the returned-volume reconstruction.
    assert seen == [("exact", 7, "jax")] * 3
    assert abs(result.scan.detector.center[0] - centre) <= 0.35


def _analytic_common_lines(true_axis):
    # Independent 3-D Fourier oracle: nonplanar Gaussian mixture, arbitrary
    # per-view translations. Intersections use cross products, not tilt formulas.
    rng = np.random.default_rng(415)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    poses = np.zeros((len(angles), 3, 3))
    poses[:, 0, 0] = poses[:, 1, 1] = np.cos(angles)
    poses[:, 1, 0] = np.sin(angles)
    poses[:, 0, 1] = -np.sin(angles)
    poses[:, 2, 2] = 1

    def orient(axis):
        cross = np.cross([0, 0, 1], axis)
        x, y, z = cross
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return (np.eye(3) + skew + skew @ skew / (1 + axis[2])) @ poses

    truth = orient(np.asarray(true_axis))
    i = np.repeat(np.arange(len(angles)), 3)
    j = (i + np.tile([3, 7, 10], len(angles))) % len(angles)
    centres = rng.uniform(-10, 10, (8, 3))
    variance = rng.uniform(1.5, 5, (8, 3)) ** 2
    shifts = rng.uniform(-5, 5, (len(angles), 2))
    frequency = np.linspace(0.01, 0.25, 40)

    def sample(axis):
        candidate = orient(axis)
        line = np.cross(candidate[i, 1], candidate[j, 1])
        line /= np.linalg.norm(line, axis=1, keepdims=True)
        spectra = []
        for ids in (i, j):
            uv = np.einsum("pki,pi->pk", candidate[ids][:, [0, 2]], line)
            uv = uv[:, None, :] * frequency[None, :, None]
            q = np.einsum("pki,pfk->pfi", truth[ids][:, [0, 2]], uv)
            exponent = -2 * np.pi**2 * np.einsum("pfi,gi->pfg", q**2, variance)
            phase = -2j * np.pi * np.einsum("pfi,gi->pfg", q, centres)
            value = np.exp(exponent + phase).sum(axis=-1)
            value *= np.exp(-2j * np.pi * np.einsum("pfi,pi->pf", uv, shifts[ids]))
            spectra.append(value)
        return tuple(spectra)

    return sample


@pytest.mark.parametrize(
    ("tilt", "azimuth", "roll", "nominal_tilt"),
    [(35.0, 0.4, 1.7, 31.0), (24.0, -2.0, 0.0, 20.0), (-35.0, 0.4, 1.7, 31.0)],
)
def test_common_line_axis_seed_recovers_nonplanar_geometry_despite_view_shifts(
    tilt, azimuth, roll, nominal_tilt
):
    t, a = np.deg2rad([tilt, azimuth + roll])
    axis = np.array([np.cos(t) * np.sin(a), np.sin(t), np.cos(t) * np.cos(a)])
    n = np.deg2rad(nominal_tilt)
    nominal = np.array([0, np.sin(n), np.cos(n)])
    result = init.fit_axis_direction(_analytic_common_lines(axis), nominal, detector_roll_deg=roll)
    np.testing.assert_allclose(result.rotations_deg, [tilt - nominal_tilt, azimuth], atol=0.03)
    assert result.correlation_after > result.correlation_before
    assert result.correlation_after > 0.99999


def test_common_line_axis_seed_preserves_frozen_bounds_and_uninformative_estimates():
    t = np.deg2rad(35)
    sample = _analytic_common_lines(np.array([0, np.sin(t), np.cos(t)]))
    n = np.deg2rad(31)
    nominal = np.array([0, np.sin(n), np.cos(n)])
    result = init.fit_axis_direction(
        sample, nominal, active=(True, False), bounds_deg=((0, 2), (-60, 60))
    )
    np.testing.assert_allclose(result.rotations_deg, [2, 0], atol=1e-3)
    empty = init.fit_axis_direction(
        lambda axis: (np.ones((3, 8), complex), np.ones((3, 8), complex)),
        nominal,
        rotations_deg=(1.25, -0.75),
    )
    assert empty.rotations_deg == (1.25, -0.75)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("roll", [0.0, 3.0])
@pytest.mark.parametrize("tilt", [35.0, -35.0])
def test_full_setup_seeds_jointly_in_native_units_and_preserves_frozen_axes(
    monkeypatch, bounded, roll, tilt
):
    from tomojax.alignment.api import align as pose_align
    from tomojax.core.multires import bin_projections
    from tomojax.geometry.api import detector_grid_from_geometry_inputs
    from tomojax.io import build_geometry_from_dataset_metadata

    true = np.array([0.0, np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))])
    sample = _analytic_common_lines(
        init.axis_in_detector(true, np.zeros(2), detector_roll_deg=roll)
    )
    declined = bool(roll and bounded)
    target = 1.6 if declined else 2.4 if bounded else 3.0
    monkeypatch.setattr(
        init,
        "prepare_common_lines",
        lambda data, angles, detector: SimpleNamespace(
            sample=sample,
            center=lambda axis, **kwargs: None
            if declined
            else (3.0 + detector.center[0] - 1.6) / detector.du,
        ),
    )
    grid, detector = Grid(8, 8, 8, 0.8, 0.8, 0.8), Detector(8, 8, 0.8, 0.8, (1.6, 0.0))
    geometry = build_geometry_from_dataset_metadata(
        {
            "grid": grid.to_dict(),
            "detector": detector.to_dict(),
            "angles": np.linspace(0, 360, 8, endpoint=False),
            "geometry_type": "lamino",
            "tilt_deg": 31.0,
            "detector_roll_deg": roll,
        }
    )[2]
    seeded_at = []
    monkeypatch.setattr(
        "tomojax.alignment._stages._stage_runners.seeded_translation_params",
        lambda geometry, grid, detector, data, cfg: seeded_at.append(
            (geometry.axis_unit_lab, detector.center)
        ),
    )
    cfg = replace(
        _COR_CONFIG,
        bounds={"axis_rot_x_deg": (-2, 2) if tilt > 0 else (-70, -60), "det_u_px": (-1, 1)}
        if bounded
        else (),
        freeze=("axis_rot_y_deg",),
        seed_translations=True,
    )

    def check(geometry, grid, level_detector, projections, **kwargs):
        if roll:
            reference = replace(detector, center=(target, 0.0))
            native = np.stack(
                detector_grid_from_geometry_inputs(reference, {"detector_roll_deg": roll})
            ).reshape(2, 8, 8)
            selected = bin_projections(native, detector.nu // level_detector.nu)
            np.testing.assert_allclose(
                np.stack(kwargs["det_grid_override"]), selected.reshape(2, -1), atol=5e-7
            )
        kwargs["config"] = kwargs.pop("cfg")
        kwargs.pop("anchor_translations")
        return pose_align(geometry, grid, level_detector, projections, **kwargs)

    monkeypatch.setattr("tomojax.alignment._stages._stage_runners.align", check)
    result = tj.align(
        tj.Scan(np.zeros((8, 8, 8), np.float32), geometry), mode="full", levels=(2, 1), config=cfg
    )
    calibration = result.info["geometry_calibration_state"]
    scan = {item["name"]: item["value"] for item in calibration["scan"]}
    detector_fit = {item["name"]: item["value"] for item in calibration["detector"]}
    expected_tilt = 33.0 if bounded and tilt > 0 else tilt
    assert scan["axis_rot_x_deg"] == pytest.approx(expected_tilt - 31.0, abs=0.03)
    assert scan["axis_rot_y_deg"] == 0.0
    assert detector_fit["detector_roll_deg"] == pytest.approx(roll)
    assert result.scan.detector.center[0] == pytest.approx(target, abs=1e-5)
    assert bool(result.info["outer_stats"][0].get("detector_center_seed_applied")) != declined

    if roll:
        assert not seeded_at  # The broad translation initializer has no rolled-ray override.
        return
    assert len(seeded_at) == 1
    assert seeded_at[0][0][1] == pytest.approx(np.sin(np.deg2rad(expected_tilt)), abs=5e-4)
    # Point decimation shifts the coarse detector centre by half a native pixel.
    assert seeded_at[0][1][0] == pytest.approx(2.8 if bounded else 3.4, abs=1e-5)
