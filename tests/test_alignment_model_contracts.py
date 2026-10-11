from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_blocks import _theta_span_from_geometry

# check-public-imports: allow-private
from tomojax.alignment._geometry.initializers import (
    _vo_mask,
    _vo_metrics,
    projection_pair_det_u_seed,
    sinogram_det_u_seed,
)

# check-public-imports: allow-private
from tomojax.alignment._stages._reconstruction_stage import _is_oom_error_message
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage, SetupGeometryState
from tomojax.geometry import Detector, Grid, ParallelGeometry


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
def test_sinogram_real_fourier_metric_matches_full_spectrum(columns):
    sino = np.random.default_rng(3).normal(size=(37, columns)).astype(np.float32)
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
