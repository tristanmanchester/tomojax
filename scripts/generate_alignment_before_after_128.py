from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align.geometry_blocks import (
    GeometryCalibrationState,
    geometry_with_axis_state,
    level_detector_grid,
    normalize_geometry_dofs,
    summarize_geometry_calibration_stats,
)
from tomojax.align.pipeline import AlignConfig, align_multires
from tomojax.calibration.detector_grid import detector_grid_from_calibration
from tomojax.core.geometry import Detector, Geometry, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.data.phantoms import random_cubes_spheres
from tomojax.recon.fbp import fbp
from tomojax.recon.quicklook import scale_to_uint8


PHANTOM_KIND = "random_shapes/full_volume_random_cubes_spheres"
PHANTOM_SOURCE = "tomojax.data.phantoms.random_cubes_spheres"
PHANTOM_SEED = 20260458
DEFAULT_LEVELS = (8, 4, 2, 1)


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    description: str
    geometry_type: str
    geometry_dofs: tuple[str, ...]
    hidden_det_u_px: float = 0.0
    hidden_det_v_px: float = 0.0
    hidden_detector_roll_deg: float = 0.0
    hidden_axis_rot_x_deg: float = 0.0
    hidden_axis_rot_y_deg: float = 0.0
    supplied_det_u_px: float | None = None
    supplied_det_v_px: float | None = None
    supplied_detector_roll_deg: float | None = None
    supplied_axis_rot_x_deg: float | None = None
    supplied_axis_rot_y_deg: float | None = None
    nominal_tilt_deg: float = 30.0
    theta_span_deg: float | None = None

    @property
    def true_tilt_deg(self) -> float:
        return float(self.nominal_tilt_deg + self.hidden_axis_rot_x_deg)


@dataclass(frozen=True)
class RunProfile:
    name: str
    size: int
    views: int
    levels: tuple[int, ...]
    outer_iters: int
    recon_iters: int
    tv_prox_iters: int
    views_per_batch: int
    gather_dtype: str
    early_stop: bool
    early_stop_rel_impr: float
    early_stop_patience: int


def docs_profile() -> RunProfile:
    return RunProfile(
        name="docs_128",
        size=128,
        views=128,
        levels=DEFAULT_LEVELS,
        outer_iters=12,
        recon_iters=20,
        tv_prox_iters=12,
        views_per_batch=1,
        gather_dtype="bf16",
        early_stop=True,
        early_stop_rel_impr=1e-3,
        early_stop_patience=2,
    )


def smoke_profile() -> RunProfile:
    return RunProfile(
        name="smoke_32",
        size=32,
        views=32,
        levels=(4, 2, 1),
        outer_iters=2,
        recon_iters=4,
        tv_prox_iters=4,
        views_per_batch=1,
        gather_dtype="fp32",
        early_stop=False,
        early_stop_rel_impr=1e-3,
        early_stop_patience=2,
    )


def _default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            slug="parallel_det_u_m004",
            title="Parallel CT: detector/ray-grid centre -4 px",
            description="Hidden detector-u centre offset. Estimated with the det_u_px block.",
            geometry_type="parallel",
            geometry_dofs=("det_u_px",),
            hidden_det_u_px=-4.0,
        ),
        Scenario(
            slug="parallel_det_u_p004",
            title="Parallel CT: detector/ray-grid centre +4 px",
            description="Opposite detector-u sign check for convention regressions.",
            geometry_type="parallel",
            geometry_dofs=("det_u_px",),
            hidden_det_u_px=4.0,
        ),
        Scenario(
            slug="parallel_detector_roll_p2p5",
            title="Parallel CT: detector roll +2.5 deg",
            description="Hidden detector-plane roll. Estimated with the detector_roll_deg block.",
            geometry_type="parallel",
            geometry_dofs=("detector_roll_deg",),
            hidden_detector_roll_deg=2.5,
        ),
        Scenario(
            slug="parallel_axis_pitch_p2p0",
            title="Parallel CT: axis pitched +2 deg",
            description="Rotation axis is tipped forward/backward in the lab frame.",
            geometry_type="parallel",
            geometry_dofs=("axis_rot_x_deg",),
            hidden_axis_rot_x_deg=2.0,
        ),
        Scenario(
            slug="parallel_axis_yaw_m2p0",
            title="Parallel CT: axis yawed -2 deg",
            description="Rotation axis has a side-to-side lab-frame component.",
            geometry_type="parallel",
            geometry_dofs=("axis_rot_y_deg",),
            hidden_axis_rot_y_deg=-2.0,
        ),
        Scenario(
            slug="parallel_det_u_roll_combo",
            title="Parallel CT: centre then roll",
            description="Combined detector-u centre offset and detector roll.",
            geometry_type="parallel",
            geometry_dofs=("det_u_px", "detector_roll_deg"),
            hidden_det_u_px=-3.0,
            hidden_detector_roll_deg=2.0,
        ),
        Scenario(
            slug="parallel_det_u_axis_refine",
            title="Parallel CT: centre plus axis direction",
            description="Combined centre offset and axis pitch, exercising staged blocks.",
            geometry_type="parallel",
            geometry_dofs=("det_u_px", "axis_rot_x_deg"),
            hidden_det_u_px=-3.0,
            hidden_axis_rot_x_deg=1.8,
        ),
        Scenario(
            slug="lamino_tilt_34p4",
            title="Laminography: true tilt 34.4 deg",
            description="Nominal tilt is 30 deg; the hidden instrument tilt delta is +4.4 deg.",
            geometry_type="lamino",
            geometry_dofs=("tilt_deg",),
            hidden_axis_rot_x_deg=4.4,
            nominal_tilt_deg=30.0,
        ),
        Scenario(
            slug="lamino_det_u_tilt_combo",
            title="Laminography: detector centre plus tilt",
            description="Combined -3 px detector-u offset and +4.4 deg tilt error.",
            geometry_type="lamino",
            geometry_dofs=("det_u_px", "tilt_deg"),
            hidden_det_u_px=-3.0,
            hidden_axis_rot_x_deg=4.4,
            nominal_tilt_deg=30.0,
        ),
        Scenario(
            slug="known_det_u_control",
            title="Control: supplied known detector centre",
            description="The known -4 px detector-u correction is supplied, not estimated.",
            geometry_type="parallel",
            geometry_dofs=(),
            hidden_det_u_px=-4.0,
            supplied_det_u_px=-4.0,
        ),
    ]


def scenario_catalog() -> list[Scenario]:
    return _default_scenarios()


def visual_stress_scenario_catalog() -> list[Scenario]:
    """More aggressive perturbations used to find visually useful naive-FBP demos."""
    return [
        Scenario(
            slug="stress_parallel_detector_roll_p10",
            title="Parallel CT: detector roll +10 deg",
            description="Large hidden detector-plane roll for visual naive-FBP artifact screening.",
            geometry_type="parallel",
            geometry_dofs=("detector_roll_deg",),
            hidden_detector_roll_deg=10.0,
            theta_span_deg=180.0,
        ),
        Scenario(
            slug="stress_parallel_axis_pitch_p18",
            title="Full-rotation arbitrary axis: pitch +18 deg",
            description=(
                "Large forward/backward lab-frame axis tilt with full angular coverage for visual "
                "artifact screening."
            ),
            geometry_type="parallel",
            geometry_dofs=("axis_rot_x_deg",),
            hidden_axis_rot_x_deg=18.0,
            theta_span_deg=360.0,
        ),
        Scenario(
            slug="stress_parallel_axis_yaw_m18",
            title="Full-rotation arbitrary axis: yaw -18 deg",
            description=(
                "Large side-to-side lab-frame axis tilt with full angular coverage for visual "
                "artifact screening."
            ),
            geometry_type="parallel",
            geometry_dofs=("axis_rot_y_deg",),
            hidden_axis_rot_y_deg=-18.0,
            theta_span_deg=360.0,
        ),
        Scenario(
            slug="stress_lamino_tilt_50",
            title="Laminography: true tilt 50 deg",
            description="Large hidden tilt delta from nominal 30 deg for visual artifact screening.",
            geometry_type="lamino",
            geometry_dofs=("tilt_deg",),
            hidden_axis_rot_x_deg=20.0,
            nominal_tilt_deg=30.0,
            theta_span_deg=360.0,
        ),
    ]


def scenario_catalog_for_kind(kind: str) -> list[Scenario]:
    if kind == "visual_stress":
        scenarios = visual_stress_scenario_catalog()
        _validate_visual_stress_acquisition(scenarios)
        return scenarios
    return scenario_catalog()


def profile_from_args(args: argparse.Namespace) -> RunProfile:
    base = docs_profile() if args.profile == "docs" else smoke_profile()
    return RunProfile(
        name=base.name,
        size=int(args.size or base.size),
        views=int(args.views or base.views),
        levels=tuple(int(v) for v in (args.levels or base.levels)),
        outer_iters=int(args.outer_iters if args.outer_iters is not None else base.outer_iters),
        recon_iters=int(args.recon_iters if args.recon_iters is not None else base.recon_iters),
        tv_prox_iters=int(
            args.tv_prox_iters if args.tv_prox_iters is not None else base.tv_prox_iters
        ),
        views_per_batch=int(
            args.views_per_batch if args.views_per_batch is not None else base.views_per_batch
        ),
        gather_dtype=str(args.gather_dtype or base.gather_dtype),
        early_stop=bool(args.early_stop if args.early_stop is not None else base.early_stop),
        early_stop_rel_impr=float(
            args.early_stop_rel_impr
            if args.early_stop_rel_impr is not None
            else base.early_stop_rel_impr
        ),
        early_stop_patience=int(
            args.early_stop_patience
            if args.early_stop_patience is not None
            else base.early_stop_patience
        ),
    )


def _phantom(size: int) -> np.ndarray:
    return random_cubes_spheres(
        size,
        size,
        size,
        n_cubes=4,
        n_spheres=7,
        min_size=max(5, size // 18),
        max_size=max(9, size // 8),
        min_value=0.45,
        max_value=1.0,
        seed=PHANTOM_SEED,
        use_inscribed_fov=True,
    ).astype(np.float32)


def _build_geometry(
    *,
    grid: Grid,
    detector: Detector,
    thetas: np.ndarray,
    geometry_type: str,
    tilt_deg: float,
) -> Geometry:
    if geometry_type == "lamino":
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(tilt_deg),
            tilt_about="x",
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)


def _theta_span_deg(scenario: Scenario) -> float:
    if scenario.theta_span_deg is not None:
        return float(scenario.theta_span_deg)
    if scenario.geometry_type == "lamino":
        return 360.0
    return 180.0


def _has_axis_direction_perturbation(scenario: Scenario) -> bool:
    return (
        abs(float(scenario.hidden_axis_rot_x_deg)) > 1e-7
        or abs(float(scenario.hidden_axis_rot_y_deg)) > 1e-7
    )


def _validate_visual_stress_acquisition(scenarios: Sequence[Scenario]) -> None:
    for scenario in scenarios:
        if _has_axis_direction_perturbation(scenario) and scenario.theta_span_deg is None:
            raise ValueError(
                f"Visual-stress axis scenario {scenario.slug!r} must set theta_span_deg "
                "explicitly so nominal geometry type does not silently choose acquisition span."
            )


def _state_from_values(
    geometry: Geometry,
    *,
    active_geometry_dofs: Sequence[str],
    det_u_px: float = 0.0,
    det_v_px: float = 0.0,
    detector_roll_deg: float = 0.0,
    axis_rot_x_deg: float = 0.0,
    axis_rot_y_deg: float = 0.0,
) -> GeometryCalibrationState:
    active = normalize_geometry_dofs(active_geometry_dofs, geometry=geometry)
    state = GeometryCalibrationState.from_geometry(geometry, active_geometry_dofs=active)
    return state.replace_values(
        (
            "det_u_px",
            "det_v_px",
            "detector_roll_deg",
            "axis_rot_x_deg",
            "axis_rot_y_deg",
        ),
        jnp.asarray(
            [det_u_px, det_v_px, detector_roll_deg, axis_rot_x_deg, axis_rot_y_deg],
            dtype=jnp.float32,
        ),
    )


def _hidden_state(scenario: Scenario, geometry: Geometry) -> GeometryCalibrationState:
    return _state_from_values(
        geometry,
        active_geometry_dofs=scenario.geometry_dofs,
        det_u_px=float(scenario.hidden_det_u_px),
        det_v_px=float(scenario.hidden_det_v_px),
        detector_roll_deg=float(scenario.hidden_detector_roll_deg),
        axis_rot_x_deg=float(scenario.hidden_axis_rot_x_deg),
        axis_rot_y_deg=float(scenario.hidden_axis_rot_y_deg),
    )


def _supplied_state(scenario: Scenario, geometry: Geometry) -> GeometryCalibrationState:
    return _state_from_values(
        geometry,
        active_geometry_dofs=(),
        det_u_px=float(scenario.supplied_det_u_px or 0.0),
        det_v_px=float(scenario.supplied_det_v_px or 0.0),
        detector_roll_deg=float(scenario.supplied_detector_roll_deg or 0.0),
        axis_rot_x_deg=float(scenario.supplied_axis_rot_x_deg or 0.0),
        axis_rot_y_deg=float(scenario.supplied_axis_rot_y_deg or 0.0),
    )


def _state_values(state: GeometryCalibrationState) -> dict[str, Any]:
    return {
        "det_u_px": float(state.det_u_px),
        "det_v_px": float(state.det_v_px),
        "detector_roll_deg": float(state.detector_roll_deg),
        "axis_rot_x_deg": float(state.axis_rot_x_deg),
        "axis_rot_y_deg": float(state.axis_rot_y_deg),
        "axis_unit_lab": [float(v) for v in state.axis_unit_lab()],
    }


def _simulate(
    scenario: Scenario,
    *,
    nominal_geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    views_per_batch: int,
    gather_dtype: str,
) -> jnp.ndarray:
    true_state = _hidden_state(scenario, nominal_geometry)
    true_geometry = geometry_with_axis_state(nominal_geometry, grid, detector, true_state)
    true_det_grid = level_detector_grid(detector, state=true_state, factor=1)
    chunks = []
    n_views = len(getattr(nominal_geometry, "thetas_deg"))
    for start in range(0, n_views, max(1, int(views_per_batch))):
        stop = min(start + max(1, int(views_per_batch)), n_views)
        chunk = [
            forward_project_view(
                true_geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype=gather_dtype,
                det_grid=true_det_grid,
            )
            for i in range(start, stop)
        ]
        chunks.append(jnp.stack(chunk, axis=0))
    return jnp.concatenate(chunks, axis=0)


def _run_fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    views_per_batch: int,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> np.ndarray:
    x = fbp(
        geometry,
        grid,
        detector,
        projections,
        views_per_batch=max(1, int(views_per_batch)),
        gather_dtype=gather_dtype,
        checkpoint_projector=True,
        det_grid=det_grid,
    )
    return np.asarray(x, dtype=np.float32)


def _run_geometry_alignment(
    scenario: Scenario,
    *,
    nominal_geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    profile: RunProfile,
) -> tuple[np.ndarray, GeometryCalibrationState, dict[str, Any]]:
    geometry_dofs = normalize_geometry_dofs(scenario.geometry_dofs, geometry=nominal_geometry)
    x_aligned, _, info = align_multires(
        nominal_geometry,
        grid,
        detector,
        projections,
        factors=profile.levels,
        cfg=AlignConfig(
            outer_iters=int(profile.outer_iters),
            recon_iters=int(profile.recon_iters),
            lambda_tv=0.0015,
            tv_prox_iters=int(profile.tv_prox_iters),
            geometry_dofs=geometry_dofs,
            freeze_dofs=("alpha", "beta", "phi", "dx", "dz"),
            early_stop=bool(profile.early_stop),
            early_stop_rel_impr=float(profile.early_stop_rel_impr),
            early_stop_patience=int(profile.early_stop_patience),
            gather_dtype=profile.gather_dtype,
            checkpoint_projector=True,
            views_per_batch=max(1, int(profile.views_per_batch)),
            projector_unroll=1,
            gn_damping=1e-3,
        ),
    )
    state = GeometryCalibrationState.from_checkpoint(
        info.get("geometry_calibration_state"),
        nominal_geometry,
        active_geometry_dofs=geometry_dofs,
    )
    return np.asarray(x_aligned, dtype=np.float32), state, dict(info)


def _slice_xy(volume: np.ndarray) -> np.ndarray:
    return volume[:, :, volume.shape[2] // 2].T


def _slice_xz(volume: np.ndarray) -> np.ndarray:
    return volume[:, volume.shape[1] // 2, :].T


def _slice_yz(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2, :, :].T


def _scale(image: np.ndarray) -> np.ndarray:
    return scale_to_uint8(image, lower_percentile=1.0, upper_percentile=99.7)


def _label_bar(width: int, text: str, *, height: int = 28) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 18)
    draw = ImageDraw.Draw(img)
    draw.text((8, 7), text[:170], fill=238)
    return np.asarray(img, dtype=np.uint8)


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    out = np.full((height, image.shape[1]), 20, dtype=np.uint8)
    out[: image.shape[0], : image.shape[1]] = image
    return out


def _hstack(images: list[np.ndarray], *, pad: int = 6) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    padded = [_pad_to_height(im, height) for im in images]
    width = sum(im.shape[1] for im in padded) + pad * (len(padded) - 1)
    out = np.full((height, width), 20, dtype=np.uint8)
    x = 0
    for image in padded:
        out[:, x : x + image.shape[1]] = image
        x += image.shape[1] + pad
    return out


def _vstack(images: list[np.ndarray], *, pad: int = 8) -> np.ndarray:
    width = max(im.shape[1] for im in images)
    height = sum(im.shape[0] for im in images) + pad * (len(images) - 1)
    out = np.full((height, width), 20, dtype=np.uint8)
    y = 0
    for image in images:
        out[y : y + image.shape[0], : image.shape[1]] = image
        y += image.shape[0] + pad
    return out


def _volume_nmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.mean(np.square(truth))) + 1e-6
    return float(np.mean(np.square(candidate - truth)) / denom)


def _write_visuals(
    scenario: Scenario,
    *,
    out_dir: Path,
    truth: np.ndarray,
    naive_fbp: np.ndarray,
    calibrated_fbp: np.ndarray,
    aligned_tv: np.ndarray,
    estimates: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_xy = _scale(_slice_xy(truth))
    naive_xy = _scale(_slice_xy(naive_fbp))
    calibrated_xy = _scale(_slice_xy(calibrated_fbp))
    tv_xy = _scale(_slice_xy(aligned_tv))
    delta_xy = _scale(np.abs(_slice_xy(naive_fbp) - _slice_xy(calibrated_fbp)))
    calibrated_orthos = _hstack(
        [
            _scale(_slice_xy(calibrated_fbp)),
            _scale(_slice_xz(calibrated_fbp)),
            _scale(_slice_yz(calibrated_fbp)),
        ]
    )
    truth_orthos = _hstack([_scale(_slice_xy(truth)), _scale(_slice_xz(truth)), _scale(_slice_yz(truth))])
    header = _hstack(
        [
            _label_bar(truth_xy.shape[1], "truth"),
            _label_bar(naive_xy.shape[1], "naive FBP"),
            _label_bar(calibrated_xy.shape[1], "calibrated FBP"),
            _label_bar(tv_xy.shape[1], "aligned TV"),
            _label_bar(delta_xy.shape[1], "naive/calib delta"),
        ],
        pad=6,
    )
    body = _hstack([truth_xy, naive_xy, calibrated_xy, tv_xy, delta_xy])
    subtitle = (
        f"{scenario.slug} | {scenario.title} | dofs={','.join(scenario.geometry_dofs) or 'none'} | "
        f"est det_u={estimates.get('det_u_px', 0.0):.3g} roll={estimates.get('detector_roll_deg', 0.0):.3g} "
        f"axis=({estimates.get('axis_rot_x_deg', 0.0):.3g},{estimates.get('axis_rot_y_deg', 0.0):.3g})"
    )
    panel = _vstack([_label_bar(body.shape[1], subtitle), header, body], pad=0)
    paths = {
        "truth_xy": str(out_dir / "truth_xy.png"),
        "naive_fbp_xy": str(out_dir / "naive_fbp_xy.png"),
        "calibrated_fbp_xy": str(out_dir / "calibrated_fbp_xy.png"),
        "aligned_tv_xy": str(out_dir / "aligned_tv_xy.png"),
        "before_after_panel": str(out_dir / "before_after_panel.png"),
        "truth_orthos": str(out_dir / "truth_orthos.png"),
        "calibrated_orthos": str(out_dir / "calibrated_fbp_orthos.png"),
        "absolute_difference_xy": str(out_dir / "absolute_difference_xy.png"),
    }
    iio.imwrite(paths["truth_xy"], truth_xy)
    iio.imwrite(paths["naive_fbp_xy"], naive_xy)
    iio.imwrite(paths["calibrated_fbp_xy"], calibrated_xy)
    iio.imwrite(paths["aligned_tv_xy"], tv_xy)
    iio.imwrite(paths["before_after_panel"], panel)
    iio.imwrite(paths["truth_orthos"], truth_orthos)
    iio.imwrite(paths["calibrated_orthos"], calibrated_orthos)
    iio.imwrite(paths["absolute_difference_xy"], delta_xy)
    return paths


def _write_naive_visuals(
    scenario: Scenario,
    *,
    out_dir: Path,
    truth: np.ndarray,
    naive_fbp: np.ndarray,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_xy = _scale(_slice_xy(truth))
    naive_xy = _scale(_slice_xy(naive_fbp))
    diff_xy = _scale(np.abs(_slice_xy(truth) - _slice_xy(naive_fbp)))
    truth_orthos = _hstack(
        [_scale(_slice_xy(truth)), _scale(_slice_xz(truth)), _scale(_slice_yz(truth))]
    )
    naive_orthos = _hstack(
        [_scale(_slice_xy(naive_fbp)), _scale(_slice_xz(naive_fbp)), _scale(_slice_yz(naive_fbp))]
    )
    header = _hstack(
        [
            _label_bar(truth_xy.shape[1], "truth"),
            _label_bar(naive_xy.shape[1], "naive FBP"),
            _label_bar(diff_xy.shape[1], "abs truth/naive diff"),
        ],
        pad=6,
    )
    body = _hstack([truth_xy, naive_xy, diff_xy])
    subtitle = (
        f"{scenario.slug} | {scenario.title} | hidden "
        f"det_u={scenario.hidden_det_u_px:g} roll={scenario.hidden_detector_roll_deg:g} "
        f"axis=({scenario.hidden_axis_rot_x_deg:g},{scenario.hidden_axis_rot_y_deg:g})"
    )
    panel = _vstack([_label_bar(body.shape[1], subtitle), header, body], pad=0)
    paths = {
        "truth_xy": str(out_dir / "truth_xy.png"),
        "naive_fbp_xy": str(out_dir / "naive_fbp_xy.png"),
        "calibrated_fbp_xy": "",
        "aligned_tv_xy": "",
        "before_after_panel": str(out_dir / "truth_vs_naive_panel.png"),
        "truth_orthos": str(out_dir / "truth_orthos.png"),
        "calibrated_orthos": str(out_dir / "naive_fbp_orthos.png"),
        "absolute_difference_xy": str(out_dir / "truth_naive_absolute_difference_xy.png"),
    }
    iio.imwrite(paths["truth_xy"], truth_xy)
    iio.imwrite(paths["naive_fbp_xy"], naive_xy)
    iio.imwrite(paths["before_after_panel"], panel)
    iio.imwrite(paths["truth_orthos"], truth_orthos)
    iio.imwrite(paths["calibrated_orthos"], naive_orthos)
    iio.imwrite(paths["absolute_difference_xy"], diff_xy)
    return paths


def _scenario_truth_payload(scenario: Scenario) -> dict[str, float]:
    return {
        "det_u_px": float(scenario.hidden_det_u_px),
        "det_v_px": float(scenario.hidden_det_v_px),
        "detector_roll_deg": float(scenario.hidden_detector_roll_deg),
        "axis_rot_x_deg": float(scenario.hidden_axis_rot_x_deg),
        "axis_rot_y_deg": float(scenario.hidden_axis_rot_y_deg),
        "nominal_tilt_deg": float(scenario.nominal_tilt_deg),
        "true_tilt_deg": float(scenario.true_tilt_deg),
    }


def _scenario_supplied_payload(scenario: Scenario) -> dict[str, float]:
    supplied: dict[str, float] = {}
    for name in (
        "det_u_px",
        "det_v_px",
        "detector_roll_deg",
        "axis_rot_x_deg",
        "axis_rot_y_deg",
    ):
        value = getattr(scenario, f"supplied_{name}")
        if value is not None:
            supplied[name] = float(value)
    return supplied


def _geometry_status_label(diagnostics: Any) -> str:
    if not isinstance(diagnostics, dict):
        return ""
    overall = diagnostics.get("overall_status")
    if isinstance(overall, str) and overall:
        return overall
    blocks = diagnostics.get("blocks")
    if not isinstance(blocks, list):
        return ""
    statuses = [
        str(block.get("status"))
        for block in blocks
        if isinstance(block, dict) and block.get("status")
    ]
    if not statuses:
        return ""
    if "ill_conditioned" in statuses:
        return "ill_conditioned"
    if "underconverged" in statuses:
        return "underconverged"
    if all(status == "converged" for status in statuses):
        return "converged"
    return ",".join(statuses)


def build_run_manifest(profile: RunProfile, scenarios: Sequence[Scenario]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "scripts/generate_alignment_before_after_128.py",
        "purpose": "geometry_block_before_after_taxonomy",
        "phantom": {
            "kind": PHANTOM_KIND,
            "seed": PHANTOM_SEED,
            "shared_across_cases": True,
            "source": PHANTOM_SOURCE,
        },
        "profile": asdict(profile),
        "scenario_set": "default",
        "scenarios": [
            {
                "slug": s.slug,
                "title": s.title,
                "description": s.description,
                "geometry_type": s.geometry_type,
                "geometry_dofs": list(s.geometry_dofs),
                "theta_span_deg": _theta_span_deg(s),
                "n_views": int(profile.views),
                "hidden_truth": _scenario_truth_payload(s),
                "supplied_corrections": _scenario_supplied_payload(s),
            }
            for s in scenarios
        ],
        "gauge_notes": {
            "det_u_px": (
                "Canonical detector/ray-grid centre offset in native detector pixels under the "
                "detector_ray_grid_center gauge; not a standalone physical COR proof."
            ),
            "supplied_controls": "Supplied known corrections are controls and are not reported as estimated.",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_scenario(
    scenario: Scenario,
    *,
    out_dir: Path,
    profile: RunProfile,
    grid: Grid,
    detector: Detector,
    truth: np.ndarray,
    naive_only: bool = False,
) -> dict[str, Any]:
    start_time = time.time()
    volume = jnp.asarray(truth, dtype=jnp.float32)
    theta_span = _theta_span_deg(scenario)
    thetas = np.linspace(0.0, theta_span, int(profile.views), endpoint=False, dtype=np.float32)
    nominal_geometry = _build_geometry(
        grid=grid,
        detector=detector,
        thetas=thetas,
        geometry_type=scenario.geometry_type,
        tilt_deg=scenario.nominal_tilt_deg,
    )
    projections = _simulate(
        scenario,
        nominal_geometry=nominal_geometry,
        grid=grid,
        detector=detector,
        volume=volume,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
    )
    projections.block_until_ready()

    naive_fbp = _run_fbp(
        nominal_geometry,
        grid,
        detector,
        projections,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
    )

    if naive_only:
        visual_paths = _write_naive_visuals(
            scenario,
            out_dir=out_dir,
            truth=truth,
            naive_fbp=naive_fbp,
        )
        elapsed = time.time() - start_time
        row: dict[str, Any] = {
            "slug": scenario.slug,
            "title": scenario.title,
            "geometry_type": scenario.geometry_type,
            "geometry_dofs": ",".join(scenario.geometry_dofs),
            "theta_span_deg": theta_span,
            "n_views": int(profile.views),
            "parameter_provenance": "naive_only",
            "hidden_truth_json": json.dumps(_scenario_truth_payload(scenario), sort_keys=True),
            "supplied_corrections_json": "{}",
            "estimates_json": "{}",
            "geometry_diagnostics_json": "{}",
            "geometry_status": "",
            "naive_volume_nmse": _volume_nmse(naive_fbp, truth),
            "calibrated_volume_nmse": np.nan,
            "aligned_tv_volume_nmse": np.nan,
            "total_outer_iters": 0,
            "elapsed_sec": elapsed,
            "error": "",
            **visual_paths,
        }
        _write_json(
            out_dir / "case_manifest.json",
            {
                "schema_version": 1,
                "scenario": asdict(scenario),
                "phantom": {
                    "kind": PHANTOM_KIND,
                    "seed": PHANTOM_SEED,
                    "shared_across_cases": True,
                    "source": PHANTOM_SOURCE,
                },
                "profile": asdict(profile),
                "acquisition": {
                    "theta_span_deg": theta_span,
                    "n_views": int(profile.views),
                    "geometry_type": scenario.geometry_type,
                },
                "hidden_truth": _scenario_truth_payload(scenario),
                "parameter_provenance": "naive_only",
                "metrics": {"naive_volume_nmse": row["naive_volume_nmse"]},
                "artifacts": visual_paths,
                "elapsed_sec": elapsed,
            },
        )
        jax.clear_caches()
        return row

    supplied = _scenario_supplied_payload(scenario)
    if scenario.geometry_dofs:
        aligned_tv, state, info = _run_geometry_alignment(
            scenario,
            nominal_geometry=nominal_geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            profile=profile,
        )
        provenance = "estimated"
    elif supplied:
        state = _supplied_state(scenario, nominal_geometry)
        calibrated_geometry_for_tv = geometry_with_axis_state(nominal_geometry, grid, detector, state)
        supplied_grid = level_detector_grid(detector, state=state, factor=1)
        aligned_tv = _run_fbp(
            calibrated_geometry_for_tv,
            grid,
            detector,
            projections,
            views_per_batch=profile.views_per_batch,
            gather_dtype=profile.gather_dtype,
            det_grid=supplied_grid,
        )
        info = {
            "geometry_calibration_state": state.to_calibration_state().to_dict(),
            "geometry_calibration_diagnostics": {},
            "outer_stats": [],
            "total_outer_iters": 0,
            "control": "supplied_known_correction",
        }
        provenance = "supplied"
    else:
        aligned_tv, state, info = _run_geometry_alignment(
            scenario,
            nominal_geometry=nominal_geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            profile=profile,
        )
        provenance = "frozen"
    if "geometry_calibration_diagnostics" not in info:
        info["geometry_calibration_diagnostics"] = summarize_geometry_calibration_stats(
            info.get("outer_stats", [])
        )

    calibrated_geometry = geometry_with_axis_state(nominal_geometry, grid, detector, state)
    calibrated_det_grid = level_detector_grid(detector, state=state, factor=1)
    calibrated_fbp = _run_fbp(
        calibrated_geometry,
        grid,
        detector,
        projections,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
        det_grid=calibrated_det_grid,
    )

    estimates = _state_values(state)
    visual_paths = _write_visuals(
        scenario,
        out_dir=out_dir,
        truth=truth,
        naive_fbp=naive_fbp,
        calibrated_fbp=calibrated_fbp,
        aligned_tv=aligned_tv,
        estimates=estimates,
    )
    elapsed = time.time() - start_time
    row: dict[str, Any] = {
        "slug": scenario.slug,
        "title": scenario.title,
        "geometry_type": scenario.geometry_type,
        "geometry_dofs": ",".join(scenario.geometry_dofs),
        "theta_span_deg": theta_span,
        "n_views": int(profile.views),
        "parameter_provenance": provenance,
        "hidden_truth_json": json.dumps(_scenario_truth_payload(scenario), sort_keys=True),
        "supplied_corrections_json": json.dumps(supplied, sort_keys=True),
        "estimates_json": json.dumps(estimates, sort_keys=True),
        "geometry_diagnostics_json": json.dumps(
            info.get("geometry_calibration_diagnostics", {}), sort_keys=True
        ),
        "geometry_status": _geometry_status_label(
            info.get("geometry_calibration_diagnostics", {})
        ),
        "naive_volume_nmse": _volume_nmse(naive_fbp, truth),
        "calibrated_volume_nmse": _volume_nmse(calibrated_fbp, truth),
        "aligned_tv_volume_nmse": _volume_nmse(aligned_tv, truth),
        "total_outer_iters": int(info.get("total_outer_iters", 0)),
        "elapsed_sec": elapsed,
        "error": "",
        **visual_paths,
    }
    manifest = {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "phantom": {
            "kind": PHANTOM_KIND,
            "seed": PHANTOM_SEED,
            "shared_across_cases": True,
            "source": PHANTOM_SOURCE,
        },
        "profile": asdict(profile),
        "acquisition": {
            "theta_span_deg": theta_span,
            "n_views": int(profile.views),
            "geometry_type": scenario.geometry_type,
        },
        "hidden_truth": _scenario_truth_payload(scenario),
        "supplied_corrections": supplied,
        "estimated_corrections": estimates if provenance == "estimated" else {},
        "final_calibrated_geometry": estimates,
        "parameter_provenance": provenance,
        "calibration_state": info.get("geometry_calibration_state"),
        "geometry_calibration_diagnostics": info.get("geometry_calibration_diagnostics", {}),
        "outer_stats": info.get("outer_stats", []),
        "metrics": {
            "naive_volume_nmse": row["naive_volume_nmse"],
            "calibrated_volume_nmse": row["calibrated_volume_nmse"],
            "aligned_tv_volume_nmse": row["aligned_tv_volume_nmse"],
        },
        "artifacts": visual_paths,
        "elapsed_sec": elapsed,
    }
    _write_json(out_dir / "case_manifest.json", manifest)
    jax.clear_caches()
    return row


def _select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = scenario_catalog_for_kind(str(args.scenario_set))
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.slug in wanted]
        missing = sorted(wanted - {s.slug for s in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
    if args.limit is not None:
        scenarios = scenarios[: int(args.limit)]
    return scenarios


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_master_panel(rows: list[dict[str, Any]], master_path: Path) -> None:
    panels: list[np.ndarray] = []
    for row in rows:
        panel_path = row.get("before_after_panel")
        if not isinstance(panel_path, str) or not panel_path.strip():
            continue
        path = Path(panel_path)
        if path.is_file():
            panels.append(iio.imread(path))
    if panels:
        iio.imwrite(master_path, _vstack(panels, pad=10))


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    artifacts = out_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    profile = profile_from_args(args)
    scenarios = _select_scenarios(args)
    manifest = build_run_manifest(profile, scenarios)
    manifest["scenario_set"] = str(args.scenario_set)
    manifest["naive_only"] = bool(args.naive_only)
    _write_json(out_root / "run_manifest.json", manifest)
    _write_json(artifacts / "scenario_catalog.json", manifest["scenarios"])

    if args.dry_run:
        _write_json(
            artifacts / "status.json",
            {
                "state": "dry_run_completed",
                "profile": asdict(profile),
                "scenario_count": len(scenarios),
                "scenarios": [s.slug for s in scenarios],
                "run_manifest": str(out_root / "run_manifest.json"),
            },
        )
        return

    grid = Grid(profile.size, profile.size, profile.size, 1.0, 1.0, 1.0)
    detector = Detector(profile.size, profile.size, 1.0, 1.0, det_center=(0.0, 0.0))
    truth = _phantom(profile.size)
    rows: list[dict[str, Any]] = []
    summary_path = artifacts / "summary.csv"
    master_path = artifacts / "alignment_before_after_master.png"

    for index, scenario in enumerate(scenarios, start=1):
        _write_json(
            artifacts / "status.json",
            {
                "state": "running",
                "scenario": scenario.slug,
                "index": index,
                "total": len(scenarios),
                "profile": asdict(profile),
                "summary_csv": str(summary_path),
                "master_panel": str(master_path),
            },
        )
        try:
            row = _run_scenario(
                scenario,
                out_dir=artifacts / scenario.slug,
                profile=profile,
                grid=grid,
                detector=detector,
                truth=truth,
                naive_only=bool(args.naive_only),
            )
            row["status"] = "completed"
        except Exception as exc:
            row = {
                "slug": scenario.slug,
                "title": scenario.title,
                "geometry_type": scenario.geometry_type,
                "geometry_dofs": ",".join(scenario.geometry_dofs),
                "theta_span_deg": _theta_span_deg(scenario),
                "n_views": int(profile.views),
                "parameter_provenance": "failed",
                "hidden_truth_json": json.dumps(_scenario_truth_payload(scenario), sort_keys=True),
                "supplied_corrections_json": json.dumps(
                    _scenario_supplied_payload(scenario), sort_keys=True
                ),
                "estimates_json": "{}",
                "geometry_diagnostics_json": "{}",
                "geometry_status": "",
                "naive_volume_nmse": np.nan,
                "calibrated_volume_nmse": np.nan,
                "aligned_tv_volume_nmse": np.nan,
                "total_outer_iters": 0,
                "elapsed_sec": 0.0,
                "truth_xy": "",
                "naive_fbp_xy": "",
                "calibrated_fbp_xy": "",
                "aligned_tv_xy": "",
                "before_after_panel": "",
                "truth_orthos": "",
                "calibrated_orthos": "",
                "absolute_difference_xy": "",
                "status": "failed",
                "error": repr(exc),
            }
            _write_json(
                artifacts / scenario.slug / "case_manifest.json",
                {
                    "schema_version": 1,
                    "scenario": asdict(scenario),
                    "profile": asdict(profile),
                    "status": "failed",
                    "error": repr(exc),
                },
            )
            if not args.continue_on_error:
                rows.append(row)
                _write_summary(rows, summary_path)
                raise
        rows.append(row)
        _write_summary(rows, summary_path)
        _write_master_panel(rows, master_path)

    _write_json(
        artifacts / "status.json",
        {
            "state": "completed",
            "profile": asdict(profile),
            "scenarios": [row["slug"] for row in rows],
            "summary_csv": str(summary_path),
            "master_panel": str(master_path),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=["docs", "smoke"], default="docs")
    parser.add_argument("--scenario-set", choices=["default", "visual_stress"], default="default")
    parser.add_argument("--naive-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--views-per-batch", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--outer-iters", type=int, default=None)
    parser.add_argument("--recon-iters", type=int, default=None)
    parser.add_argument("--tv-prox-iters", type=int, default=None)
    early_stop = parser.add_mutually_exclusive_group()
    early_stop.add_argument("--early-stop", dest="early_stop", action="store_true", default=None)
    early_stop.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--early-stop-rel-impr", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--gather-dtype", default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
