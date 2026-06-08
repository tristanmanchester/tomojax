"""Runtime planning for the reconstruction CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

from tomojax.backends import default_gather_dtype, estimate_views_per_batch_info
from tomojax.geometry import (
    Detector,
    Geometry,
    Grid,
    detector_grid_from_geometry_inputs,
)
from tomojax.io import (
    ProjectionDataset,
    build_geometry_from_dataset_metadata,
    load_projection_payload,
)
from tomojax.recon.api import (
    ReconstructionAlgorithmOptions,
    ReconstructionAlgorithmRequest,
)

from ._recon_command import ReconCommand
from ._reconstruction_region import resolve_reconstruction_region, solver_volume_mask

if TYPE_CHECKING:
    from tomojax.io import JsonValue


@dataclass(frozen=True)
class ReconRuntimePlan:
    """Resolved runtime inputs for one reconstruction execution."""

    meta: ProjectionDataset
    geometry_meta: dict[str, object]
    input_grid: Grid
    recon_grid: Grid
    detector: Detector
    detector_center_override: JsonValue
    geometry: Geometry
    projections: jnp.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    roi_mode: str
    is_parallel: bool
    volume_mask: jnp.ndarray | None
    views_per_batch: int
    views_per_batch_mode: str
    gather_dtype: str
    algorithm_request: ReconstructionAlgorithmRequest


def build_recon_runtime_plan(command: ReconCommand) -> ReconRuntimePlan:
    """Resolve dataset metadata, geometry, batching, and algorithm inputs."""
    meta = load_projection_payload(command.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = (
        command.grid if (meta.grid is None and command.grid is not None) else None
    )
    grid, detector, geom = build_geometry_from_dataset_metadata(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=bool(command.apply_saved_alignment),
    )
    detector, detector_center_override = _apply_detector_center_override(
        detector,
        geometry_meta,
        det_u_px=command.det_u_px,
        det_v_px=command.det_v_px,
    )
    if command.det_u_px is not None or command.det_v_px is not None:
        grid, detector, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=initial_grid_override,
            apply_saved_alignment=bool(command.apply_saved_alignment),
        )
    if command.apply_saved_alignment and meta.align_params is not None:
        logging.info("Applying saved per-view alignment parameters from input metadata")

    projections = _jnp_float32_array(meta.projections)
    det_grid = detector_grid_from_geometry_inputs(detector, geometry_meta)
    if det_grid is not None:
        logging.info(
            "Applying saved detector_roll_deg=%s from geometry metadata",
            geometry_meta.get("detector_roll_deg"),
        )

    gather_dtype = _resolve_gather_dtype(command.gather_dtype)
    region = resolve_reconstruction_region(
        grid,
        detector,
        geometry_type=str(meta.geometry_type),
        roi_mode=str(command.roi),
        grid_override=command.grid,
        mask_mode=str(command.mask_vol),
    )
    recon_grid = region.recon_grid
    is_parallel = meta.geometry_type == "parallel"

    if recon_grid is not grid:
        # Once ROI and explicit sizing resolve an effective grid, keep that grid's
        # origin/centre metadata authoritative when rebuilding geometry.
        _, _, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=bool(command.apply_saved_alignment),
        )

    volume_mask = solver_volume_mask(region, detector)
    views_per_batch, views_per_batch_mode = _resolve_views_per_batch(
        command.views_per_batch,
        algo=str(command.algo),
        n_views=int(projections.shape[0]),
        grid=recon_grid,
        detector=detector,
        gather_dtype=gather_dtype,
        checkpoint_projector=bool(command.checkpoint_projector),
    )
    logging.info(
        "Reconstruction views_per_batch=%d (mode=%s, algo=%s)",
        views_per_batch,
        views_per_batch_mode,
        command.algo,
    )

    algorithm_request = _reconstruction_algorithm_request(
        command,
        geom=geom,
        recon_grid=recon_grid,
        detector=detector,
        projections=projections,
        det_grid=det_grid,
        volume_mask=volume_mask,
        resolved_views_per_batch=views_per_batch,
        views_per_batch_mode=views_per_batch_mode,
        gather_dtype=gather_dtype,
    )
    return ReconRuntimePlan(
        meta=meta,
        geometry_meta=geometry_meta,
        input_grid=grid,
        recon_grid=recon_grid,
        detector=detector,
        detector_center_override=detector_center_override,
        geometry=geom,
        projections=projections,
        detector_grid=det_grid,
        roi_mode=str(region.roi_metadata["requested"]),
        is_parallel=bool(is_parallel),
        volume_mask=volume_mask,
        views_per_batch=views_per_batch,
        views_per_batch_mode=views_per_batch_mode,
        gather_dtype=gather_dtype,
        algorithm_request=algorithm_request,
    )


def _jnp_float32_array(value: object) -> jnp.ndarray:
    return jnp.asarray(value, dtype=np.float32)  # pyright: ignore[reportUnknownMemberType]


def _resolve_gather_dtype(requested: str) -> str:
    value = str(requested)
    return default_gather_dtype() if value == "auto" else value


def _default_views_per_batch(algo: str) -> int:
    return 16 if str(algo).lower() == "spdhg" else 1


def _resolve_views_per_batch(
    requested: int | str | None,
    *,
    algo: str,
    n_views: int,
    grid: Grid,
    detector: Detector,
    gather_dtype: str,
    checkpoint_projector: bool,
) -> tuple[int, str]:
    """Resolve CLI batching after ROI/grid choices are known."""
    if requested is None:
        return _default_views_per_batch(algo), "default"

    if isinstance(requested, str) and requested.lower() == "auto":
        estimate = estimate_views_per_batch_info(
            n_views=int(n_views),
            grid_nxyz=(int(grid.nx), int(grid.ny), int(grid.nz)),
            det_nuv=(int(detector.nv), int(detector.nu)),
            gather_dtype=str(gather_dtype),
            projection_dtype="fp32",
            volume_dtype="fp32",
            checkpoint_projector=bool(checkpoint_projector),
            algo=str(algo),
            fallback_batch=1,
        )
        if estimate.fallback_used:
            logging.warning(
                "Could not determine available memory for --views-per-batch auto; "
                "using views_per_batch=%d",
                estimate.views_per_batch,
            )
        return int(estimate.views_per_batch), "auto"

    return max(1, int(requested)), "explicit"


def _apply_detector_center_override(
    detector: Detector,
    geometry_meta: dict[str, object],
    *,
    det_u_px: float | None,
    det_v_px: float | None,
) -> tuple[Detector, dict[str, JsonValue]]:
    """Apply CLI detector-centre overrides specified in detector pixels."""
    requested: dict[str, JsonValue] = {"det_u_px": det_u_px, "det_v_px": det_v_px}
    if det_u_px is None and det_v_px is None:
        u_px = float(detector.det_center[0]) / float(detector.du)
        v_px = float(detector.det_center[1]) / float(detector.dv)
        metadata_override: dict[str, JsonValue] = {
            "source": "metadata",
            "requested_px": requested,
            "effective_px": {"det_u_px": u_px, "det_v_px": v_px},
            "effective_world": {
                "det_u": float(detector.det_center[0]),
                "det_v": float(detector.det_center[1]),
            },
        }
        return detector, metadata_override
    u_px = float(detector.det_center[0]) / float(detector.du)
    v_px = float(detector.det_center[1]) / float(detector.dv)
    if det_u_px is not None:
        u_px = float(det_u_px)
    if det_v_px is not None:
        v_px = float(det_v_px)
    updated = replace(
        detector,
        det_center=(u_px * float(detector.du), v_px * float(detector.dv)),
    )
    detector_meta = dict(cast("dict[str, object]", geometry_meta.get("detector", {})))
    detector_meta["det_center"] = [float(updated.det_center[0]), float(updated.det_center[1])]
    geometry_meta["detector"] = detector_meta
    override: dict[str, JsonValue] = {
        "source": "cli_override",
        "requested_px": requested,
        "effective_px": {"det_u_px": u_px, "det_v_px": v_px},
        "effective_world": {
            "det_u": float(updated.det_center[0]),
            "det_v": float(updated.det_center[1]),
        },
    }
    logging.info(
        "Applying detector centre override: det_u_px=%.3f det_v_px=%.3f",
        u_px,
        v_px,
    )
    return updated, override


def _reconstruction_algorithm_request(
    command: ReconCommand,
    *,
    geom: Geometry,
    recon_grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    volume_mask: jnp.ndarray | None,
    resolved_views_per_batch: int,
    views_per_batch_mode: str,
    gather_dtype: str,
) -> ReconstructionAlgorithmRequest:
    options = ReconstructionAlgorithmOptions(
        algorithm=command.algo,
        filter_name=str(command.filter),
        iters=int(command.iters),
        lambda_tv=float(command.lambda_tv),
        regulariser=command.regulariser,
        huber_delta=float(command.huber_delta),
        lipschitz=float(command.lipschitz) if command.lipschitz is not None else None,
        positivity=bool(command.positivity),
        lower_bound=float(command.lower_bound) if command.lower_bound is not None else None,
        upper_bound=float(command.upper_bound) if command.upper_bound is not None else None,
        theta=float(command.theta),
        spdhg_seed=int(command.spdhg_seed),
        spdhg_tau=float(command.spdhg_tau) if command.spdhg_tau is not None else None,
        spdhg_sigma_data=(
            float(command.spdhg_sigma_data) if command.spdhg_sigma_data is not None else None
        ),
        spdhg_sigma_tv=(
            float(command.spdhg_sigma_tv) if command.spdhg_sigma_tv is not None else None
        ),
        warm_start=command.warm_start,
        checkpoint_projector=bool(command.checkpoint_projector),
        tv_prox_iters=int(command.tv_prox_iters),
    )
    return ReconstructionAlgorithmRequest(
        options=options,
        geometry=geom,
        grid=recon_grid,
        detector=detector,
        projections=projections,
        detector_grid=det_grid,
        volume_mask=volume_mask,
        views_per_batch=int(resolved_views_per_batch),
        views_per_batch_mode=str(views_per_batch_mode),
        gather_dtype=str(gather_dtype),
    )


__all__ = ["ReconRuntimePlan", "build_recon_runtime_plan"]
