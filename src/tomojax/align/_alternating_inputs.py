"""Input assembly for alternating smoke runs."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.datasets import load_synthetic_dataset_sidecars, make_benchmark_phantom
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState

if TYPE_CHECKING:
    from tomojax.align._alternating_types import AlternatingSmokeConfig


@dataclass(frozen=True)
class SmokeInputs:
    """Resolved arrays and geometry for one alternating smoke run."""

    truth_volume: jax.Array
    observed_projections: jax.Array
    mask: jax.Array
    true_geometry: GeometryState
    initial_geometry: GeometryState


def build_smoke_inputs(config: AlternatingSmokeConfig) -> SmokeInputs:
    """Resolve in-memory or sidecar-backed deterministic smoke inputs."""
    if config.synthetic_dataset_artifact_dir is not None:
        return _with_projection_loss_mask(_sidecar_smoke_inputs(config), config=config)
    return _with_projection_loss_mask(_default_smoke_inputs(config), config=config)


def _default_smoke_inputs(config: AlternatingSmokeConfig) -> SmokeInputs:
    truth = jnp.asarray(make_benchmark_phantom(config.size, seed=config.seed), dtype=jnp.float32)
    true_geometry = _synthetic_true_geometry(config.n_views)
    initial_geometry = _synthetic_initial_geometry(config.n_views)
    observed = project_parallel_reference(truth, true_geometry)
    return SmokeInputs(
        truth_volume=truth,
        observed_projections=observed,
        mask=jnp.ones_like(observed, dtype=jnp.float32),
        true_geometry=true_geometry,
        initial_geometry=initial_geometry,
    )


def _sidecar_smoke_inputs(config: AlternatingSmokeConfig) -> SmokeInputs:
    if config.synthetic_dataset_artifact_dir is None:
        raise ValueError("synthetic_dataset_artifact_dir is required for sidecar inputs")
    sidecars = load_synthetic_dataset_sidecars(config.synthetic_dataset_artifact_dir)
    if not sidecars.consistency.passed:
        raise ValueError("synthetic sidecar consistency checks must pass before ingestion")
    truth = jnp.asarray(np.load(sidecars.volume.path), dtype=jnp.float32)
    observed = jnp.asarray(np.load(sidecars.projections.path), dtype=jnp.float32)
    mask = jnp.asarray(np.load(sidecars.mask.path), dtype=jnp.float32)
    _validate_sidecar_shapes(
        truth=truth,
        observed=observed,
        mask=mask,
        expected_size=config.size,
        expected_views=config.n_views,
    )
    return SmokeInputs(
        truth_volume=truth,
        observed_projections=observed,
        mask=mask,
        true_geometry=sidecars.true_geometry,
        initial_geometry=sidecars.corrupted_geometry,
    )


def _with_projection_loss_mask(
    inputs: SmokeInputs,
    *,
    config: AlternatingSmokeConfig,
) -> SmokeInputs:
    if not config.projection_loss_mode.startswith("otsu_"):
        return inputs
    otsu = _otsu_projection_mask(np.asarray(inputs.observed_projections))
    combined = np.asarray(inputs.mask, dtype=bool) & otsu
    return SmokeInputs(
        truth_volume=inputs.truth_volume,
        observed_projections=inputs.observed_projections,
        mask=jnp.asarray(combined, dtype=jnp.float32),
        true_geometry=inputs.true_geometry,
        initial_geometry=inputs.initial_geometry,
    )


def _otsu_projection_mask(projections: np.ndarray) -> np.ndarray:
    views = np.asarray(projections, dtype=np.float32)
    masks: list[np.ndarray] = []
    for view in views:
        threshold = _otsu_threshold(view)
        masks.append(view >= threshold)
    return np.stack(masks)


def _otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(np.asarray(values, dtype=np.float32).reshape(-1), bins=256)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) * 0.5
    weighted = hist.astype(np.float64) * centers.astype(np.float64)
    weight_bg = np.cumsum(hist, dtype=np.float64)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(weighted, dtype=np.float64) / np.maximum(weight_bg, 1.0)
    mean_fg = np.cumsum(weighted[::-1], dtype=np.float64)[::-1] / np.maximum(weight_fg, 1.0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def _validate_sidecar_shapes(
    *,
    truth: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    expected_size: int,
    expected_views: int,
) -> None:
    expected_volume_shape = (expected_size, expected_size, expected_size)
    if tuple(int(dim) for dim in truth.shape) != expected_volume_shape:
        raise ValueError(
            "synthetic sidecar volume shape "
            f"{tuple(truth.shape)!r} does not match configured size {expected_size}"
        )
    if len(observed.shape) != 3:
        raise ValueError("synthetic sidecar projections must have shape (views, rows, cols)")
    if int(observed.shape[0]) != int(expected_views):
        raise ValueError(
            f"synthetic sidecar view count {int(observed.shape[0])} "
            f"does not match configured views {expected_views}"
        )
    if tuple(mask.shape) != tuple(observed.shape):
        raise ValueError("synthetic sidecar mask shape must match projections")


def _synthetic_initial_geometry(n_views: int) -> GeometryState:
    base = _geometry_with_active_det_v(n_views)
    return GeometryState(
        setup=base.setup,
        pose=base.pose,
        acquisition=base.acquisition,
    )


def _synthetic_true_geometry(n_views: int) -> GeometryState:
    base = _geometry_with_active_det_v(n_views)
    span = np.linspace(-1.0, 1.0, num=n_views, dtype=np.float64)
    setup = base.setup.replace_parameter(
        "theta_offset_rad",
        base.setup.theta_offset_rad.with_value(0.035),
    )
    setup = setup.replace_parameter("det_u_px", setup.det_u_px.with_value(0.045))
    setup = setup.replace_parameter("det_v_px", setup.det_v_px.with_value(-0.03))
    return GeometryState(
        setup=setup,
        pose=base.pose.with_updates(
            phi_residual_rad=0.1 * span,
            dx_px=0.02 + 0.01 * span,
            dz_px=-0.0125 + 0.0075 * span,
        ),
        acquisition=base.acquisition,
    )


def _geometry_with_active_det_v(n_views: int) -> GeometryState:
    base = GeometryState.zeros(n_views)
    setup = base.setup.replace_parameter(
        "det_v_px",
        replace(base.setup.det_v_px, value=0.0, active=True),
    )
    return GeometryState(setup=setup, pose=base.pose, acquisition=base.acquisition)
