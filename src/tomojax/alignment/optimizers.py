"""Validation Levenberg-Marquardt steps for setup geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ._model.dof_specs import ActiveParameterView, optimizer_step_stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._model.dofs import DofBounds
    from ._model.state import AlignmentState


@dataclass(frozen=True)
class ActiveOptimizerResult:
    """Result from an active-state optimizer step."""

    state: AlignmentState
    loss: float
    accepted: bool
    stats: dict[str, object]


@dataclass(frozen=True)
class ValidationLmConfig:
    """Configuration for validation-LM active-state optimization."""

    damping: float = 1e-3
    min_damping: float = 1e-8
    step_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)


def run_active_validation_lm(  # noqa: PLR0915
    *,
    state: AlignmentState,
    view: ActiveParameterView,
    loss: float,
    grad: jnp.ndarray,
    hess: jnp.ndarray,
    score_fn: Callable[[jnp.ndarray], float],
    bounds: DofBounds | None = None,
    cfg: ValidationLmConfig | None = None,
) -> ActiveOptimizerResult:
    """Run one damped validation-LM step over the active whitened state."""
    cfg = ValidationLmConfig() if cfg is None else cfg
    z0 = jnp.asarray(view.pack(state), dtype=jnp.float32).reshape(-1)
    lower, upper = view.bounds_whitened(state, bounds=bounds)
    lower = jnp.asarray(lower, dtype=jnp.float32).reshape(z0.shape)
    upper = jnp.asarray(upper, dtype=jnp.float32).reshape(z0.shape)
    g = jnp.asarray(grad, dtype=jnp.float32).reshape(z0.shape)
    H = jnp.asarray(hess, dtype=jnp.float32).reshape((int(z0.size), int(z0.size)))
    H = jnp.float32(0.5) * (H + H.T)
    damping = max(float(cfg.damping), float(cfg.min_damping))
    diag = jnp.maximum(jnp.diag(H), jnp.asarray(1e-6, dtype=jnp.float32))
    H_lm = H + jnp.asarray(damping, dtype=jnp.float32) * jnp.diag(diag)

    try:
        dz = -jnp.linalg.solve(H_lm, g)
        solve_method = "solve"
    except Exception:
        dz = -(jnp.linalg.pinv(H_lm) @ g)
        solve_method = "pinv"

    try:
        singular_values = jnp.linalg.svd(H, compute_uv=False)
        sv_np = np.asarray(singular_values, dtype=np.float64)
        finite_sv = sv_np[np.isfinite(sv_np)]
        if finite_sv.size and float(np.min(np.abs(finite_sv))) > 0.0:
            condition = float(np.max(np.abs(finite_sv)) / np.min(np.abs(finite_sv)))
        else:
            condition = math.inf
        sv_list = [float(v) for v in sv_np]
    except Exception:
        condition = math.inf
        sv_list = []

    initial_loss = float(loss)
    best_loss = initial_loss
    best_z = z0
    best_scale = 0.0
    candidate_losses: list[float | None] = []
    predicted_reductions: list[float] = []
    for raw_scale in tuple(float(v) for v in cfg.step_scales):
        scale = float(raw_scale)
        candidate_z = jnp.clip(z0 + jnp.asarray(scale, dtype=jnp.float32) * dz, lower, upper)
        step = candidate_z - z0
        predicted = -float(jnp.vdot(g, step).real + 0.5 * jnp.vdot(step, H @ step).real)
        predicted_reductions.append(predicted)
        try:
            candidate_loss = float(score_fn(candidate_z))
        except Exception:
            candidate_loss = math.inf
        candidate_losses.append(candidate_loss if math.isfinite(candidate_loss) else None)
        if math.isfinite(candidate_loss) and candidate_loss < best_loss:
            best_loss = candidate_loss
            best_z = candidate_z
            best_scale = scale

    accepted = math.isfinite(best_loss) and best_loss < initial_loss
    final_state = view.unpack(state, best_z) if accepted else state
    final_loss = best_loss if accepted else initial_loss
    actual_reduction = initial_loss - final_loss
    predicted_reduction = (
        predicted_reductions[tuple(float(v) for v in cfg.step_scales).index(best_scale)]
        if best_scale in tuple(float(v) for v in cfg.step_scales)
        else 0.0
    )
    stats = {
        "validation_lm_backend": "streamed_normals",
        "validation_lm_accepted": bool(accepted),
        "validation_lm_success": bool(accepted),
        "validation_lm_initial_loss": initial_loss,
        "validation_lm_final_loss": float(final_loss),
        "validation_lm_best_loss": float(best_loss),
        "validation_lm_damping": float(damping),
        "validation_lm_solve_method": solve_method,
        "validation_lm_candidate_scales": [float(v) for v in cfg.step_scales],
        "validation_lm_candidate_losses": candidate_losses,
        "validation_lm_predicted_reductions": predicted_reductions,
        "validation_lm_selected_scale": float(best_scale),
        "validation_lm_actual_reduction": float(actual_reduction),
        "validation_lm_predicted_reduction": float(predicted_reduction),
        "validation_lm_ratio": (
            float(actual_reduction / predicted_reduction)
            if abs(predicted_reduction) > 1e-12
            else None
        ),
        "validation_lm_condition_number": condition,
        "validation_lm_singular_values": sv_list,
        **optimizer_step_stats(view=view, before=state, after=final_state, grad_whitened=g),
    }
    if not accepted:
        stats["validation_lm_failure_reason"] = "no candidate validation-LM step reduced loss"
    return ActiveOptimizerResult(
        state=final_state,
        loss=float(final_loss),
        accepted=bool(accepted),
        stats=stats,
    )
