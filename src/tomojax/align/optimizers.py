from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .motion_models import (
    PoseMotionModel,
    expand_motion_coefficients,
    fit_motion_coefficients,
)


type OptimizerStatValue = float | int | bool | str | None
type OptimizerStats = dict[str, OptimizerStatValue]


@dataclass(frozen=True)
class PoseLbfgsConfig:
    maxiter: int
    ftol: float
    gtol: float
    maxls: int
    memory_size: int = 10


@dataclass(frozen=True)
class PoseLbfgsResult:
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    loss: float | None
    accepted: bool
    stats: OptimizerStats


@dataclass(frozen=True)
class BoundTransform:
    """Differentiable map between unconstrained variables and box-bounded values."""

    lower: jnp.ndarray
    upper: jnp.ndarray
    value_shape: tuple[int, ...]

    @classmethod
    def from_bounds(
        cls,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
        *,
        value_shape: Sequence[int],
    ) -> BoundTransform:
        lower_flat = jnp.asarray(lower, dtype=jnp.float32).reshape(-1)
        upper_flat = jnp.asarray(upper, dtype=jnp.float32).reshape(-1)
        if lower_flat.shape != upper_flat.shape:
            raise ValueError(
                "BoundTransform lower/upper shape mismatch: "
                f"{tuple(lower_flat.shape)} != {tuple(upper_flat.shape)}"
            )
        expected = int(np.prod(tuple(int(v) for v in value_shape)))
        if int(lower_flat.size) != expected:
            raise ValueError(
                "BoundTransform bounds size does not match value_shape: "
                f"{int(lower_flat.size)} != {expected}"
            )
        return cls(
            lower=lower_flat,
            upper=upper_flat,
            value_shape=tuple(int(v) for v in value_shape),
        )

    @property
    def has_finite_bounds(self) -> bool:
        lower_np = np.asarray(self.lower)
        upper_np = np.asarray(self.upper)
        return bool(np.any(np.isfinite(lower_np)) or np.any(np.isfinite(upper_np)))

    def to_unconstrained(self, values: jnp.ndarray) -> jnp.ndarray:
        values_flat = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
        lower = self.lower
        upper = self.upper
        finite_lower = jnp.isfinite(lower)
        finite_upper = jnp.isfinite(upper)
        safe_lower = jnp.where(finite_lower, lower, 0.0)
        safe_upper = jnp.where(finite_upper, upper, 0.0)
        width = safe_upper - safe_lower
        has_box = finite_lower & finite_upper & (width > 0.0)
        fixed_box = finite_lower & finite_upper & (width <= 0.0)
        safe_width = jnp.where(width > 0.0, width, 1.0)
        unit = jnp.clip(
            (values_flat - safe_lower) / safe_width,
            jnp.asarray(1e-6, dtype=jnp.float32),
            jnp.asarray(1.0 - 1e-6, dtype=jnp.float32),
        )
        box_z = jnp.log(unit) - jnp.log1p(-unit)
        lower_z = _softplus_inverse(values_flat - safe_lower)
        upper_z = _softplus_inverse(safe_upper - values_flat)
        z = values_flat
        z = jnp.where(finite_lower & ~finite_upper, lower_z, z)
        z = jnp.where(~finite_lower & finite_upper, upper_z, z)
        z = jnp.where(has_box, box_z, z)
        z = jnp.where(fixed_box, 0.0, z)
        return z.reshape(self.value_shape)

    def from_unconstrained(self, z: jnp.ndarray) -> jnp.ndarray:
        z_flat = jnp.asarray(z, dtype=jnp.float32).reshape(-1)
        lower = self.lower
        upper = self.upper
        finite_lower = jnp.isfinite(lower)
        finite_upper = jnp.isfinite(upper)
        safe_lower = jnp.where(finite_lower, lower, 0.0)
        safe_upper = jnp.where(finite_upper, upper, 0.0)
        width = safe_upper - safe_lower
        has_box = finite_lower & finite_upper & (width > 0.0)
        fixed_box = finite_lower & finite_upper & (width <= 0.0)
        box_value = safe_lower + width * jax.nn.sigmoid(z_flat)
        lower_value = safe_lower + jax.nn.softplus(z_flat)
        upper_value = safe_upper - jax.nn.softplus(z_flat)
        values = z_flat
        values = jnp.where(finite_lower & ~finite_upper, lower_value, values)
        values = jnp.where(~finite_lower & finite_upper, upper_value, values)
        values = jnp.where(has_box, box_value, values)
        values = jnp.where(fixed_box, safe_lower, values)
        return values.reshape(self.value_shape)


def _softplus_inverse(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.maximum(y, jnp.asarray(1e-6, dtype=jnp.float32))
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))


def run_pose_lbfgs(
    *,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    frozen_params5: jnp.ndarray,
    active_cols: np.ndarray,
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
    loss_before_value: float | None,
    objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eval_loss_fn: Callable[[jnp.ndarray, str], float | None],
    apply_param_constraints: Callable[[jnp.ndarray], jnp.ndarray],
    is_expected_failure: Callable[[Exception], bool],
    cfg: PoseLbfgsConfig,
    motion_model: PoseMotionModel | None = None,
) -> PoseLbfgsResult:
    """Run Optax L-BFGS on active alignment variables only."""

    best_value = math.inf
    best_z: jnp.ndarray | None = None
    eval_count = 0
    total_line_search_steps = 0

    active_cols = np.asarray(active_cols, dtype=np.int32)
    if active_cols.size == 0:
        message = "L-BFGS has no active alignment DOFs"
        return PoseLbfgsResult(
            params5=params5_in,
            motion_coeffs=motion_coeffs_in,
            loss=loss_before_value,
            accepted=False,
            stats=_stats_with_aliases(
                {
                    "lbfgs_backend": "optax",
                    "lbfgs_success": False,
                    "lbfgs_accepted": False,
                    "lbfgs_fallback_to_gd": False,
                    "lbfgs_message": message,
                    "lbfgs_failure_reason": message,
                    "lbfgs_initial_loss": loss_before_value,
                    "lbfgs_final_loss": loss_before_value,
                    "lbfgs_best_loss": None,
                }
            ),
        )

    active_cols_jnp = jnp.asarray(active_cols, dtype=jnp.int32)
    active_shape = (int(params5_in.shape[0]), int(active_cols.size))
    lower_active = jnp.tile(bounds_lower[active_cols_jnp], (active_shape[0], 1)).reshape(-1)
    upper_active = jnp.tile(bounds_upper[active_cols_jnp], (active_shape[0], 1)).reshape(-1)
    bounds_transform = BoundTransform.from_bounds(
        lower_active,
        upper_active,
        value_shape=active_shape,
    )
    smooth_unbounded_coefficients = motion_model is not None and not bounds_transform.has_finite_bounds

    def _record_value(value: float, z: jnp.ndarray) -> None:
        nonlocal best_value, best_z, eval_count
        eval_count += 1
        value_f = float(value)
        if math.isfinite(value_f) and value_f < best_value:
            best_value = value_f
            best_z = jnp.asarray(z, dtype=jnp.float32)

    def _failure_result(message: str, initial_loss: float | None = None) -> PoseLbfgsResult:
        stats = _stats_with_aliases(
            {
                "lbfgs_backend": "optax",
                "lbfgs_success": False,
                "lbfgs_accepted": False,
                "lbfgs_fallback_to_gd": True,
                "lbfgs_message": message,
                "lbfgs_nfev": int(eval_count),
                "lbfgs_initial_loss": initial_loss,
                "lbfgs_final_loss": loss_before_value,
                "lbfgs_best_loss": best_value if math.isfinite(best_value) else None,
            }
        )
        return PoseLbfgsResult(
            params5=params5_in,
            motion_coeffs=motion_coeffs_in,
            loss=loss_before_value,
            accepted=False,
            stats=stats,
        )

    if smooth_unbounded_coefficients:
        if motion_coeffs_in is None or motion_model is None:
            return _failure_result("L-BFGS requires initialized smooth pose coefficients")
        z = jnp.asarray(motion_coeffs_in, dtype=jnp.float32)

        def _params_from_z(z_candidate: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            coeffs = jnp.asarray(z_candidate, dtype=jnp.float32)
            params = apply_param_constraints(expand_motion_coefficients(motion_model, coeffs))
            coeffs = fit_motion_coefficients(motion_model, params)
            params = apply_param_constraints(expand_motion_coefficients(motion_model, coeffs))
            coeffs = fit_motion_coefficients(motion_model, params)
            params = apply_param_constraints(expand_motion_coefficients(motion_model, coeffs))
            return params, coeffs

    else:
        z = bounds_transform.to_unconstrained(params5_in[:, active_cols_jnp])

        def _params_from_z(z_candidate: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            active_values = bounds_transform.from_unconstrained(z_candidate)
            candidate = (
                jnp.asarray(frozen_params5, dtype=jnp.float32)
                .at[:, active_cols_jnp]
                .set(active_values)
            )
            candidate = apply_param_constraints(candidate)
            if motion_model is None:
                return candidate, motion_coeffs_in
            coeffs = fit_motion_coefficients(motion_model, candidate)
            candidate = apply_param_constraints(expand_motion_coefficients(motion_model, coeffs))
            coeffs = fit_motion_coefficients(motion_model, candidate)
            candidate = apply_param_constraints(expand_motion_coefficients(motion_model, coeffs))
            return candidate, coeffs

    def _objective(z_candidate: jnp.ndarray) -> jnp.ndarray:
        candidate, _ = _params_from_z(z_candidate)
        value = objective_fn(candidate)
        return jnp.where(jnp.isfinite(value), value, jnp.asarray(1e30, dtype=value.dtype))

    objective_jit = jax.jit(_objective)
    solver = optax.lbfgs(
        memory_size=int(cfg.memory_size),
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(cfg.maxls),
            approx_dec_rtol=float(cfg.ftol),
        ),
    )
    opt_state = solver.init(z)
    value_and_grad = optax.value_and_grad_from_state(objective_jit)

    try:
        initial_value_jax, initial_grad = value_and_grad(z, state=opt_state)
        initial_value = float(initial_value_jax)
        initial_grad_norm = float(jnp.linalg.norm(initial_grad))
        _record_value(initial_value, z)
    except Exception as exc:
        if is_expected_failure(exc):
            return _failure_result(f"initial L-BFGS objective/gradient failed: {exc}")
        raise

    if not math.isfinite(initial_value) or not math.isfinite(initial_grad_norm):
        return _failure_result(
            "initial L-BFGS objective/gradient is non-finite",
            initial_loss=initial_value if math.isfinite(initial_value) else None,
        )

    success = initial_grad_norm <= float(cfg.gtol)
    result_message = (
        f"Optax L-BFGS initial gradient norm {initial_grad_norm:.3e} <= gtol"
        if success
        else "Optax L-BFGS reached maxiter"
    )
    nit = 0
    last_grad_norm = initial_grad_norm
    last_step_norm = 0.0

    try:
        for iteration in range(int(cfg.maxiter)):
            if success:
                break
            value, grad = value_and_grad(z, state=opt_state)
            value_f = float(value)
            grad_norm = float(jnp.linalg.norm(grad))
            if not math.isfinite(value_f) or not math.isfinite(grad_norm):
                return _failure_result(
                    "non-finite L-BFGS objective/gradient",
                    initial_loss=initial_value,
                )
            _record_value(value_f, z)
            if grad_norm <= float(cfg.gtol):
                success = True
                result_message = f"Optax L-BFGS gradient norm {grad_norm:.3e} <= gtol"
                last_grad_norm = grad_norm
                break

            updates, opt_state = solver.update(
                grad,
                opt_state,
                z,
                value=value,
                grad=grad,
                value_fn=objective_jit,
            )
            z_next = optax.apply_updates(z, updates)
            if not bool(jnp.all(jnp.isfinite(z_next))):
                return _failure_result(
                    "non-finite L-BFGS parameter update",
                    initial_loss=initial_value,
                )

            nit = iteration + 1
            last_grad_norm = grad_norm
            last_step_norm = float(jnp.linalg.norm(z_next - z))
            line_search_state = opt_state[-1] if isinstance(opt_state, tuple) else None
            line_search_info = getattr(line_search_state, "info", None)
            line_search_steps = getattr(line_search_info, "num_linesearch_steps", None)
            if line_search_steps is not None:
                total_line_search_steps += int(line_search_steps)
            line_search_value = getattr(line_search_state, "value", None)
            next_value_for_rel_change: float | None = None
            if line_search_value is not None:
                line_value_f = float(line_search_value)
                if math.isfinite(line_value_f):
                    _record_value(line_value_f, z_next)
                    next_value_for_rel_change = line_value_f

            z = z_next
            if next_value_for_rel_change is not None:
                rel_change = abs(value_f - next_value_for_rel_change) / max(1.0, abs(value_f))
                if rel_change <= float(cfg.ftol):
                    success = True
                    result_message = (
                        f"Optax L-BFGS relative objective change {rel_change:.3e} <= ftol"
                    )
    except Exception as exc:
        if is_expected_failure(exc):
            return _failure_result(
                f"L-BFGS failed during optimization: {exc}",
                initial_loss=initial_value,
            )
        raise

    def _evaluate_candidate(
        label: str,
        z_candidate: jnp.ndarray,
    ) -> tuple[str, float, jnp.ndarray, jnp.ndarray | None]:
        candidate_params, candidate_coeffs = _params_from_z(z_candidate)
        candidate_eval = eval_loss_fn(candidate_params, label)
        candidate_loss = float(candidate_eval) if candidate_eval is not None else math.inf
        return label, candidate_loss, candidate_params, candidate_coeffs

    candidates = [_evaluate_candidate("last", z)]
    if best_z is not None:
        candidates.append(_evaluate_candidate("best", best_z))
    selected_label, final_loss, candidate_params, candidate_coeffs = min(
        candidates,
        key=lambda item: item[1] if math.isfinite(item[1]) else math.inf,
    )
    if math.isfinite(final_loss):
        selected_z = z if selected_label == "last" or best_z is None else best_z
        _record_value(final_loss, selected_z)

    baseline = float(loss_before_value) if loss_before_value is not None else initial_value
    accepted = math.isfinite(final_loss) and final_loss < baseline
    if not accepted:
        candidate_params = params5_in
        candidate_coeffs = motion_coeffs_in
        final_loss = loss_before_value if loss_before_value is not None else math.nan
        if not result_message:
            result_message = "L-BFGS candidate did not reduce the objective"
        selected_label = "rejected"

    try:
        dp = candidate_params - params5_in
        rot_mean = float(jnp.mean(jnp.abs(dp[:, :3])))
        trans_mean = float(jnp.mean(jnp.abs(dp[:, 3:])))
    except Exception:
        rot_mean = None
        trans_mean = None

    stats: OptimizerStats = {
        "lbfgs_backend": "optax",
        "lbfgs_success": bool(success),
        "lbfgs_accepted": bool(accepted),
        "lbfgs_fallback_to_gd": False,
        "lbfgs_status": 0 if success else 1,
        "lbfgs_message": result_message,
        "lbfgs_nit": int(nit),
        "lbfgs_nfev": int(eval_count),
        "lbfgs_objective_evals": int(eval_count),
        "lbfgs_line_search_steps": int(total_line_search_steps),
        "lbfgs_initial_loss": float(initial_value),
        "lbfgs_final_loss": float(final_loss) if math.isfinite(final_loss) else None,
        "lbfgs_best_loss": best_value if math.isfinite(best_value) else None,
        "lbfgs_selected_candidate": selected_label,
        "lbfgs_initial_grad_norm": float(initial_grad_norm),
        "lbfgs_final_grad_norm": float(last_grad_norm),
        "lbfgs_last_step_norm": float(last_step_norm),
    }
    if rot_mean is not None:
        stats["rot_mean"] = rot_mean
    if trans_mean is not None:
        stats["trans_mean"] = trans_mean
    if (not accepted) and result_message:
        stats["lbfgs_failure_reason"] = result_message
    return PoseLbfgsResult(
        params5=candidate_params,
        motion_coeffs=candidate_coeffs,
        loss=final_loss,
        accepted=bool(accepted),
        stats=_stats_with_aliases(stats),
    )


def _stats_with_aliases(stats: OptimizerStats) -> OptimizerStats:
    out: dict[str, Any] = dict(stats)
    out.setdefault("optimizer", "lbfgs")
    out.setdefault("optimizer_backend", out.get("lbfgs_backend", "optax"))
    out.setdefault("optimizer_accepted", out.get("lbfgs_accepted"))
    out.setdefault("optimizer_initial_loss", out.get("lbfgs_initial_loss"))
    out.setdefault("optimizer_final_loss", out.get("lbfgs_final_loss"))
    out.setdefault("optimizer_best_loss", out.get("lbfgs_best_loss"))
    return out
