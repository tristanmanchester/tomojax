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
from .dof_specs import ActiveParameterView, optimizer_step_stats
from .dofs import DofBounds
from .state import AlignmentState


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
class PoseOptimizationContext:
    """Static transform and bounds context for pose L-BFGS variables."""

    active_cols: np.ndarray
    frozen_params5: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    apply_param_constraints: Callable[[jnp.ndarray], jnp.ndarray]
    motion_model: PoseMotionModel | None = None

    @property
    def active_cols_np(self) -> np.ndarray:
        return np.asarray(self.active_cols, dtype=np.int32)

    def active_bound_transform(self, n_views: int) -> tuple[jnp.ndarray, BoundTransform]:
        active_cols_jnp = jnp.asarray(self.active_cols_np, dtype=jnp.int32)
        active_shape = (int(n_views), int(active_cols_jnp.size))
        lower_active = jnp.tile(self.bounds_lower[active_cols_jnp], (active_shape[0], 1)).reshape(-1)
        upper_active = jnp.tile(self.bounds_upper[active_cols_jnp], (active_shape[0], 1)).reshape(-1)
        return active_cols_jnp, BoundTransform.from_bounds(
            lower_active,
            upper_active,
            value_shape=active_shape,
        )


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
    loss_before_value: float | None,
    objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eval_loss_fn: Callable[[jnp.ndarray, str], float | None],
    is_expected_failure: Callable[[Exception], bool],
    cfg: PoseLbfgsConfig,
    context: PoseOptimizationContext,
) -> PoseLbfgsResult:
    """Run Optax L-BFGS on active alignment variables only."""

    best_value = math.inf
    best_z: jnp.ndarray | None = None
    eval_count = 0
    total_line_search_steps = 0

    active_cols = context.active_cols_np
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

    active_cols_jnp, bounds_transform = context.active_bound_transform(int(params5_in.shape[0]))
    motion_model = context.motion_model
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
            params = context.apply_param_constraints(
                expand_motion_coefficients(motion_model, coeffs)
            )
            coeffs = fit_motion_coefficients(motion_model, params)
            params = context.apply_param_constraints(
                expand_motion_coefficients(motion_model, coeffs)
            )
            coeffs = fit_motion_coefficients(motion_model, params)
            params = context.apply_param_constraints(
                expand_motion_coefficients(motion_model, coeffs)
            )
            return params, coeffs

    else:
        z = bounds_transform.to_unconstrained(params5_in[:, active_cols_jnp])

        def _params_from_z(z_candidate: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            active_values = bounds_transform.from_unconstrained(z_candidate)
            candidate = (
                jnp.asarray(context.frozen_params5, dtype=jnp.float32)
                .at[:, active_cols_jnp]
                .set(active_values)
            )
            candidate = context.apply_param_constraints(candidate)
            if motion_model is None:
                return candidate, motion_coeffs_in
            coeffs = fit_motion_coefficients(motion_model, candidate)
            candidate = context.apply_param_constraints(
                expand_motion_coefficients(motion_model, coeffs)
            )
            coeffs = fit_motion_coefficients(motion_model, candidate)
            candidate = context.apply_param_constraints(
                expand_motion_coefficients(motion_model, coeffs)
            )
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


@dataclass(frozen=True)
class ActiveLbfgsConfig:
    maxiter: int = 12
    ftol: float = 1e-6
    gtol: float = 1e-5
    maxls: int = 20
    memory_size: int = 10


@dataclass(frozen=True)
class ActiveOptimizerResult:
    state: AlignmentState
    loss: float
    accepted: bool
    stats: dict[str, object]


@dataclass(frozen=True)
class ValidationLmConfig:
    damping: float = 1e-3
    min_damping: float = 1e-8
    step_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)


def run_active_validation_lm(
    *,
    state: AlignmentState,
    view: ActiveParameterView,
    loss: float,
    grad: jnp.ndarray,
    hess: jnp.ndarray,
    score_fn: Callable[[jnp.ndarray], float],
    bounds: DofBounds | None = None,
    cfg: ValidationLmConfig = ValidationLmConfig(),
) -> ActiveOptimizerResult:
    """Run one damped validation-LM step over the active whitened state."""
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
        "optimizer": "validation_lm",
        "optimizer_backend": "streamed_normals",
        "optimizer_accepted": bool(accepted),
        "optimizer_success": bool(accepted),
        "optimizer_initial_loss": initial_loss,
        "optimizer_final_loss": float(final_loss),
        "optimizer_best_loss": float(best_loss),
        "optimizer_damping": float(damping),
        "optimizer_solve_method": solve_method,
        "optimizer_candidate_scales": [float(v) for v in cfg.step_scales],
        "optimizer_candidate_losses": candidate_losses,
        "optimizer_predicted_reductions": predicted_reductions,
        "optimizer_selected_scale": float(best_scale),
        "optimizer_actual_reduction": float(actual_reduction),
        "optimizer_predicted_reduction": float(predicted_reduction),
        "optimizer_lm_ratio": (
            float(actual_reduction / predicted_reduction)
            if abs(predicted_reduction) > 1e-12
            else None
        ),
        "optimizer_condition_number": condition,
        "optimizer_singular_values": sv_list,
        **optimizer_step_stats(view=view, before=state, after=final_state, grad_whitened=g),
    }
    if not accepted:
        stats["optimizer_failure_reason"] = "no candidate validation-LM step reduced loss"
    return ActiveOptimizerResult(
        state=final_state,
        loss=float(final_loss),
        accepted=bool(accepted),
        stats=stats,
    )


def run_active_lbfgs(
    *,
    state: AlignmentState,
    view: ActiveParameterView,
    objective_fn: Callable[[AlignmentState], jnp.ndarray],
    objective_value_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    objective_value_and_grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    | None = None,
    cfg: ActiveLbfgsConfig = ActiveLbfgsConfig(),
) -> ActiveOptimizerResult:
    """Run Optax L-BFGS over a unified whitened active-state vector."""
    z0 = view.pack(state)
    lower, upper = view.bounds_whitened(state)
    bounds_transform = BoundTransform.from_bounds(lower, upper, value_shape=tuple(z0.shape))
    u = bounds_transform.to_unconstrained(z0)

    def _state_from_u(u_candidate: jnp.ndarray) -> AlignmentState:
        z_candidate = bounds_transform.from_unconstrained(u_candidate)
        return view.unpack(state, z_candidate)

    def _objective(u_candidate: jnp.ndarray) -> jnp.ndarray:
        z_candidate = bounds_transform.from_unconstrained(u_candidate)
        value = (
            objective_value_fn(z_candidate)
            if objective_value_fn is not None
            else objective_fn(view.unpack(state, z_candidate))
        )
        return jnp.where(jnp.isfinite(value), value, jnp.asarray(1e30, dtype=jnp.float32))

    def _value_and_grad(u_candidate: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if objective_value_and_grad_fn is None:
            return jax.value_and_grad(_objective)(u_candidate)
        z_candidate, z_pullback = jax.vjp(bounds_transform.from_unconstrained, u_candidate)
        value, grad_z = objective_value_and_grad_fn(z_candidate)
        grad_u = z_pullback(grad_z)[0]
        value = jnp.where(jnp.isfinite(value), value, jnp.asarray(1e30, dtype=jnp.float32))
        grad_u = jnp.where(jnp.isfinite(grad_u), grad_u, jnp.zeros_like(grad_u))
        return value, grad_u

    value_and_grad = _value_and_grad
    initial_value, initial_grad = value_and_grad(u)
    initial_loss = float(initial_value)
    initial_grad_norm = float(jnp.linalg.norm(initial_grad))
    if not math.isfinite(initial_loss) or not math.isfinite(initial_grad_norm):
        return ActiveOptimizerResult(
            state=state,
            loss=initial_loss,
            accepted=False,
            stats={
                "optimizer": "lbfgs",
                "optimizer_backend": "optax",
                "optimizer_accepted": False,
                "optimizer_initial_loss": initial_loss if math.isfinite(initial_loss) else None,
                "optimizer_final_loss": None,
                "optimizer_failure_reason": "non-finite initial objective/gradient",
            },
        )

    solver = optax.lbfgs(
        memory_size=int(cfg.memory_size),
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(cfg.maxls),
            approx_dec_rtol=float(cfg.ftol),
        ),
    )
    opt_state = solver.init(u)
    best_u = u
    best_loss = initial_loss
    last_grad_norm = initial_grad_norm
    last_step_norm = 0.0
    nit = 0
    success = initial_grad_norm <= float(cfg.gtol)

    for iteration in range(int(cfg.maxiter)):
        if success:
            break
        value, grad = value_and_grad(u)
        value_f = float(value)
        grad_norm = float(jnp.linalg.norm(grad))
        if math.isfinite(value_f) and value_f < best_loss:
            best_loss = value_f
            best_u = u
        if grad_norm <= float(cfg.gtol):
            success = True
            last_grad_norm = grad_norm
            break
        updates, opt_state = solver.update(
            grad,
            opt_state,
            u,
            value=value,
            grad=grad,
            value_fn=_objective,
        )
        u_next = optax.apply_updates(u, updates)
        if not bool(jnp.all(jnp.isfinite(u_next))):
            break
        last_step_norm = float(jnp.linalg.norm(u_next - u))
        last_grad_norm = grad_norm
        u = u_next
        nit = iteration + 1

    final_state = _state_from_u(best_u)
    final_loss = float(_objective(best_u))
    accepted = math.isfinite(final_loss) and final_loss <= initial_loss
    if not accepted:
        final_state = state
        final_loss = initial_loss
    if objective_value_and_grad_fn is None:
        z_grad = jax.grad(lambda z: objective_fn(view.unpack(state, z)))(view.pack(state))
    else:
        _, z_grad = objective_value_and_grad_fn(view.pack(state))
    stats = {
        "optimizer": "lbfgs",
        "optimizer_backend": "optax",
        "optimizer_accepted": bool(accepted),
        "optimizer_success": bool(success),
        "optimizer_initial_loss": initial_loss,
        "optimizer_final_loss": final_loss,
        "optimizer_best_loss": best_loss,
        "optimizer_nit": int(nit),
        "optimizer_initial_grad_norm": initial_grad_norm,
        "optimizer_final_grad_norm": last_grad_norm,
        "optimizer_last_step_norm": last_step_norm,
        **optimizer_step_stats(view=view, before=state, after=final_state, grad_whitened=z_grad),
    }
    return ActiveOptimizerResult(
        state=final_state,
        loss=final_loss,
        accepted=bool(accepted),
        stats=stats,
    )
