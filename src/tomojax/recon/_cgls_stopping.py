"""Stopping policy shared by resident, streamed and native CGLS."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

if TYPE_CHECKING:
    import jax
    from jax import Array
    import numpy as np

    from tomojax.core.cuda_types import CuPyValue

    from .types import DataTerm

    @overload
    def precision_stalled(
        norm2: float,
        previous_norm2: float,
        previous_verified: bool | np.bool_,
        bounded: Literal[True, False],
    ) -> bool | np.bool_: ...

    @overload
    def precision_stalled(
        norm2: Array, previous_norm2: Array, previous_verified: Array, bounded: Array
    ) -> Array: ...


def precision_stalled(
    norm2: float | Array,
    previous_norm2: float | Array,
    previous_verified: bool | np.bool_ | Array,
    bounded: bool | Array,
) -> bool | np.bool_ | Array:
    """A cancellation bound is not proof that a requested tolerance is unattainable.

    Keep improving a verified normal residual even inside its FP32 bound. Only
    compare two recomputed residuals: a recurrence estimate can drift below the
    measured norm, falsely suggesting that progress has stopped.

    After a near-precision check restarts the direction, check the next small
    residual too. Waiting another period lets an unverified recurrence leave
    the cancellation floor before two measured gradients can be compared.
    """
    return bounded & previous_verified & (norm2 >= previous_norm2)


class CGLSState(NamedTuple):
    iteration: Array
    x: Array
    residual: Array
    direction: Array
    gamma: Array
    failed: Array
    roundoff: Array
    residual_verified: Array
    discrepancy_margin: Array | None
    restarted: Array
    residual_recomputations: Array

    def advanced(
        self,
        x: Array,
        residual: Array,
        gradient: Array,
        gamma: Array,
        margin: Array | None,
        *,
        valid: Array,
        stepped: Array,
        restart: Array,
        verified: Array,
        roundoff: Array,
    ) -> CGLSState:
        """Restart a drifted direction and preserve the last finite norm on breakdown."""
        import jax.numpy as jnp

        direction = jnp.where(restart, gradient, gradient + (gamma / self.gamma) * self.direction)
        return CGLSState(
            self.iteration + stepped.astype(jnp.int32),
            x,
            residual,
            direction,
            jnp.where(valid, gamma, self.gamma),
            ~valid,
            jnp.where(valid, roundoff, self.roundoff),
            jnp.where(valid, verified, self.residual_verified),
            margin,
            restart,
            self.residual_recomputations + verified.astype(jnp.int32),
        )


def running(state: CGLSState, *, max_iters: Array, threshold: Array) -> Array:
    """Whether the CGLS loop continues: budget left, tolerance unmet, no breakdown or roundoff."""
    return (
        (state.iteration < max_iters)
        & (state.gamma > threshold * threshold)
        & ~state.failed
        & ~state.roundoff
        & (True if state.discrepancy_margin is None else state.discrepancy_margin > 0)
    )


def termination(*, failed: bool, converged: bool, roundoff: bool) -> str:
    """The reported reason a CGLS solve stopped."""
    if failed:
        return "numerical_breakdown"
    if converged:
        return "converged"
    return "roundoff_limit" if roundoff else "iteration_limit"


def cancellation_magnitude(
    data: jax.Array | CuPyValue, prediction: jax.Array | CuPyValue, data_term: DataTerm
) -> jax.Array | CuPyValue:
    """Nonnegative projection envelope for the FP32 cancellation bound.

    Contrast uses C = I - P, with P the whole-view mean. Since |C| <= I + P,
    applying |A|ᵀ to this envelope bounds cancellation in Aᵀ C (data - prediction).
    Centering the envelope itself would invalidate the bound.
    """
    size = abs(data) + abs(prediction)
    if data_term == "per_view_contrast":
        size = size + size.mean(axis=(-2, -1), keepdims=True)
    return size


def stopping_record(
    *,
    count: int,
    budget: int,
    gamma: float,
    first_norm: float,
    tolerance: float,
    failed: bool,
    roundoff: bool,
    verified: bool,
    recomputations: int,
    target: float | None,
    margin: float | Array | None,
    samples: int,
) -> dict[str, object]:
    """Numerical convergence, statistical discrepancy and work, in physical units."""
    converged = not failed and gamma <= tolerance**2
    reason = termination(failed=failed, converged=converged, roundoff=roundoff)
    if reason == "iteration_limit" and margin is not None and float(margin) <= 0:
        reason = "projection_discrepancy"
    return {
        "effective_iterations": count,
        "converged": converged,
        "termination": reason,
        "early_stop": count < budget and not failed,
        "normal_residual_norm": math.sqrt(gamma),
        "normal_residual_is_recomputed": verified,
        "residual_recomputations": recomputations,
        "initial_normal_residual_norm": first_norm,
        "normal_residual_tolerance": tolerance,
        "projection_error_rms": None if target is None else (target / samples) ** 0.5,
        "residual_rms": None
        if margin is None or target is None
        else (max(0, target + float(margin)) / samples) ** 0.5,
    }
