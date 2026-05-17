from __future__ import annotations

from collections.abc import Callable
import logging
import re

import jax
import jax.numpy as jnp

from ._objectives.loss_specs import loss_is_within_relative_tolerance


def _should_prefer_gn_candidate(
    loss_before: float,
    current_loss: float,
    candidate_loss: float,
    rel_tol: float,
) -> bool:
    """Accept tolerated GN candidates only when they improve the current best step."""
    candidate_ok = candidate_loss < loss_before or loss_is_within_relative_tolerance(
        loss_before, candidate_loss, rel_tol
    )
    return candidate_ok and candidate_loss < current_loss


def _second_difference_gram(n: int) -> jnp.ndarray:
    if n < 3:
        return jnp.zeros((n, n), dtype=jnp.float32)
    d2 = jnp.zeros((n - 2, n), dtype=jnp.float32)
    rows = jnp.arange(n - 2, dtype=jnp.int32)
    d2 = d2.at[rows, rows].set(1.0)
    d2 = d2.at[rows, rows + 1].set(-2.0)
    d2 = d2.at[rows, rows + 2].set(1.0)
    return d2.T @ d2


def _smooth_gn_candidate(
    params5: jnp.ndarray,
    smoothness_gram: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Project a per-view GN candidate through the quadratic curvature prior."""
    n_views = int(params5.shape[0])
    if n_views < 3:
        return params5

    eye = jnp.eye(n_views, dtype=jnp.float32)

    def solve_one_dim(rhs: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.cond(
            weight > 0.0,
            lambda _: jnp.linalg.solve(eye + 2.0 * weight * smoothness_gram, rhs),
            lambda _: rhs,
            operand=None,
        )

    return jax.vmap(solve_one_dim, in_axes=(1, 0), out_axes=1)(params5, weights)


def _select_gn_candidate(
    params5_prev: jnp.ndarray,
    dp_all: jnp.ndarray,
    *,
    loss_before: float,
    eval_loss: Callable[[jnp.ndarray], float],
    gn_accept_tol: float,
    constrain_candidate: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    smooth_candidate: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    light_smoothness_weights_sq: jnp.ndarray | None = None,
    medium_smoothness_weights_sq: jnp.ndarray | None = None,
    smoothness_weights_sq: jnp.ndarray | None = None,
    trans_only_smoothness_weights_sq: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float]:
    """Pick a GN candidate using a small hierarchical full-loss search."""

    def _accepts(candidate_loss: float, current_best_loss: float = float("inf")) -> bool:
        return _should_prefer_gn_candidate(
            loss_before,
            current_best_loss,
            candidate_loss,
            gn_accept_tol,
        )

    def _constrain(candidate: jnp.ndarray) -> jnp.ndarray:
        if constrain_candidate is None:
            return candidate
        return constrain_candidate(candidate)

    raw_params = _constrain(params5_prev + dp_all)
    raw_loss = eval_loss(raw_params)
    if _accepts(raw_loss):
        return raw_params, raw_loss

    half_params = _constrain(params5_prev + jnp.float32(0.5) * dp_all)
    half_loss = eval_loss(half_params)
    if _accepts(half_loss):
        return half_params, half_loss

    base_params = raw_params if raw_loss <= half_loss else half_params

    def _has_active_weights(weights: jnp.ndarray | None) -> bool:
        return weights is not None and bool(jnp.any(weights > 0.0))

    if smooth_candidate is None:
        return params5_prev, loss_before

    smooth_weights = [
        weights
        for weights in (
            light_smoothness_weights_sq,
            medium_smoothness_weights_sq,
            smoothness_weights_sq,
            trans_only_smoothness_weights_sq,
        )
        if _has_active_weights(weights)
    ]

    if not smooth_weights:
        return params5_prev, loss_before

    best_params = params5_prev
    best_loss = float("inf")
    accepted = False
    for weights in smooth_weights:
        candidate_params = _constrain(smooth_candidate(base_params, weights))
        candidate_loss = eval_loss(candidate_params)
        if _accepts(candidate_loss, best_loss):
            best_params = candidate_params
            best_loss = candidate_loss
            accepted = True

    if accepted:
        return best_params, best_loss
    return params5_prev, loss_before


_EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS = (
    "allocator",
    "cholesky",
    "failed to converge",
    "non-finite",
    "nonfinite",
    "not positive definite",
    "out of memory",
    "resource_exhausted",
    "singular",
    "svd",
)


_EXPECTED_ALIGN_EVAL_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_])(?:[+-]?(?:inf|nan)s?|infinite|infinity)(?![a-z0-9_])"
)


def _is_expected_align_eval_failure(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    msg = str(exc).lower()
    return bool(_EXPECTED_ALIGN_EVAL_NUMERIC_TOKEN_RE.search(msg)) or any(
        snippet in msg for snippet in _EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS
    )


def _evaluate_align_loss(
    eval_loss: Callable[[], float | jnp.ndarray],
    *,
    fallback: float | None,
    context: str,
) -> float | None:
    try:
        return float(eval_loss())
    except Exception as exc:
        if _is_expected_align_eval_failure(exc):
            logging.warning("%s after expected numeric failure: %s", context, exc)
            return fallback
        raise
