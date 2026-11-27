"""Stopping policy shared by resident, streamed and native CGLS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

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
