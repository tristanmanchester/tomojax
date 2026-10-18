"""Scalar stopping policy shared by resident, streamed and native CGLS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


def precision_stalled(
    norm2: float | Array,
    previous_norm2: float | Array,
    previous_verified: bool | Array,
    bounded: bool | Array,
) -> bool | Array:
    """A cancellation bound is not proof that a requested tolerance is unattainable.

    Keep improving a verified normal residual even inside its FP32 bound. Only
    compare two recomputed residuals: a recurrence estimate can drift below the
    measured norm, falsely suggesting that progress has stopped.

    After a near-precision check continues, the solver checks the next small
    residual too. Waiting another period lets an unverified recurrence leave
    the cancellation floor before two measured gradients can be compared.
    """
    return bounded & previous_verified & (norm2 >= previous_norm2)
