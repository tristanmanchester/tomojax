from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager

from tomojax._typed_arrays import object_list, object_mapping

LOG = logging.getLogger(__name__)
_DISABLED_TRANSFER_GUARD_MODES = frozenset(("off", "none", "disable", "disabled"))


def _warn_transfer_guard_demotion(mode: str, reason: str) -> None:
    LOG.warning(
        "Requested transfer guard mode %r, but support is unavailable (%s); falling back to no-op.",
        mode,
        reason,
    )


def transfer_guard_context(mode: str | None = None) -> AbstractContextManager[object]:
    """Return the configured JAX transfer-guard context, if available."""
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log")
    mode = str(mode).lower()
    if mode in _DISABLED_TRANSFER_GUARD_MODES:
        return nullcontext()
    try:
        import jax

        return jax.transfer_guard(mode)
    except (ImportError, AttributeError, ValueError, RuntimeError) as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        return nullcontext()


def beyond_detector(info: Mapping[str, object]) -> str | None:
    """Say what share of a scan's attenuation lies beyond the detector, from 1% up.

    ``info`` is a reconstruction's record or its summary; a scan that ends
    inside the detector has nothing to say.
    """
    fraction = object_mapping(info.get("truncation")).get("outside_fraction")
    if isinstance(fraction, float) and fraction >= 0.005:
        return f"{fraction:.0%} of attenuation beyond the detector"
    return None


def report_automatic(info: Mapping[str, object]) -> None:
    """Keep a successful but limited automatic result visible without a manifest."""
    automatic = object_mapping(info.get("automatic"))
    if automatic.get("status") == "limited":
        reasons = object_list(automatic.get("reasons"))
        LOG.warning("Automatic workflow is limited: %s", "; ".join(map(str, reasons)))


_STOPS = {
    "projection_discrepancy": "stopped at the estimated projection discrepancy",
    "noise_score": "stopped at the estimated noise level",
    "objective_tolerance": "stopped at the objective tolerance",
    "converged": "converged",
    "roundoff_limit": "stopped at the float32 roundoff limit",
    "numerical_breakdown": "stopped at a numerical breakdown",
    "iteration_limit": "ran to the iteration limit",
}


def solve_summary(info: Mapping[str, object]) -> str | None:
    """How an iterative solve ended and the TV weight it chose, or None for a direct method."""
    count, reason = info.get("effective_iterations"), info.get("termination")
    if not isinstance(count, int) or not isinstance(reason, str):
        return None
    text = f"{count} updates, {_STOPS.get(reason, reason.replace('_', ' '))}"
    weight = info.get("resolved_tv_weight")
    if info.get("tv_weight") is None and isinstance(weight, float):
        text += f"; automatic TV weight {weight:.3g}"
    return text
