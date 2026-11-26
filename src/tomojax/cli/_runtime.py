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


def report_automatic(info: Mapping[str, object]) -> None:
    """Keep a successful but limited automatic result visible without a manifest."""
    automatic = object_mapping(info.get("automatic"))
    if automatic.get("status") == "limited":
        reasons = object_list(automatic.get("reasons"))
        LOG.warning("Automatic workflow is limited: %s", "; ".join(map(str, reasons)))
