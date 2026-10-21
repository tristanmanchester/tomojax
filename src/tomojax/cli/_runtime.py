from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from contextlib import AbstractContextManager

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

        return cast("AbstractContextManager[object]", jax.transfer_guard(mode))
    except Exception as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        return nullcontext()


def report_automatic(info: Mapping[str, object]) -> None:
    """Keep a successful but limited automatic result visible without a manifest."""
    automatic = info.get("automatic")
    if isinstance(automatic, dict):
        record = cast("Mapping[str, object]", automatic)
        if record.get("status") == "limited":
            reasons = cast("Sequence[object]", record.get("reasons", ()))
            LOG.warning("Automatic workflow is limited: %s", "; ".join(map(str, reasons)))
