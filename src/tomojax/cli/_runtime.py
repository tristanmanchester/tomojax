from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
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
    except Exception as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        return nullcontext()

    transfer_guard = getattr(jax, "transfer_guard", None)
    if transfer_guard is not None:
        try:
            return cast("AbstractContextManager[object]", transfer_guard(mode))
        except Exception as exc:
            _warn_transfer_guard_demotion(mode, str(exc))
            return nullcontext()
    try:
        from jax.experimental import transfer_guard as experimental_transfer_guard  # type: ignore

        try:
            return cast("AbstractContextManager[object]", experimental_transfer_guard(mode))
        except Exception as exc:
            _warn_transfer_guard_demotion(mode, str(exc))
            return nullcontext()
    except Exception as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        return nullcontext()
