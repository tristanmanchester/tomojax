from __future__ import annotations

from contextlib import contextmanager, nullcontext
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

LOG = logging.getLogger(__name__)
_DISABLED_TRANSFER_GUARD_MODES = frozenset(("off", "none", "disable", "disabled"))


def _warn_transfer_guard_demotion(mode: str, reason: str) -> None:
    LOG.warning(
        "Requested transfer guard mode %r, but support is unavailable (%s); falling back to no-op.",
        mode,
        reason,
    )


@contextmanager
def transfer_guard_context(mode: str | None = None) -> Generator[None, None, None]:
    """Return the configured JAX transfer-guard context, if available."""
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log")
    mode = str(mode).lower()
    if mode in _DISABLED_TRANSFER_GUARD_MODES:
        with nullcontext():
            yield
        return
    try:
        import jax
    except Exception as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        with nullcontext():
            yield
        return

    transfer_guard = getattr(jax, "transfer_guard", None)
    if transfer_guard is not None:
        try:
            with transfer_guard(mode):
                yield
            return
        except Exception as exc:
            _warn_transfer_guard_demotion(mode, str(exc))
            with nullcontext():
                yield
            return
    try:
        from jax.experimental import transfer_guard as experimental_transfer_guard  # type: ignore

        try:
            with experimental_transfer_guard(mode):
                yield
            return
        except Exception as exc:
            _warn_transfer_guard_demotion(mode, str(exc))
            with nullcontext():
                yield
            return
    except Exception as exc:
        _warn_transfer_guard_demotion(mode, str(exc))
        with nullcontext():
            yield
