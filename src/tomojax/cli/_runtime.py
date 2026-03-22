from __future__ import annotations

import os
from contextlib import nullcontext


def transfer_guard_context(mode: str | None = None):
    """Return the configured JAX transfer-guard context, if available."""
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
    if mode in ("off", "none", "disable", "disabled"):
        return nullcontext()
    try:
        import jax

        transfer_guard = getattr(jax, "transfer_guard", None)
        if transfer_guard is not None:
            return transfer_guard(mode)
        try:
            from jax.experimental import transfer_guard as experimental_transfer_guard  # type: ignore

            return experimental_transfer_guard(mode)
        except Exception:
            return nullcontext()
    except Exception:
        return nullcontext()
