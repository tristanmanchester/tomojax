from __future__ import annotations

import logging
import os
from typing import Iterable, Iterator, Optional


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")


def log_jax_env() -> None:
    try:
        import jax
        logging.info("JAX backend: %s", jax.default_backend())
        logging.info("Devices: %s", jax.devices())
    except Exception:  # pragma: no cover
        logging.info("JAX not available for logging")


def _progress_enabled() -> bool:
    v = os.environ.get("TOMOJAX_PROGRESS", "0").lower()
    return v in ("1", "true", "yes", "on")


def progress_iter(iterable: Iterable, *, total: Optional[int] = None, desc: str = "") -> Iterator:
    """Yield elements from iterable, showing a progress bar if enabled.

    Enable by setting environment variable `TOMOJAX_PROGRESS=1` or calling CLIs with `--progress`.
    Uses tqdm if available; otherwise prints occasional step counters.
    """
    if not _progress_enabled():
        for x in iterable:
            yield x
        return
    try:
        from tqdm import tqdm  # type: ignore

        for x in tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False):
            yield x
    except Exception:
        # Lightweight fallback: print at ~10% increments
        if total is None:
            for i, x in enumerate(iterable, 1):
                if i == 1 or i % 10 == 0:
                    print(f"{desc} step {i}", flush=True)
                yield x
        else:
            step = max(1, total // 10)
            for i, x in enumerate(iterable, 1):
                if i == 1 or i == total or i % step == 0:
                    print(f"{desc} {i}/{total}", flush=True)
                yield x
