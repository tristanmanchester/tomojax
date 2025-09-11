from __future__ import annotations

import logging
import os


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

