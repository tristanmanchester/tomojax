from __future__ import annotations

import logging
import math
import os


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    # Quiet noisy backend probe logs from JAX unless explicitly enabled
    if os.environ.get("TOMOJAX_BACKEND_LOG", "0").lower() not in ("1", "true", "yes", "on"):
        for name in (
            "jax._src.xla_bridge",
            "jax._src.xla_client",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


def log_jax_env() -> None:
    try:
        import jax

        logging.info("JAX backend: %s", jax.default_backend())
        logging.info(
            "Devices: %s",
            repr(jax.devices()),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        )
    except Exception:  # pragma: no cover
        logging.info("JAX not available for logging")


def format_duration(seconds: float | None) -> str:
    """Render a wall-clock duration as a compact human-readable string."""
    if seconds is None:
        return "-"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    value = max(value, 0.0)
    if value < 1e-3:
        return f"{value * 1e6:.0f}µs"
    if value < 1.0:
        return f"{value * 1e3:.0f}ms" if value < 0.1 else f"{value:.2f}s"

    minutes, seconds_rem = divmod(value, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h{int(minutes):02d}m{seconds_rem:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m{seconds_rem:04.1f}s"
    return f"{seconds_rem:.1f}s"
