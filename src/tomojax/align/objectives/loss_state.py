from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class LossState:
    kind: str
    params: dict[str, float]
    # Optional per-view mask (n, nv, nu) for masked/ROI losses
    mask: jnp.ndarray | None = None
    # Optional per-view precomputes
    bins_x: jnp.ndarray | None = None
    bins_y: jnp.ndarray | None = None
    bw_x: float | None = None
    bw_y: float | None = None
    dt_edge: jnp.ndarray | None = None
    thr: jnp.ndarray | None = None  # per-view scalar thresholds broadcastable
