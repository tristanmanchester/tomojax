from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp


@dataclass
class LossState:
    kind: str
    params: Dict[str, float]
    # Optional per-view mask (n, nv, nu) for masked/ROI losses
    mask: Optional[jnp.ndarray] = None
    # Optional per-view precomputes
    bins_x: Optional[jnp.ndarray] = None
    bins_y: Optional[jnp.ndarray] = None
    bw_x: Optional[float] = None
    bw_y: Optional[float] = None
    dt_edge: Optional[jnp.ndarray] = None
    thr: Optional[jnp.ndarray] = None  # per-view scalar thresholds broadcastable
