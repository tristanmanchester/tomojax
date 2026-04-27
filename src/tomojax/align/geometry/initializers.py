from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Geometry

from .._loss_adapters import LossAdapter


@dataclass(frozen=True, slots=True)
class DetectorCenterSeed:
    """Projection-domain detector-u seed in native detector pixels."""

    det_u_px: float
    intercept_px: float
    amplitude_px: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "det_u_px": float(self.det_u_px),
            "intercept_px": float(self.intercept_px),
            "amplitude_px": float(self.amplitude_px),
            "status": self.status,
        }


def train_heldout_view_indices(
    n_views: int,
    *,
    holdout_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic interleaved train/held-out view indices for characterization."""
    n = int(n_views)
    if n < 4:
        raise ValueError("held-out split requires at least 4 views")
    stride = max(3, int(holdout_stride))
    all_indices = np.arange(n, dtype=np.int32)
    heldout = all_indices[(all_indices % stride) == (stride // 2)]
    if heldout.size == 0:
        heldout = all_indices[:: max(2, n // 4)]
    train_mask = np.ones((n,), dtype=bool)
    train_mask[heldout] = False
    train = all_indices[train_mask]
    if train.size == 0 or heldout.size == 0:
        raise ValueError("held-out split produced an empty partition")
    return train, heldout


def projection_com_det_u_seed(
    projections: jnp.ndarray,
    geometry: Geometry,
    loss_adapter: LossAdapter,
) -> DetectorCenterSeed:
    """Estimate a cheap detector-u initializer from projection centre-of-mass evidence.

    This is deliberately only an initializer/diagnostic. It does not choose the
    calibration result; setup geometry is solved by the active-state objective path.
    """
    y = np.asarray(projections, dtype=np.float32)
    n_views, _nv, nu = y.shape
    u = np.arange(nu, dtype=np.float32) - (float(nu) - 1.0) * 0.5
    mask = getattr(loss_adapter.state, "mask", None)
    if mask is not None:
        weights = np.asarray(mask, dtype=np.float32)
    else:
        shifted = y - np.min(y, axis=(1, 2), keepdims=True)
        weights = np.maximum(shifted, 0.0)
    denom = np.sum(weights, axis=(1, 2))
    numerator = np.sum(weights * u[None, None, :], axis=(1, 2))
    valid = denom > 1e-6
    if int(np.count_nonzero(valid)) < 3:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="insufficient_projection_mass",
        )
    com = np.zeros((n_views,), dtype=np.float32)
    com[valid] = numerator[valid] / denom[valid]
    theta = np.deg2rad(np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32))
    design = np.stack(
        [
            np.ones_like(theta[valid]),
            np.cos(theta[valid]),
            np.sin(theta[valid]),
        ],
        axis=1,
    )
    coeffs, *_ = np.linalg.lstsq(design, com[valid], rcond=None)
    intercept = float(coeffs[0])
    amplitude = float(np.hypot(coeffs[1], coeffs[2]))
    return DetectorCenterSeed(
        det_u_px=intercept,
        intercept_px=intercept,
        amplitude_px=amplitude,
        status="ok",
    )
