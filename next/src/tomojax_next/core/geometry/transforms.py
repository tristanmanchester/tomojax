"""SE(3) transforms and utilities (NumPy implementation).

Provides jit-agnostic helpers for geometry composition and conversions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def hat_so3(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = hat_so3(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) twist to SE(3) matrix.

    xi = [wx, wy, wz, vx, vy, vz]
    """
    w = np.asarray(xi[:3], dtype=np.float64)
    v = np.asarray(xi[3:], dtype=np.float64)
    theta = float(np.linalg.norm(w))
    R = exp_so3(w)
    if theta < 1e-12:
        V = np.eye(3, dtype=np.float64)
    else:
        K = hat_so3(w)
        theta2 = theta * theta
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta2 * K
            + (theta - np.sin(theta)) / (theta2 * theta) * (K @ K)
        )
    t = V @ v
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Compose homogeneous transforms: returns T_a @ T_b."""
    return T_a @ T_b


def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ri = R.T
    Ti[:3, :3] = Ri
    Ti[:3, 3] = -Ri @ t
    return Ti


def rotz(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    R = exp_so3(a * theta)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T

