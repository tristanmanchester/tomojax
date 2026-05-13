"""SE(3) transforms and utilities (NumPy implementation).

Provides jit-agnostic helpers for geometry composition and conversions.
"""

from __future__ import annotations

import numpy as np


def hat_so3(w: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix for a 3-vector."""
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Map an axis-angle vector in so(3) to a 3x3 rotation matrix."""
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
    """Invert a homogeneous SE(3) transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ri = R.T
    Ti[:3, :3] = Ri
    Ti[:3, 3] = -Ri @ t
    return Ti


def rotz(phi: float) -> np.ndarray:
    """Return a homogeneous transform for rotation about +z."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Return a homogeneous transform for rotation about an arbitrary axis."""
    a = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(a))
    if axis_norm < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    a = a / axis_norm
    R = exp_so3(a * theta)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def align_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return a rotation matrix mapping unit vector ``u`` to unit vector ``v``."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, u)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        k = tmp - np.dot(tmp, u) * u
        k = k / (np.linalg.norm(k) + 1e-12)
        K = hat_so3(k)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
    k = np.cross(u, v)
    s = float(np.linalg.norm(k))
    k = k / (s + 1e-12)
    K = hat_so3(k)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
