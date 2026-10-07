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
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
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


def pose_rotations(angles: np.ndarray) -> np.ndarray:
    """Rotation matrices ``R_y(beta) R_x(alpha) R_z(phi)`` of pose angles.

    ``angles`` holds ``(alpha, beta, phi)`` in radians in its first three
    columns, one row per view, as in a pose table; returns ``(views, 3, 3)``.
    """
    a = np.asarray(angles, dtype=np.float64)
    ca, sa = np.cos(a[:, 0]), np.sin(a[:, 0])
    cb, sb = np.cos(a[:, 1]), np.sin(a[:, 1])
    cp, sp = np.cos(a[:, 2]), np.sin(a[:, 2])
    zero, one = np.zeros_like(ca), np.ones_like(ca)
    rx = np.stack([one, zero, zero, zero, ca, -sa, zero, sa, ca], -1).reshape(-1, 3, 3)
    ry = np.stack([cb, zero, sb, zero, one, zero, -sb, zero, cb], -1).reshape(-1, 3, 3)
    rz = np.stack([cp, -sp, zero, sp, cp, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    return ry @ rx @ rz


def pose_angles(rotations: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pose_rotations`: ``(alpha, beta, phi)`` per rotation."""
    r = np.asarray(rotations, dtype=np.float64)
    alpha = np.arcsin(np.clip(-r[:, 1, 2], -1.0, 1.0))
    beta = np.arctan2(r[:, 0, 2], r[:, 2, 2])
    phi = np.arctan2(r[:, 1, 0], r[:, 1, 1])
    return np.stack([alpha, beta, phi], axis=1)
