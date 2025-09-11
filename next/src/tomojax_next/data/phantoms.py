from __future__ import annotations

import numpy as np


def cube(nx: int, ny: int, nz: int, size: float = 0.5, value: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Axis-aligned cube in a zero background with side length = size * min(nx,ny,nz).

    Centered in the volume. Returns float32 array of shape (nx, ny, nz).
    """
    x = np.zeros((nx, ny, nz), dtype=np.float32)
    s = int(max(1, round(size * min(nx, ny, nz))))
    sx = (nx - s) // 2
    sy = (ny - s) // 2
    sz = (nz - s) // 2
    x[sx : sx + s, sy : sy + s, sz : sz + s] = value
    return x


def blobs(nx: int, ny: int, nz: int, n_blobs: int = 5, seed: int | None = 0) -> np.ndarray:
    """Random Gaussian blobs normalized to [0, 1]. Deterministic with seed.

    Returns float32 array (nx, ny, nz).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((nx, ny, nz), dtype=np.float32)
    for _ in range(n_blobs):
        cx = rng.uniform(0.2 * nx, 0.8 * nx)
        cy = rng.uniform(0.2 * ny, 0.8 * ny)
        cz = rng.uniform(0.2 * nz, 0.8 * nz)
        sx = rng.uniform(0.05 * nx, 0.2 * nx)
        sy = rng.uniform(0.05 * ny, 0.2 * ny)
        sz = rng.uniform(0.05 * nz, 0.2 * nz)
        amp = rng.uniform(0.5, 1.0)
        gx = np.exp(-((np.arange(nx) - cx) ** 2) / (2 * sx * sx))
        gy = np.exp(-((np.arange(ny) - cy) ** 2) / (2 * sy * sy))
        gz = np.exp(-((np.arange(nz) - cz) ** 2) / (2 * sz * sz))
        G = amp * np.outer(gx, gy).reshape(nx, ny, 1) * gz.reshape(1, 1, nz)
        X += G.astype(np.float32)
    # Normalize to [0,1]
    X = X - X.min()
    if X.max() > 0:
        X = X / X.max()
    return X.astype(np.float32)


def shepp_logan_3d(nx: int, ny: int, nz: int) -> np.ndarray:
    """Simplified 3D Shepp-Logan-like phantom using stacked ellipsoids.

    Not a strict reference, but adequate for testing filters and reconstruction.
    """
    X = np.zeros((nx, ny, nz), dtype=np.float32)
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    def add_ellipsoid(axes, center, value):
        ax, ay, az = axes
        ox, oy, oz = center
        xs = (np.arange(nx) - (cx + ox)) / ax
        ys = (np.arange(ny) - (cy + oy)) / ay
        zs = (np.arange(nz) - (cz + oz)) / az
        E = (
            (xs[:, None, None] ** 2)
            + (ys[None, :, None] ** 2)
            + (zs[None, None, :] ** 2)
        ) <= 1.0
        X[E] += value
    add_ellipsoid((0.69 * nx / 2, 0.92 * ny / 2, 0.9 * nz / 2), (0, 0, 0), 1.0)
    add_ellipsoid((0.6624 * nx / 2, 0.8740 * ny / 2, 0.88 * nz / 2), (0, 0, 0), -0.2)
    add_ellipsoid((0.21 * nx / 2, 0.25 * ny / 2, 0.2 * nz / 2), (0.22 * nx / 4, 0, 0), -0.15)
    X = X - X.min()
    if X.max() > 0:
        X = X / X.max()
    return X.astype(np.float32)


# ----------------------------
# Random cubes + spheres phantom (deterministic)
# ----------------------------
def _rotation_matrix_3d(angles):
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float64)


def _add_rotated_cube_soft(vol: np.ndarray, center, size: float, value: float, angles, edge_softness: float = 1.0):
    """Add a softly edged rotated cube into vol by ROI evaluation (no SciPy)."""
    nx, ny, nz = vol.shape
    cx, cy, cz = center
    R = _rotation_matrix_3d(angles)
    half = size / 2.0
    # Cube corners for ROI bounds
    corners = np.array([
        [-half, -half, -half], [half, -half, -half],
        [-half, half, -half],  [half, half, -half],
        [-half, -half, half],  [half, -half, half],
        [-half, half, half],   [half, half, half]
    ], dtype=np.float64)
    rot = (R @ corners.T).T
    bb_min = np.floor(np.min(rot, axis=0) + np.array([cx, cy, cz])).astype(int)
    bb_max = np.ceil(np.max(rot, axis=0) + np.array([cx, cy, cz])).astype(int)
    bb_min = np.maximum(bb_min, [0, 0, 0])
    bb_max = np.minimum(bb_max, [nx - 1, ny - 1, nz - 1])
    if np.any(bb_max < bb_min):
        return
    xs = np.arange(bb_min[0], bb_max[0] + 1)
    ys = np.arange(bb_min[1], bb_max[1] + 1)
    zs = np.arange(bb_min[2], bb_max[2] + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([(X - cx), (Y - cy), (Z - cz)], axis=0).reshape(3, -1)
    Q = (R.T @ P).reshape(3, *X.shape)
    # Soft inside mask via distance to faces
    dx = np.maximum(0.0, np.abs(Q[0]) - (half - 0.5))
    dy = np.maximum(0.0, np.abs(Q[1]) - (half - 0.5))
    dz = np.maximum(0.0, np.abs(Q[2]) - (half - 0.5))
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    alpha = np.clip(1.0 - dist / max(edge_softness, 1e-6), 0.0, 1.0).astype(np.float32)
    sub = vol[xs.min():xs.max()+1, ys.min():ys.max()+1, zs.min():zs.max()+1]
    np.maximum(sub, (value * alpha).astype(np.float32), out=sub)


def random_cubes_spheres(
    nx: int,
    ny: int,
    nz: int,
    *,
    n_cubes: int = 8,
    n_spheres: int = 7,
    min_size: int = 4,
    max_size: int = 32,
    min_value: float = 0.1,
    max_value: float = 1.0,
    max_rot_degrees: float = 180.0,
    use_inscribed_fov: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Random rotated cubes + spheres phantom (deterministic).

    Ensures objects fit within FOV if `use_inscribed_fov=True`.
    """
    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    rng = np.random.default_rng(seed)

    cx0, cy0 = nx / 2.0, ny / 2.0
    fov_r = min(nx, ny) / 2.0 if use_inscribed_fov else float('inf')

    # Cubes
    for _ in range(max(0, int(n_cubes))):
        size = float(rng.uniform(min_size, max_size))
        if use_inscribed_fov:
            max_xy_extent = size * np.sqrt(2) / 2.0
            rmax = fov_r - max_xy_extent
            if rmax <= 1:
                continue
            r = rng.uniform(0, rmax)
            th = rng.uniform(0, 2 * np.pi)
            cx = cx0 + r * np.cos(th)
            cy = cy0 + r * np.sin(th)
            cz = float(rng.uniform(size / 2.0, nz - size / 2.0))
        else:
            margin = size * np.sqrt(3) / 2.0
            cx = float(rng.uniform(margin, nx - margin))
            cy = float(rng.uniform(margin, ny - margin))
            cz = float(rng.uniform(margin, nz - margin))
        val = float(rng.uniform(min_value, max_value))
        ang = np.deg2rad(rng.uniform(-max_rot_degrees, max_rot_degrees, size=3))
        _add_rotated_cube_soft(vol, (cx, cy, cz), size, val, ang)

    # Spheres
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]
    for _ in range(max(0, int(n_spheres))):
        radius = float(rng.uniform(min_size / 2.0, max_size / 2.0))
        if use_inscribed_fov:
            rmax = fov_r - radius
            if rmax <= 1:
                continue
            r = rng.uniform(0, rmax)
            th = rng.uniform(0, 2 * np.pi)
            cx = cx0 + r * np.cos(th)
            cy = cy0 + r * np.sin(th)
            cz = float(rng.uniform(radius, nz - radius))
        else:
            cx = float(rng.uniform(radius, nx - radius))
            cy = float(rng.uniform(radius, ny - radius))
            cz = float(rng.uniform(radius, nz - radius))
        val = float(rng.uniform(min_value, max_value))
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        vol[dist <= radius] = np.maximum(vol[dist <= radius], val)

    return vol.astype(np.float32)
