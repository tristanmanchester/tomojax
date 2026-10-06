"""Read binary STL meshes, check they are closed, and voxelise them exactly.

Only NumPy is required. Occupancy is computed by ray parity along z through a
supersampled grid of voxel sub-centres, so each voxel stores the fraction of
its volume inside the mesh (to the supersampling resolution).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_stl(path: str | Path) -> np.ndarray:
    """Return triangles of a binary STL as an ``(n, 3, 3)`` float64 array."""
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, dtype=record, count=count, offset=84)["v"]
    return triangles.astype(np.float64)


def check_closed(triangles: np.ndarray, tolerance: float = 1e-9) -> dict[str, int]:
    """Count edges not shared by exactly two triangles; a closed mesh has none."""
    vertices = np.round(triangles.reshape(-1, 3) / tolerance).astype(np.int64)
    _, index = np.unique(vertices, axis=0, return_inverse=True)
    faces = index.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"edges": int(counts.size), "open_edges": int(np.count_nonzero(counts != 2))}


def occupancy(
    triangles: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    *,
    supersample: int = 4,
) -> np.ndarray:
    """Return the fraction of each voxel inside the mesh.

    ``origin`` is the centre of voxel (0, 0, 0). Vertical rays through
    ``supersample`` x ``supersample`` sub-columns of each voxel record where
    they cross the surface; the inside intervals between successive crossings
    are integrated exactly over each voxel's z extent. Rays are offset by an
    irrational fraction of a sub-voxel so they never pass exactly through mesh
    vertices or edges.
    """
    nx, ny, nz = shape
    s = int(supersample)
    step = np.asarray(spacing, np.float64) / s
    jitter = np.array([np.sqrt(2) - 1, np.sqrt(3) - 1.5]) * 1e-4 * step[:2]
    start = np.asarray(origin, np.float64) - np.asarray(spacing) / 2 + step / 2
    xs = start[0] + step[0] * np.arange(nx * s) + jitter[0]
    ys = start[1] + step[1] * np.arange(ny * s) + jitter[1]
    z_edges = origin[2] - spacing[2] / 2 + spacing[2] * np.arange(nz + 1)
    hits: list[list[float]] = [[] for _ in range(xs.size * ys.size)]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    for p0, p1, p2 in zip(a, b, c, strict=True):
        lo = np.minimum(np.minimum(p0, p1), p2)
        hi = np.maximum(np.maximum(p0, p1), p2)
        i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0], side="right")
        j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1], side="right")
        if i0 >= i1 or j0 >= j1:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        # Barycentric coordinates of each ray in the triangle's xy projection.
        d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if d == 0:
            continue  # vertical triangle: no vertical ray crosses it
        w0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / d
        w1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        ii, jj = np.nonzero(inside)
        for i, j, value in zip(ii + i0, jj + j0, z[inside], strict=True):
            hits[i * ys.size + j].append(float(value))
    fine = np.zeros((xs.size, ys.size, nz))
    for column, crossings in enumerate(hits):
        if not crossings:
            continue
        crossings.sort()
        if len(crossings) % 2:
            raise ValueError("ray crossed an open surface; the mesh is not closed")
        i, j = divmod(column, ys.size)
        for z0, z1 in zip(crossings[0::2], crossings[1::2], strict=True):
            overlap = np.minimum(z_edges[1:], z1) - np.maximum(z_edges[:-1], z0)
            fine[i, j] += np.maximum(overlap, 0.0) / spacing[2]
    return fine.reshape(nx, s, ny, s, nz).mean(axis=(1, 3))
