"""Read binary STL meshes, check they are closed, and voxelise them exactly.

Only NumPy is required. Occupancy is computed from the signed surface
crossings of vertical rays through a supersampled grid of voxel sub-centres, so
each voxel stores the fraction of its volume inside the mesh: exact along z, to
the supersampling resolution across.
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
    chunk: int = 4_000_000,
) -> np.ndarray:
    """Return the fraction of each voxel inside the mesh.

    ``shape`` is `(nz, ny, nx)`; ``spacing`` and ``origin`` are physical xyz.
    ``origin`` is the centre of voxel (0, 0, 0). Vertical rays through
    ``supersample`` x ``supersample`` sub-columns of each voxel cross the
    surface; a crossing where the outward normal points down enters the mesh,
    one where it points up leaves it. Each crossing adds (or removes) the part
    of its voxel above it and every voxel higher up, so the inside intervals are
    integrated exactly over each voxel's z extent without sorting crossings.
    Rays are offset by an irrational fraction of a sub-voxel so they never pass
    exactly through mesh vertices or edges. ``chunk`` bounds the candidate
    (triangle, ray) pairs held at once.
    """
    nz, ny, nx = shape
    s = int(supersample)
    spacing_ = np.asarray(spacing, np.float64)
    step = spacing_ / s
    jitter = np.array([np.sqrt(2) - 1, np.sqrt(3) - 1.5]) * 1e-4 * step[:2]
    start = np.asarray(origin, np.float64) - spacing_ / 2 + step / 2
    xs = start[0] + step[0] * np.arange(nx * s) + jitter[0]
    ys = start[1] + step[1] * np.arange(ny * s) + jitter[1]
    bottom = origin[2] - spacing[2] / 2
    a, b, c = (np.ascontiguousarray(triangles[:, k], np.float64) for k in range(3))
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    i0, i1 = np.searchsorted(xs, lo[:, 0]), np.searchsorted(xs, hi[:, 0], side="right")
    j0, j1 = np.searchsorted(ys, lo[:, 1]), np.searchsorted(ys, hi[:, 1], side="right")
    ni, nj = i1 - i0, j1 - j0
    index = np.flatnonzero((det != 0) & (ni > 0) & (nj > 0))
    counts = (ni * nj)[index]
    cells = nx * ny * (nz + 1)
    partial, steps = np.zeros(cells), np.zeros(cells)
    balance = np.zeros(nx * s * ny * s)
    ends = np.searchsorted(np.cumsum(counts), np.arange(chunk, counts.sum() + chunk, chunk))
    first = 0
    for last in np.unique(np.minimum(np.append(ends + 1, index.size), index.size)):
        if last <= first:
            continue
        t, n = index[first:last], counts[first:last]
        first = int(last)
        offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        t = np.repeat(t, n)
        i, j = i0[t] + offset // nj[t], j0[t] + offset % nj[t]
        gx, gy = xs[i] - c[t, 0], ys[j] - c[t, 1]
        w0 = ((b[t, 1] - c[t, 1]) * gx + (c[t, 0] - b[t, 0]) * gy) / det[t]
        w1 = ((c[t, 1] - a[t, 1]) * gx + (a[t, 0] - c[t, 0]) * gy) / det[t]
        hit = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        t, i, j, w0, w1 = t[hit], i[hit], j[hit], w0[hit], w1[hit]
        z = w0 * a[t, 2] + w1 * b[t, 2] + (1 - w0 - w1) * c[t, 2]
        sign = np.where(det[t] < 0, 1.0, -1.0)  # det < 0: outward normal points down
        balance += np.bincount(i * ny * s + j, weights=sign, minlength=balance.size)
        k = np.floor((z - bottom) / spacing[2]).astype(np.int64)
        column = ((i // s) * ny + j // s) * (nz + 1)
        below, inside = k < 0, (k >= 0) & (k < nz)
        steps += np.bincount(column[below], weights=sign[below], minlength=cells)
        above = (bottom + (k[inside] + 1) * spacing[2] - z[inside]) / spacing[2]
        partial += np.bincount(
            column[inside] + k[inside], weights=sign[inside] * above, minlength=cells
        )
        steps += np.bincount(column[inside] + k[inside] + 1, weights=sign[inside], minlength=cells)
    if np.any(np.abs(balance) > 0.5):
        raise ValueError("ray crossed an open surface; the mesh is not closed")
    filled = partial.reshape(nx, ny, nz + 1) + np.cumsum(steps.reshape(nx, ny, nz + 1), axis=-1)
    return (filled[..., :nz] / (s * s)).transpose(2, 1, 0)
