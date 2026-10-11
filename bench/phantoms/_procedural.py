"""Shared voxelisation for the procedural phantoms.

Each phantom is a function of physical position in mm. A *sampler* takes point
coordinates and returns a material label per point and a *margin*: a lower
estimate, in mm, of the distance from the point to the nearest material
boundary. Voxels are first sampled at their centres; those whose margin is
smaller than about a voxel are resampled on a ``fine`` x ``fine`` x ``fine``
grid of sub-voxel points and store the mean attenuation of the samples. Edges
are therefore antialiased by occupancy, never blurred, and a finer grid gives
the same object at a higher resolution.

Features much thinner than a sub-sample (foils, cracks) are integrated
analytically instead with :func:`slab_fraction`, the exact fraction of a cube
lying inside a planar slab.

Labels index the attenuation table; :data:`KEEP` marks samples that leave an
existing volume unchanged, so a sampler can overlay one feature on another.
Volumes are ``(x, y, z)`` with voxel centres at ``(i + 0.5) * voxel - fov / 2``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

KEEP = -1
"""Label for samples that keep the volume underneath an overlay."""

Sampler = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Grid:
    """A cubic grid of ``n`` voxels per side spanning ``fov`` mm, centred on 0."""

    n: int
    fov: float

    @property
    def voxel(self) -> float:
        """Voxel edge in mm."""
        return self.fov / self.n

    def centres(self) -> np.ndarray:
        """Voxel-centre coordinates along one axis in mm."""
        return ((np.arange(self.n) + 0.5) * self.voxel - self.fov / 2).astype(np.float64)

    def index_range(self, low: float, high: float) -> tuple[int, int]:
        """Return the half-open voxel index range overlapping ``[low, high]`` mm."""
        i0 = int(np.floor((low + self.fov / 2) / self.voxel))
        i1 = int(np.ceil((high + self.fov / 2) / self.voxel))
        return max(i0, 0), min(i1, self.n)


def workers() -> int:
    """Threads to use: half the cores, at most 16."""
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def _sub_offsets(fine: int, voxel: float) -> np.ndarray:
    steps = ((np.arange(fine) + 0.5) / fine - 0.5) * voxel
    return np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), -1).reshape(-1, 3)


def sample_voxels(
    sample: Sampler,
    centres: np.ndarray,
    voxel: float,
    mu: np.ndarray,
    *,
    fine: int,
    reach: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean attenuation and kept fraction of voxels at ``centres``.

    ``centres`` is ``(m, 3)`` in mm. Voxels whose centre margin is below
    ``reach`` voxels are resampled with ``fine**3`` sub-voxel points.
    """
    table = np.append(np.asarray(mu, np.float64), 0.0)
    labels, margin = sample(centres[:, 0], centres[:, 1], centres[:, 2])
    labels = np.asarray(labels)
    value = table[labels]
    keep = (labels == KEEP).astype(np.float64)
    refine = np.flatnonzero(np.asarray(margin) < reach * voxel)
    if refine.size == 0 or fine <= 1:
        return value, keep
    offsets = _sub_offsets(fine, voxel)
    batch = max(1, (1 << 20) // offsets.shape[0])
    for start in range(0, refine.size, batch):
        index = refine[start : start + batch]
        points = (centres[index, None, :] + offsets[None]).reshape(-1, 3)
        sub, _ = sample(points[:, 0], points[:, 1], points[:, 2])
        sub = np.asarray(sub).reshape(index.size, -1)
        value[index] = table[sub].mean(axis=1)
        keep[index] = (sub == KEEP).mean(axis=1)
    return value, keep


def voxelise(
    sample: Sampler,
    grid: Grid,
    mu: np.ndarray,
    *,
    fine: int,
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    under: np.ndarray | None = None,
    reach: float = 1.2,
) -> np.ndarray:
    """Voxelise ``sample`` on ``grid``, optionally only inside ``box`` (mm).

    With ``under``, the result is composited onto that volume in place: each
    voxel keeps ``under`` in proportion to its :data:`KEEP` samples.
    """
    n = grid.n
    c = grid.centres()
    ranges = [(0, n)] * 3 if box is None else [grid.index_range(lo, hi) for lo, hi in box]
    (i0, i1), (j0, j1), (k0, k1) = ranges
    out = np.zeros((n, n, n), np.float32) if under is None else under
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return out
    xs, ys = c[i0:i1], c[j0:j1]
    planes = max(1, (1 << 17) // (xs.size * ys.size))

    def slab(k: int) -> None:
        zs = c[k : min(k + planes, k1)]
        x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
        centres = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
        value, keep = sample_voxels(sample, centres, grid.voxel, mu, fine=fine, reach=reach)
        shape = (xs.size, ys.size, zs.size)
        region = (slice(i0, i1), slice(j0, j1), slice(k, k + zs.size))
        value = value.reshape(shape)
        if under is not None:
            value = value + keep.reshape(shape) * out[region]
        out[region] = value

    with ThreadPoolExecutor(workers()) as pool:
        list(pool.map(slab, range(k0, k1, planes)))
    return out


def sample_where(
    sample: Sampler,
    grid: Grid,
    mu: np.ndarray,
    mask: np.ndarray,
    *,
    fine: int,
    reach: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`voxelise` but only for voxels where ``mask`` is true.

    Returns the mean attenuation and kept fraction of those voxels, in the
    order of ``np.flatnonzero(mask)``.
    """
    index = np.flatnonzero(mask)
    c = grid.centres()
    i, j, k = np.unravel_index(index, mask.shape)
    value = np.zeros(index.size)
    keep = np.zeros(index.size)
    step = max(1, (1 << 20) // max(fine, 1) ** 3)

    def run(start: int) -> None:
        part = slice(start, start + step)
        centres = np.stack([c[i[part]], c[j[part]], c[k[part]]], -1)
        value[part], keep[part] = sample_voxels(
            sample, centres, grid.voxel, mu, fine=fine, reach=reach
        )

    with ThreadPoolExecutor(workers()) as pool:
        list(pool.map(run, range(0, index.size, step)))
    return value, keep


def box_cdf(t: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Return P(U1 + U2 + U3 <= t) for centred uniforms of the given widths.

    ``widths`` is ``(..., 3)``; zero widths are raised to a tiny fraction of the
    largest so the closed form (a sum of truncated cubes) stays finite.
    """
    t = np.asarray(t, np.float64)
    w = np.asarray(widths, np.float64)
    w = np.maximum(w, 1e-4 * w.max(axis=-1, keepdims=True))
    a, b, c = w[..., 0], w[..., 1], w[..., 2]
    s = t + 0.5 * (a + b + c)

    def cube(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0.0) ** 3

    total = (
        cube(s)
        - cube(s - a)
        - cube(s - b)
        - cube(s - c)
        + cube(s - a - b)
        + cube(s - a - c)
        + cube(s - b - c)
        - cube(s - a - b - c)
    )
    return np.clip(total / (6.0 * a * b * c), 0.0, 1.0)


def slab_fraction(
    offset: np.ndarray, normal: np.ndarray, low: np.ndarray, high: np.ndarray, voxel: float
) -> np.ndarray:
    """Return the fraction of a voxel cube inside a planar slab.

    The slab is ``low <= normal . (p - centre) + offset <= high``: ``offset`` is
    the voxel centre's coordinate across the slab and ``normal`` the unit slab
    normal, ``(..., 3)``. Exact for a plane; for curved layers it is exact to
    the curvature over one voxel.
    """
    widths = voxel * np.abs(normal)
    return np.maximum(box_cdf(high - offset, widths) - box_cdf(low - offset, widths), 0.0)


def hash_state(*keys: np.ndarray, salt: int = 0) -> np.ndarray:
    """Return a 64-bit hash state of integer key tuples (SplitMix64 rounds)."""
    seed = (0x9E3779B97F4A7C15 + salt * 0x632BE59BD9B4E019) % (1 << 64)
    h = np.full(np.shape(keys[0]), seed, np.uint64)
    for key in keys:
        h = hash_mix(h, key)
    return h


def hash_mix(state: np.ndarray, key: np.ndarray | int) -> np.ndarray:
    """Mix one more integer key into a hash state."""
    h = state ^ np.asarray(key).astype(np.int64).view(np.uint64)
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(31))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(29))


def hash_value(state: np.ndarray) -> np.ndarray:
    """Map a hash state to a uniform value in [0, 1)."""
    return (state >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def hash_uniform(*keys: np.ndarray, salt: int = 0) -> np.ndarray:
    """Return a deterministic uniform [0, 1) value for integer key tuples."""
    return hash_value(hash_state(*keys, salt=salt))


@dataclass(frozen=True)
class WaveNoise:
    """Smooth band-limited noise: a sum of random plane waves of unit RMS."""

    k: np.ndarray
    phase: np.ndarray

    @classmethod
    def make(cls, rng: np.random.Generator, wavelength: float, waves: int = 8) -> WaveNoise:
        """Draw ``waves`` directions with wavelengths within 30% of ``wavelength`` mm."""
        direction = rng.normal(size=(waves, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        length = wavelength * rng.uniform(0.7, 1.3, waves)
        return cls(2 * np.pi * direction / length[:, None], rng.uniform(0, 2 * np.pi, waves))

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate at points; zero mean, unit RMS."""
        total = np.zeros(np.shape(x), np.float64)
        for (kx, ky, kz), phase in zip(self.k, self.phase, strict=True):
            total += np.sin(kx * x + ky * y + kz * z + phase)
        return total * np.sqrt(2.0 / len(self.phase))


def random_rotations(rng: np.random.Generator, count: int) -> np.ndarray:
    """Return ``(count, 3, 3)`` uniformly random rotation matrices."""
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def preview(
    volume: np.ndarray,
    meta: dict[str, Any],
    path: str | Path,
    *,
    zoom: tuple[str, float, tuple[float, float, float, float]],
    vmax: float | None = None,
) -> None:
    """Save central xy and xz slices and one zoomed slice of ``volume``.

    ``zoom`` is ``(plane, position_mm, (h0, h1, v0, v1))``: plane ``"xy"`` or
    ``"xz"``, the slice position along the remaining axis and the window in mm.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n = volume.shape[0]
    grid = Grid(n, float(meta["fov_mm"]))
    half = grid.fov / 2
    top = vmax if vmax is not None else float(np.percentile(volume, 99.8))

    def plane(name: str, position: float) -> np.ndarray:
        k = int(np.clip(np.floor((position + half) / grid.voxel), 0, n - 1))
        return volume[:, :, k].T if name == "xy" else volume[:, k, :].T

    name, position, window = zoom
    panels = [
        ("xy", 0.0, (-half, half, -half, half), "central xy (z = 0)"),
        ("xz", 0.0, (-half, half, -half, half), "central xz (y = 0)"),
        (
            name,
            position,
            window,
            f"detail, {name} at {'z' if name == 'xy' else 'y'} = {position:g} mm",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    for ax, (kind, pos, (h0, h1, v0, v1), title) in zip(axes, panels, strict=True):
        image = ax.imshow(
            plane(kind, pos),
            cmap="gray",
            origin="lower",
            extent=(-half, half, -half, half),
            vmin=0.0,
            vmax=top,
            interpolation="nearest",
        )
        ax.set_xlim(h0, h1)
        ax.set_ylim(v0, v1)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if kind == "xy" else "z (mm)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85, label="attenuation (mm$^{-1}$)")
    fig.suptitle(
        f"{meta['name']}: {n}$^3$, {1000 * meta['voxel_mm']:.2f} µm voxels, "
        f"{meta['energy_keV']:g} keV, display clipped at {top:.3g} mm$^{{-1}}$"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(build: Callable[..., tuple[np.ndarray, dict[str, Any]]], **preview_args: Any) -> None:
    """Command line shared by the phantom modules: build, time and preview."""
    import argparse

    parser = argparse.ArgumentParser(
        description=build.__doc__.splitlines()[0] if build.__doc__ else None
    )
    parser.add_argument("--n", type=int, default=256, help="voxels per side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=Path, help="write a PNG of three slices here")
    parser.add_argument("--save", type=Path, help="write the volume and metadata to this .npz")
    args = parser.parse_args()
    start = time.perf_counter()
    volume, meta = build(args.n, seed=args.seed)
    seconds = time.perf_counter() - start
    print(f"{meta['name']}: {args.n}^3 in {seconds:.1f} s, voxel {1000 * meta['voxel_mm']:.2f} um")
    for material, value in meta["materials"].items():
        print(f"  {material:>24s}  {value:.4g} /mm")
    if args.preview:
        preview(volume, meta, args.preview, **preview_args)
    if args.save:
        np.savez_compressed(args.save, volume=volume, meta=json.dumps(meta))
