"""Direct volume rendering of an attenuation volume by ray casting, in JAX.

Rays from a perspective camera march through the volume in steps of half a
voxel. A transfer function gives each attenuation an emitted colour and an
opacity per millimetre, and samples are composited front to back (the
emission-absorption model). Opacity follows the attenuation at each sample;
colour follows the largest attenuation among its neighbouring voxels, so the
partial-volume rim of a dense part takes that part's colour rather than the
colour of a lighter material that shares the rim's value.

Surfaces are lit by a fixed light with Lambert and Blinn-Phong terms from the
volume's gradient, faded out where the gradient is small so the inside of a
uniform part is not shaded as a surface. Light materials are drawn by their
boundaries: their opacity is scaled by the gradient, so a translucent plate
shows its edges rather than a haze. Everything drawn is the volume and the
transfer function.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Transfer:
    """Piecewise-linear colour and opacity (per mm) against attenuation (mm^-1)."""

    stops: tuple[float, ...]
    colours: tuple[tuple[float, float, float], ...]
    opacity: tuple[float, ...]


@dataclass(frozen=True)
class Camera:
    """A perspective camera looking at ``target`` from ``distance`` mm."""

    target: tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float
    fov_deg: float = 24.0

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Eye position and the right, up and forward unit vectors."""
        a, e = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        eye = np.asarray(self.target) + self.distance * back
        forward = -back
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return eye, right, np.cross(right, forward), forward


def neighbourhood_max(volume: np.ndarray) -> np.ndarray:
    """The largest value among each voxel and its 26 neighbours."""
    out = volume.copy()
    for axis in range(3):
        padded = np.pad(out, [(1, 1) if a == axis else (0, 0) for a in range(3)], mode="edge")
        n = out.shape[axis]
        out = np.maximum(
            np.maximum(padded.take(range(n), axis), padded.take(range(1, n + 1), axis)),
            padded.take(range(2, n + 2), axis),
        )
    return out


def fields(
    volume: np.ndarray, voxel: float, light: np.ndarray, view: np.ndarray, edge: float
) -> dict[str, jax.Array]:
    """The volumes the rays sample: attenuation, colour key, light factor and boundary.

    A voxel is a boundary in proportion to its gradient up to ``edge`` (mm^-2).
    """
    volume = np.asarray(volume, np.float32)
    g = np.stack(np.gradient(volume, voxel), axis=-1)
    size = np.linalg.norm(g, axis=-1)
    normal = g / np.maximum(size, 1e-12)[..., None]
    half = (light + view) / np.linalg.norm(light + view)
    boundary = np.clip(size / edge, 0, 1)
    lit = 1 + boundary * (
        0.25 + 0.85 * np.abs(normal @ light) + 0.9 * np.abs(normal @ half) ** 40 - 1
    )
    del g, normal
    return {
        "mu": jnp.asarray(volume),
        "hue": jnp.asarray(neighbourhood_max(volume)),
        "lit": jnp.asarray(lit.astype(np.float32)),
        "boundary": jnp.asarray(boundary.astype(np.float32)),
    }


@partial(jax.jit, static_argnames=("steps",))
def _march(
    data: dict[str, jax.Array],
    rays: tuple[jax.Array, jax.Array, jax.Array],
    transfer: tuple[jax.Array, jax.Array, jax.Array],
    grid: tuple[jax.Array, float, float, float],
    steps: int,
) -> jax.Array:
    """Front-to-back composite along ``rays``; returns colour and transmittance."""
    origins, directions, near = rays
    stops, colours, opacity = transfer
    low, voxel, step, clear = grid

    def sample(name: str, points: jax.Array) -> jax.Array:
        index = (points - low) / voxel - 0.5
        return jax.scipy.ndimage.map_coordinates(
            data[name], list(index.T), order=1, mode="constant", cval=0.0
        )

    def body(carry: tuple[jax.Array, jax.Array], k: jax.Array) -> tuple[tuple, None]:
        colour, through = carry
        points = origins + (near + (k + 0.5) * step)[:, None] * directions
        mu, key = sample("mu", points), sample("hue", points)
        rgb = jnp.stack([jnp.interp(key, stops, colours[:, c]) for c in range(3)], axis=-1)
        emphasis = jnp.where(key < clear, 0.15 + 0.85 * sample("boundary", points), 1.0)
        alpha = 1 - jnp.exp(-jnp.interp(mu, stops, opacity) * emphasis * step)
        rgb = rgb * sample("lit", points)[:, None]
        return (colour + (through * alpha)[:, None] * rgb, through * (1 - alpha)), None

    start = (jnp.zeros_like(origins), jnp.ones(origins.shape[0]))
    (colour, through), _ = jax.lax.scan(body, start, jnp.arange(steps, dtype=jnp.float32))
    return jnp.concatenate([colour, through[:, None]], axis=-1)


def render(
    volume: np.ndarray,
    voxel: float,
    low: np.ndarray,
    camera: Camera,
    transfer: Transfer,
    *,
    width: int,
    height: int,
    light: np.ndarray,
    background: tuple[float, float, float],
    edge: float,
    clear: float = 0.0,
    rows: int = 48,
) -> np.ndarray:
    """An ``(height, width, 3)`` image in [0, 1]; ``low`` is the volume's lower corner.

    Materials whose neighbourhood stays below ``clear`` (mm^-1) are drawn by
    their boundaries, their insides nearly transparent.
    """
    eye, right, up, forward = camera.frame()
    light = np.asarray(light, np.float64) / np.linalg.norm(light)
    data = fields(volume, voxel, light, -forward, edge)
    high = low + np.array(volume.shape) * voxel
    step = 0.5 * voxel
    steps = int(np.ceil(np.linalg.norm(high - low) / step)) + 2
    tables = tuple(
        jnp.asarray(t, jnp.float32) for t in (transfer.stops, transfer.colours, transfer.opacity)
    )
    grid = (jnp.asarray(low, jnp.float32), voxel, step, clear)
    half = math.tan(math.radians(camera.fov_deg) / 2)
    xs = ((np.arange(width) + 0.5) / width * 2 - 1) * half
    image = np.zeros((height, width, 3), np.float32)
    for r0 in range(0, height, rows):
        ys = (1 - (np.arange(r0, min(r0 + rows, height)) + 0.5) / height * 2) * half
        sx, sy = np.meshgrid(xs, ys * height / width)
        d = forward + sx[..., None] * right + sy[..., None] * up
        d = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (low - eye) / d, (high - eye) / d
        near = np.nanmax(np.minimum(t0, t1), axis=1)
        near = np.minimum(near, np.nanmin(np.maximum(t0, t1), axis=1))
        rays = (
            jnp.asarray(np.broadcast_to(eye, d.shape), jnp.float32),
            jnp.asarray(d, jnp.float32),
            jnp.asarray(near, jnp.float32),
        )
        out = np.asarray(_march(data, rays, tables, grid, steps))
        colour = out[:, :3] + out[:, 3:] * np.asarray(background)
        image[r0 : r0 + rows] = colour.reshape(len(ys), width, 3)
    return np.clip(image, 0, 1)
