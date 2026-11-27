"""Estimate the part of a measured line integral made outside the requested voxel grid."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

from tomojax import progress
from tomojax.core import checkpoint
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import host_view_poses
from tomojax.core.host_volume import host_array
from tomojax.core.robust import MAD_TO_SD
from tomojax.geometry import Detector, Geometry, Grid, grid_volume_origin

from ._continuation import Continuation
from ._truncation import axis_distance
from .cgls import CGLSConfig, cgls

if TYPE_CHECKING:
    import jax

    from .fista_tv import FISTAConfig
    from .mlem import MLEMConfig
    from .sirt import SIRTConfig
    from .spdhg_tv import SPDHGConfig


def _bin(values: np.ndarray, count: int, axis: int) -> np.ndarray:
    """Equal-area averages over a detector axis, including fractional end cells."""
    length = values.shape[axis]
    if length == count:
        return values
    source = values.swapaxes(axis, -1)
    if length % count == 0:
        means = source.reshape(*source.shape[:-1], count, length // count).mean(axis=-1)
    else:
        edges = np.linspace(0, length, count + 1)
        cell = edges.astype(np.intp)
        sums = np.pad(
            np.cumsum(source, axis=-1, dtype=np.float64), [(0, 0)] * (source.ndim - 1) + [(1, 0)]
        )
        integrals = sums[..., cell] + (edges - cell) * source[..., np.minimum(cell, length - 1)]
        means = (np.diff(integrals) * (count / length)).astype(np.float32)
    return means.swapaxes(axis, -1)


def _poses(geometry: Geometry, views: int) -> np.ndarray:
    poses = host_view_poses(geometry, views)
    if poses is None:
        poses = np.asarray([geometry.pose_matrix(i) for i in range(views)], np.float32)
    return poses


def _unseen(geometry: Geometry, detector: Detector, grid: Grid, poses: np.ndarray) -> np.ndarray:
    """Extended detector columns whose entire pixel misses the grid in each view."""
    spacing = np.array([grid.vx, grid.vy, grid.vz])
    lower = np.asarray(grid_volume_origin(grid)) - spacing / 2
    box = np.stack([lower, lower + np.array([grid.nx, grid.ny, grid.nz]) * spacing])
    corners = box[np.indices((2, 2, 2)).reshape(3, 8).T, np.arange(3)]
    world = np.einsum("vij,kj->vki", poses[:, :3, :3], corners) + poses[:, None, :3, 3]
    parts = cone_parts(geometry, detector)
    if parts is None:
        roll = np.deg2rad(float(getattr(geometry, "detector_roll", 0)))
        u = (
            world[..., 0] * np.cos(roll) + world[..., 2] * np.sin(roll) - detector.center[0]
        ) / detector.du
    else:
        u = np.empty(world.shape[:2])
        first = 0
        for count, beam, own in parts:
            last = first + (count or len(poses))
            source = beam.source()
            center, horizontal, vertical = beam.detector_frame(own)
            normal = np.cross(horizontal, vertical)
            rays = world[first:last] - source
            depth = rays @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                hit = source + rays * (((center - source) @ normal) / depth)[..., None]
                columns = (hit - center) @ horizontal / own.du
            # A box crossing the source plane has no bounded detector silhouette.
            crossing = np.min(depth, axis=1) * np.max(depth, axis=1) <= 0
            columns[crossing, :2] = [-np.inf, np.inf]
            u[first:last] = columns
            first = last
    columns = np.arange(detector.nu) - (detector.nu - 1) / 2
    return (columns + 0.5 < u.min(axis=1, keepdims=True)) | (
        columns - 0.5 > u.max(axis=1, keepdims=True)
    )


@dataclass(frozen=True)
class _Sampling:
    detector: Detector
    grid: Grid

    @classmethod
    def for_scan(cls, grid: Grid, detector: Detector, reach: float) -> _Sampling:
        # The exterior determines a smooth background. Resolve at least 64 columns
        # on small scans; bound the auxiliary lattice independently of the fine grid.
        cols = min(detector.nu, max(64, min(128, math.ceil(detector.nu / 4))))
        ratio = detector.nu / cols
        rows = min(128, math.ceil(detector.nv / ratio))
        coarse = replace(
            detector, nu=cols, nv=rows, du=detector.du * ratio, dv=detector.dv * detector.nv / rows
        )
        vx, vy = max(grid.vx * ratio, reach / 192), max(grid.vy * ratio, reach / 192)
        nz = min(128, math.ceil(grid.nz / ratio))
        oz = grid_volume_origin(grid)[2]
        wide = Grid(
            2 * math.ceil(reach / vx),
            2 * math.ceil(reach / vy),
            nz,
            vx,
            vy,
            grid.vz * grid.nz / nz,
            vol_center=(0.0, 0.0, oz + (grid.nz - 1) * grid.vz / 2),
        )
        return cls(coarse, wide)

    def completed(self, geometry: Geometry, data: np.ndarray | jax.Array) -> np.ndarray:
        d = self.detector
        views, nv, nu = data.shape
        full = host_array((views, d.nv, 3 * d.nu), copies=4)
        middle = full[..., d.nu : 2 * d.nu]
        batch = max(1, (16 << 20) // (4 * nv * nu))
        for i in range(0, views, batch):
            progress.check_canceled()
            block = np.asarray(data[i : i + batch], np.float32)
            middle[i : i + batch] = _bin(_bin(block, d.nu, -1), d.nv, -2)
        fit = Continuation(axis_distance(geometry, d, d.nu), d.nu, d.nu)
        radius = fit.stack_radius(middle)[None]
        batch = max(1, (16 << 20) // (8 * d.nv * d.nu))
        for i in range(0, views, batch):
            lhs, rhs = fit.continued(np.asarray, middle[i : i + batch], radius)
            full[i : i + batch, :, : d.nu] = lhs
            full[i : i + batch, :, 2 * d.nu :] = rhs
        return full

    def exterior(self, volume: np.ndarray, grid: Grid) -> np.ndarray:
        wide = self.grid
        fractions = []
        for axis, (n, step, count, pitch) in enumerate(
            zip(
                (wide.nx, wide.ny),
                (wide.vx, wide.vy),
                (grid.nx, grid.ny),
                (grid.vx, grid.vy),
                strict=True,
            )
        ):
            lower = grid_volume_origin(grid)[axis] - pitch / 2
            cells = grid_volume_origin(wide)[axis] + np.arange(n) * step
            overlap = np.minimum(cells + step / 2, lower + count * pitch) - np.maximum(
                cells - step / 2, lower
            )
            fractions.append(np.clip(overlap / step, 0, 1))
        inside = fractions[1][:, None] * fractions[0]
        return volume * (1 - inside.astype(np.float32))


def _project(
    geometry: Geometry,
    detector: Detector,
    grid: Grid,
    poses: np.ndarray,
    volume: np.ndarray,
    config: CGLSConfig,
    *,
    native: bool,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Use the solver's physical forward projector with a bounded block of output rays."""
    batch = max(1, min(64, (16 << 20) // (4 * detector.nu * detector.nv)))
    if native:
        import cupy as cp

        from tomojax.core.joseph_cuda import JosephCUDA

        device_volume = cp.asarray(volume)
        for first in range(0, len(poses), batch):
            part = slice(first, first + batch)
            operator = JosephCUDA(poses[part], grid, detector, views_per_batch=batch)
            yield part, operator.forward(device_volume).get()
        return
    import jax.numpy as jnp

    from tomojax.core.cone import ConeModel, cone_project
    from tomojax.core.joseph import forward_project_planes, plane_coefficients
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    from ._projection import resolve_geometry_projector

    lattice = detector_grid_from_geometry_inputs(detector, geometry)
    model, backend = resolve_geometry_projector(
        geometry, config.backend, detector=detector, det_grid=lattice, context="reconstruct"
    )
    pose = jnp.asarray(poses)
    coeff = (
        model.coefficients(pose, grid)
        if isinstance(model, ConeModel)
        else plane_coefficients(pose, grid, detector, lattice)
    )
    device = jnp.asarray(volume)
    for first in range(0, len(poses), batch):
        part = slice(first, first + batch)
        if isinstance(model, ConeModel):
            block = cone_project(device, coeff[part], grid, detector, backend=backend)
        else:
            block = forward_project_planes(
                coeff[part], device, grid, detector, backend=backend, interpolation="linear"
            )
        yield part, np.asarray(block)


def outside_data(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray | jax.Array,
    config: CGLSConfig | FISTAConfig | SIRTConfig | MLEMConfig | SPDHGConfig,
) -> tuple[np.ndarray | None, float]:
    """Estimate an additive ray background and its signed share of measured attenuation."""
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    from ._fbp_storage import first_device

    views, nv, nu = data.shape
    distance = axis_distance(geometry, detector, nu)
    sample = _Sampling.for_scan(grid, detector, float(np.max(np.abs(distance))))
    d, wide = sample.detector, sample.grid
    complete = replace(d, nu=3 * d.nu)
    poses = _poses(geometry, views)
    mask = _unseen(geometry, complete, grid, poses)
    mask[:, d.nu : 2 * d.nu] = False
    if not mask.any():
        return None, 0.0
    full = sample.completed(geometry, data)
    observed = np.einsum("vru,vu->vr", full, mask, dtype=np.float64)
    # Positive exterior attenuation, fitted to the integral of the virtual rays
    # that no voxel can reach. Adjacent-view differences estimate its standard
    # error. Soft thresholding estimates a nonnegative mass for each coarse row.
    uncertainty = (
        MAD_TO_SD * np.median(np.abs(np.diff(observed, axis=0)), axis=0) / math.sqrt(2 * views)
        if views > 1
        else 0.0
    )
    mass = np.maximum(observed.mean(axis=0) - math.sqrt(2 * math.log(2 * views)) * uncertainty, 0.0)
    if not mass.any():
        return None, 0.0
    coarse = CGLSConfig(backend=config.backend, devices=getattr(config, "devices", None))
    with (
        checkpoint.suspended(),
        first_device(coarse) if coarse.devices is not None else nullcontext(),
    ):
        reconstruction, info = cgls(
            geometry,
            wide,
            complete,
            full,
            config=coarse,
            det_grid=detector_grid_from_geometry_inputs(complete, geometry),
        )
        native = info.get("execution_backend") == "cupy"
        volume = np.asarray(reconstruction)
        predicted = np.zeros(d.nv)
        residual = host_array((views, d.nv, d.nu))
        for part, block in _project(geometry, complete, wide, poses, volume, coarse, native=native):
            predicted += np.einsum("vru,vu->r", block, mask[part], dtype=np.float64) / views
            residual[part] = (full[part] - block)[..., d.nu : 2 * d.nu]
        scale = np.divide(mass, predicted, out=np.zeros_like(mass), where=predicted > 0)
        if not scale.any():
            return None, 0.0
        outside = sample.exterior(volume, grid)
        outside = np.concatenate([outside[:1], outside, outside[-1:]])
        wide = replace(wide, nz=wide.nz + 2)
        row = (np.arange(nv) + 0.5) * d.nv / nv - 0.5
        col = (np.arange(nu) + 0.5) * d.nu / nu - 0.5
        uv = np.stack(np.meshgrid(row, col, indexing="ij"))
        amplitude = np.interp(row, np.arange(d.nv), scale).astype(np.float32)[:, None]
        source = host_array(data.shape, copies=2)
        measured = removed = 0.0
        for part, block in _project(
            geometry, detector, wide, poses, outside, coarse, native=native
        ):
            progress.check_canceled()
            first = part.start or 0
            for i, projected in enumerate(block, first):
                mismatch = map_coordinates(residual[i], uv, order=1, mode="nearest")
                source[i] = (projected + mismatch) * amplitude
            removed += float(source[part].sum(dtype=np.float64))
            measured += float(np.asarray(data[part]).sum(dtype=np.float64))
        return source, removed / measured if measured else 0.0


def subtract_background(
    data: np.ndarray | jax.Array, background: np.ndarray
) -> np.ndarray | jax.Array:
    """Replace private host scratch with corrected measurements, retaining device placement."""
    batch = max(1, (16 << 20) // (4 * data.shape[1] * data.shape[2]))
    for first in range(0, len(data), batch):
        part = slice(first, first + batch)
        np.subtract(np.asarray(data[part]), background[part], out=background[part])
    if isinstance(data, np.ndarray):
        return background
    import jax

    return jax.device_put(background, data.sharding)
