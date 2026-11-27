"""Estimate the part of a measured line integral made outside the requested voxel grid."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from tomojax import progress
from tomojax._typed_arrays import (
    Float32Array,
    Float64Array,
    max_float,
    numpy_float32_array,
    numpy_float64_array,
)
from tomojax.core import checkpoint
from tomojax.core.cuda_types import CuPyValue
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import host_view_poses
from tomojax.core.host_volume import host_array
from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry, grid_volume_origin

from ._continuation import Continuation
from ._exterior_oblique import ObliqueExterior
from ._truncation import axis_distance
from .cgls import CGLSConfig, cgls, resident_bytes

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from .fista_tv import FISTAConfig
    from .mlem import MLEMConfig
    from .sirt import SIRTConfig
    from .spdhg_tv import SPDHGConfig


def _bin(values: Float32Array, count: int, axis: int) -> Float32Array:
    """Equal-area averages over a detector axis, including fractional end cells."""
    length = int(values.shape[axis])
    if length == count:
        return values
    source = values.swapaxes(axis, -1)
    if length % count == 0:
        shape = (*map(int, source.shape[:-1]), count, length // count)
        means = source.reshape(shape).mean(axis=-1, dtype=np.float32)
    else:
        edges = np.linspace(0, length, count + 1, dtype=np.float64)
        cell = edges.astype(np.intp)
        totals = np.add.reduceat(source, cell[:-1], axis=-1, dtype=np.float64)
        partial = (edges[1:-1] - cell[1:-1]) * source[..., cell[1:-1]]
        totals[..., :-1] += partial
        totals[..., 1:] -= partial
        means = (totals * (count / length)).astype(np.float32)
    return means.swapaxes(axis, -1)


def _poses(geometry: Geometry, views: int) -> Float32Array:
    poses = host_view_poses(geometry, views)
    if poses is None:
        from tomojax.core.geometry.views import stack_view_poses

        poses = np.asarray(stack_view_poses(geometry, views))
    return numpy_float32_array(poses)


def _unseen(
    geometry: Geometry, detector: Detector, grid: Grid, poses: Float32Array
) -> NDArray[np.bool_]:
    """Extended detector columns whose entire pixel misses the grid in each view."""
    spacing = np.array([grid.vx, grid.vy, grid.vz])
    lower = np.asarray(grid_volume_origin(grid)) - spacing / 2
    box = np.stack([lower, lower + np.array([grid.nx, grid.ny, grid.nz]) * spacing])
    corners = box[np.indices((2, 2, 2)).reshape(3, 8).T, np.arange(3)]
    world: Float64Array = corners @ poses[:, :3, :3].swapaxes(1, 2) + poses[:, None, :3, 3]
    parts = cone_parts(geometry, detector)
    if parts is None:
        roll = math.radians(float(getattr(geometry, "detector_roll", 0)))
        u = numpy_float64_array(
            (world[..., 0] * math.cos(roll) + world[..., 2] * math.sin(roll) - detector.center[0])
            / detector.du
        )
    else:
        u = np.empty(world.shape[:2])
        first = 0
        for count, beam, own in parts:
            last = first + (count or len(poses))
            source = beam.source()
            center, horizontal, vertical = beam.detector_frame(own)
            normal = np.cross(horizontal, vertical)
            rays: Float64Array = world[first:last] - source
            depth: Float64Array = rays @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                hit: Float64Array = (
                    source + rays * (((center - source) @ normal) / depth)[..., None]
                )
                columns: Float64Array = (hit - center) @ horizontal / own.du
            # A box crossing the source plane has no bounded detector silhouette.
            ordered = np.sort(depth, axis=1)
            crossing = (ordered[:, 0] <= 0) & (ordered[:, -1] >= 0)
            columns[crossing, :2] = [-np.inf, np.inf]
            u[first:last] = columns
            first = last
    column = np.arange(detector.nu, dtype=np.float64) - (detector.nu - 1) / 2
    bounds = np.sort(u, axis=1)
    return (column[None, :] + 0.5 < bounds[:, :1]) | (column[None, :] - 0.5 > bounds[:, -1:])


class _Binning(NamedTuple):
    factor: int


@dataclass(frozen=True)
class _Sampling:
    detector: Detector
    grid: Grid
    binning: int

    @classmethod
    def for_scan(cls, grid: Grid, detector: Detector, reach: float, binning: int = 1) -> _Sampling:
        # The exterior determines a smooth background. Resolve at least 64 columns
        # on small scans; bound the auxiliary lattice independently of the fine grid.
        cols = min(detector.nu, max(64, min(128, math.ceil(detector.nu / 4))))
        cols = max(1, cols // binning)
        ratio = detector.nu / cols
        rows = min(detector.nv, max(8, min(128, math.ceil(detector.nv / ratio))))
        rows = max(1, rows // binning)
        coarse = replace(
            detector, nu=cols, nv=rows, du=detector.du * ratio, dv=detector.dv * detector.nv / rows
        )
        vx, vy = max(grid.vx * ratio, reach / 192), max(grid.vy * ratio, reach / 192)
        nz = min(grid.nz, max(8, min(128, math.ceil(grid.nz / ratio))))
        nz = max(1, nz // binning)
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
        return cls(coarse, wide, binning)

    @classmethod
    def fitting(
        cls, geometry: Geometry, grid: Grid, detector: Detector, views: int, room: int | None
    ) -> _Sampling:
        """Use the finest auxiliary lattice that fits beside the solver's counting buffers."""
        reach = max_float(np.abs(axis_distance(geometry, detector, detector.nu)))
        binning = 1
        while True:
            sample = cls.for_scan(grid, detector, reach, binning)
            d, wide = sample.detector, sample.grid
            fitted = len(range(0, views, _stride(geometry, wide)))
            if (
                room is None
                or resident_bytes(wide, (fitted, d.nv, 3 * d.nu)) <= room
                or (d.nu, d.nv, wide.nz) == (1, 1, 1)
            ):
                return sample
            binning *= 2

    def planes(self, rows: int, pitch: float) -> _Sampling:
        """Use this in-plane lattice independently on each physical detector row."""
        return replace(
            self,
            grid=replace(self.grid, nz=rows, vz=pitch, vol_center=(0.0, 0.0, 0.0)),
            detector=replace(
                self.detector, nv=rows, dv=pitch, center=(self.detector.center[0], 0.0)
            ),
        )

    def depth(self, geometry: Geometry, views: int, rows: int, room: int | None) -> int:
        shape = (len(range(0, views, _stride(geometry, self.grid))), 1, 3 * self.detector.nu)
        # Batch right-hand sides within 128 MiB of completed host rows and the
        # available resident device budget. A full scan reuses the same workspace.
        fitting = min(rows, max(1, (128 << 20) // (4 * views * 3 * self.detector.nu)))
        if room is not None:
            fitting = min(fitting, max(1, room // resident_bytes(self.grid, shape)))
        return 2 ** (fitting.bit_length() - 1)

    def completed(
        self, geometry: Geometry, data: np.ndarray | jax.Array | CuPyValue
    ) -> Float32Array:
        d = self.detector
        views, nv, nu = data.shape
        full: Float32Array = host_array((views, d.nv, 3 * d.nu), copies=4)
        middle = full[..., d.nu : 2 * d.nu]
        batch = max(1, (16 << 20) // (4 * nv * nu))
        for i in range(0, views, batch):
            progress.check_canceled()
            block = data[i : i + batch]
            if not isinstance(block, np.ndarray) and not hasattr(block, "devices"):
                from ._exterior_cuda import host_detector_block

                block = host_detector_block(block, d.nu)
            host = np.asarray(block, dtype=np.float32)
            middle[i : i + batch] = _bin(_bin(host, d.nu, -1), d.nv, -2)
        fit = Continuation(axis_distance(geometry, d, d.nu), d.nu, d.nu)
        radius = fit.stack_radius(middle)[None]
        batch = max(1, (16 << 20) // (8 * d.nv * d.nu))
        for i in range(0, views, batch):
            lhs, rhs = fit.continued(np.asarray, middle[i : i + batch], radius)
            full[i : i + batch, :, : d.nu] = lhs
            full[i : i + batch, :, 2 * d.nu :] = rhs
        return full

    def exterior(self, volume: Float32Array, grid: Grid) -> Float32Array:
        wide = self.grid
        fractions: list[Float64Array] = []
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
        inside: Float64Array = fractions[1][:, None] * fractions[0]
        return volume * (1 - inside.astype(np.float32))


def storage(
    geometry: Geometry, grid: Grid, detector: Detector, views: int, room: int | None = None
) -> tuple[int, int]:
    """Peak auxiliary host and device bytes, including the private projection stack."""
    from ._axial_slabs import independent_planes

    full_data = 4 * views * detector.nv * detector.nu
    independent = independent_planes(geometry, grid, detector)
    rows = detector.nv
    if independent:
        grid = replace(grid, nz=1)
        detector = replace(detector, nv=1)
    oblique = ObliqueExterior.for_scan(geometry, detector, views)
    if oblique is not None:
        geometry, detector = oblique.geometry, oblique.detector
    sample = _Sampling.fitting(geometry, grid, detector, views, room)
    if independent:
        depth = sample.depth(geometry, views, rows, room) if room is not None else 1
        sample = sample.planes(depth, detector.dv)
        detector = replace(detector, nv=depth)
    d, wide = sample.detector, sample.grid
    shape = (views, d.nv, 3 * d.nu)
    completed = 4 * math.prod(shape)
    volume = 4 * math.prod(wide.shape)
    # Completed rows and their measured-column residual survive through reprojection.
    # A streamed host result and the extended exterior lattice can coexist.
    block = min(views, max(1, (16 << 20) // (4 * detector.nv * detector.nu)))
    images = 4 * block * detector.nv * detector.nu
    project = 4 * (wide.nz + 2) * wide.ny * wide.nx + images
    # Rolled chords have a distinct measured interval on each horizontal row.
    mask = math.prod(shape) if oblique is not None else 0
    host = full_data + completed + completed // 3 + 3 * volume + 4 * images + mask
    fitted_shape = (len(range(0, views, _stride(geometry, wide))), *shape[1:])
    return host, max(resident_bytes(wide, fitted_shape), project)


def _stride(geometry: Geometry, grid: Grid) -> int:
    """Largest angular stride moving a corner by at most one auxiliary voxel between views."""
    if type(geometry) is not ParallelGeometry or len(geometry.angles) < 2:
        return 1
    gap = max_float(np.abs(np.diff(np.unwrap(np.deg2rad(geometry.angles)))))
    distance = 0.5 * math.hypot(grid.nx * grid.vx, grid.ny * grid.vy)
    return max(1, math.floor(min(grid.vx, grid.vy) / (distance * gap))) if gap else 1


class _Runtime(NamedTuple):
    native: bool


def _fit(
    geometry: Geometry,
    wide: Grid,
    detector: Detector,
    full: np.ndarray,
    config: CGLSConfig,
    *,
    native: bool,
    axial: bool,
) -> tuple[Float32Array, bool]:
    """Resume the coarse fit independently and retain its finished coefficients for the fine fit."""
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    with checkpoint.scope("continued-exterior"), checkpoint.loop("volume") as fitted:
        if fitted.saved is not None:
            return fitted.saved.arrays[0], native and fitted.saved.scalars(_Runtime).native
        stride = _stride(geometry, wide)
        if isinstance(geometry, ParallelGeometry) and stride > 1:
            geometry = replace(geometry, angles=np.asarray(geometry.angles)[::stride])
            full = np.ascontiguousarray(full[::stride])
        if axial:
            from ._exterior_cuda import axial_fit

            volume = axial_fit(_poses(geometry, len(full)), wide, detector, full, config)
        else:
            reconstruction, info = cgls(
                geometry,
                wide,
                detector,
                full,
                config=config,
                det_grid=detector_grid_from_geometry_inputs(detector, geometry),
            )
            volume = numpy_float32_array(reconstruction)
            if native and cone_parts(geometry, detector) is not None:
                from tomojax.core.cone import use_cuda_cone

                # Reproject the reference cone iterate with the same native kernel.
                # Its CGLS reductions determine the fitted low-frequency background.
                native = use_cuda_cone()
            else:
                native = info.get("execution_backend") == "cupy"
        fitted.save((volume,), _Runtime(native))
        return volume, native


def _project(
    geometry: Geometry,
    detector: Detector,
    grid: Grid,
    poses: Float32Array,
    volume: Float32Array,
    config: CGLSConfig,
    *,
    native: bool,
) -> Iterator[tuple[slice[int, int, None], Float32Array | CuPyValue]]:
    """Use the solver's physical forward projector with a bounded block of output rays."""
    batch = max(1, (16 << 20) // (4 * detector.nu * detector.nv))
    if native:
        from tomojax.core.joseph_cuda import JosephCUDA

        from ._cgls_cuda import array_module, native_device

        cp = array_module()
        with native_device(cp):
            device_volume = cp.asarray(volume, order="C")
            if cone_parts(geometry, detector) is not None:
                from ._exterior_cuda import project_cone

                yield from project_cone(geometry, grid, detector, poses, device_volume, batch)
                return
            z_fast = grid.nz > 1
            if z_fast:
                device_volume = cp.ascontiguousarray(device_volume.transpose(2, 1, 0))
            for first in range(0, len(poses), batch):
                part = slice(first, min(first + batch, len(poses)))
                operator = JosephCUDA(
                    poses[part], grid, detector, views_per_batch=batch, z_fast=z_fast
                )
                yield part, operator.forward(device_volume)
        return
    batch = min(64, batch)
    from tomojax.core.cone import ConeModel, cone_project
    from tomojax.core.joseph import forward_project_planes, plane_coefficients
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    from ._projection import resolve_geometry_projector

    lattice = detector_grid_from_geometry_inputs(detector, geometry)
    model, backend = resolve_geometry_projector(
        geometry, config.backend, detector=detector, det_grid=lattice, context="reconstruct"
    )
    from jax.lax import convert_element_type

    pose = convert_element_type(poses, np.float32)
    coeff = (
        model.coefficients(pose, grid)
        if isinstance(model, ConeModel)
        else plane_coefficients(pose, grid, detector, lattice)
    )
    device = convert_element_type(volume, np.float32)
    for first in range(0, len(poses), batch):
        part = slice(first, min(first + batch, len(poses)))
        if isinstance(model, ConeModel):
            block = cone_project(device, coeff[part], grid, detector, backend=backend)
        else:
            block = forward_project_planes(
                coeff[part], device, grid, detector, backend=backend, interpolation="linear"
            )
        yield part, numpy_float32_array(block)


class Treatment(NamedTuple):
    """The data model recorded with the fine solve and in its checkpoint."""

    outside_grid: float
    data_model: str
    background_treatment: str

    @staticmethod
    def slab_totals(
        data: np.ndarray | jax.Array, info: dict[str, object]
    ) -> dict[str, float | bool]:
        """Weight a slab's fitted share by its original measured attenuation."""
        import jax

        record = info["truncation"]
        assert isinstance(record, dict)
        with jax.enable_x64(True):
            mass = float(data.sum(dtype=np.float64))
        return {
            "measured_sum": mass,
            "exterior_sum": mass * record["outside_grid"],
            "exterior_applied": record["background_treatment"] != "none",
        }

    @classmethod
    def from_slabs(cls, method: str, finished: checkpoint.Slabs) -> Treatment:
        """Combine additive attenuation totals and remove them from the per-slab reports."""
        measured, removed, applied = 0.0, 0.0, False
        for report in finished.reports.values():
            mass, outside = report.pop("measured_sum", 0.0), report.pop("exterior_sum", 0.0)
            assert isinstance(mass, float)
            assert isinstance(outside, float)
            measured += mass
            removed += outside
            applied |= bool(report.pop("exterior_applied", False))
        return cls.for_fit(method, removed / measured if measured else 0.0, applied=applied)

    @classmethod
    def record(cls, method: str, share: float, *, applied: bool) -> Treatment:
        treatment = cls.for_fit(method, share, applied=applied)
        with checkpoint.loop("projection-model") as model:
            if model.saved is None:
                model.save((), treatment)
        return treatment

    @classmethod
    def for_fit(cls, method: str, share: float, *, applied: bool) -> Treatment:
        return (
            cls(
                share,
                "continued_exterior",
                "additive_poisson" if method == "mlem" else "subtracted",
            )
            if applied
            else cls(0.0, "measured", "none")
        )


def outside_data(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray | jax.Array,
    config: CGLSConfig | FISTAConfig | SIRTConfig | MLEMConfig | SPDHGConfig,
) -> tuple[np.ndarray | jax.Array | None, float]:
    """Estimate an additive ray background and its share of measured attenuation."""
    from ._fbp_storage import first_device

    # Every fine device fits the same fixed exterior. Consult the first selected
    # allocator before fitting independent rows or the coupled volume.
    with first_device(config) if getattr(config, "devices", None) is not None else nullcontext():
        room, native = _resources()
        result, share = _background(
            geometry, grid, detector, data, config, room=room, native=native
        )
        if native and result is not None:
            # The fine JAX solve cannot reuse cached CuPy blocks from the fit.
            from ._cgls_cuda import array_module, native_device

            cp = array_module()
            with native_device(cp):
                cp.get_default_memory_pool().free_all_blocks()
        return result, share


class _HostRows:
    """Collect bounded model blocks in a zeroed host stack, allocating only on the first hit."""

    def __init__(self, data: np.ndarray | jax.Array) -> None:
        self.input = data
        self.output: Float32Array | None = None
        self.removed = 0.0

    def write(self, part: slice, rows: slice, block: Float32Array | CuPyValue) -> None:
        if self.output is None:
            self.output = host_array(self.input.shape, copies=2)
        target = self.output[part, rows]
        if isinstance(block, np.ndarray):
            target[:] = block
        elif target.flags.c_contiguous:
            block.get(out=target)
        else:
            target[:] = block.get()
        self.removed += float(target.sum(dtype=np.float64))

    def result(self) -> tuple[Float32Array | None, float]:
        if self.output is None:
            return None, 0.0
        if isinstance(self.input, np.ndarray):
            measured = float(self.input.sum(dtype=np.float64))
        else:
            import jax

            with jax.enable_x64(True):
                measured = float(self.input.sum(dtype=np.float64))
        return self.output, self.removed / measured if measured else 0.0


def _background(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray | jax.Array,
    config: CGLSConfig | FISTAConfig | SIRTConfig | MLEMConfig | SPDHGConfig,
    *,
    room: int | None,
    native: bool,
) -> tuple[np.ndarray | jax.Array | None, float]:
    from ._axial_slabs import independent_planes

    independent = independent_planes(geometry, grid, detector)
    rows = detector.nv
    if independent:
        assert isinstance(geometry, ParallelGeometry)
        origin = grid_volume_origin(grid)
        grid = replace(grid, nz=1, vol_center=None, vol_origin=(*origin[:2], 0.0))
        detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
        geometry = replace(geometry, grid=grid, detector=detector)

    views = data.shape[0]
    oblique = ObliqueExterior.for_scan(geometry, detector, views)
    if oblique is not None:
        geometry, detector = oblique.geometry, oblique.detector
    with checkpoint.loop("exterior-sampling") as sampling:
        if sampling.saved is None:
            sample = _Sampling.fitting(geometry, grid, detector, views, room)
            sampling.save((), _Binning(sample.binning))
        else:
            reach = max_float(np.abs(axis_distance(geometry, detector, detector.nu)))
            sample = _Sampling.for_scan(
                grid, detector, reach, sampling.saved.scalars(_Binning).factor
            )
    depth = rows
    if independent:
        depth = sample.depth(geometry, views, rows, room) if native else 1
    result = None
    if not isinstance(data, np.ndarray) and native and len(data.devices()) == 1:
        from ._exterior_cuda import DeviceRows

        result = DeviceRows.on_selected_device(data)
    result = result or _HostRows(data)
    for z in range(0, rows, depth):
        section = slice(z, min(z + depth, rows))
        count = section.stop - z
        local = sample.planes(count, detector.dv) if independent else sample
        measured_detector = replace(detector, nv=count) if independent else detector
        target = (
            replace(
                grid,
                nz=count,
                vol_center=None,
                vol_origin=(*grid_volume_origin(grid)[:2], -(count - 1) * grid.vz / 2),
            )
            if independent
            else grid
        )
        with checkpoint.scope(f"exterior-rows{z}-{section.stop}") if independent else nullcontext():
            blocks = _model(
                geometry,
                target,
                measured_detector,
                result.input[:, section],
                config,
                local,
                native=native,
                axial=independent and native,
                oblique=oblique,
            )
            if oblique is not None:
                blocks = oblique.measured_rays(blocks, positive=config.method == "mlem")
            for part, block in blocks:
                result.write(part, section, block)
    return result.result()


@dataclass(frozen=True)
class LinearAxis:
    lower: NDArray[np.intp]
    upper: NDArray[np.intp]
    fraction: Float32Array

    @classmethod
    def cells(cls, source: int, target: int) -> LinearAxis:
        coordinate = np.clip((np.arange(target) + 0.5) * source / target - 0.5, 0, source - 1)
        lower = coordinate.astype(np.intp)
        return cls(
            lower, np.minimum(lower + 1, source - 1), (coordinate - lower).astype(np.float32)
        )

    def interpolate(self, rows: Float32Array) -> Float32Array:
        """Linear interpolation along the last axis, clamped at both end centres."""
        left = np.take(rows, self.lower, axis=-1)
        right = np.take(rows, self.upper, axis=-1)
        return numpy_float32_array(left + (right - left) * self.fraction)


def _resources() -> tuple[int | None, bool]:
    """Room for the coarse fit and availability of native kernels on its device."""
    from tomojax.core import backend_policy

    from ._cgls_cuda import array_module, native_device, native_platform

    if native_platform():
        cp = array_module()
        if cp.cuda.is_available():
            with native_device(cp):
                return backend_policy.cupy_available_bytes(), True
    if backend_policy.cuda_backend():
        from tomojax.backends import device_free_memory_bytes

        return device_free_memory_bytes(), False
    return None, False


def _model(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray | jax.Array,
    config: CGLSConfig | FISTAConfig | SIRTConfig | MLEMConfig | SPDHGConfig,
    sample: _Sampling,
    *,
    native: bool,
    axial: bool,
    oblique: ObliqueExterior | None,
) -> Iterator[tuple[slice[int, int, None], Float32Array | CuPyValue]]:
    views, nv, nu = data.shape
    coarse = CGLSConfig(iterations=50)
    d, wide = sample.detector, sample.grid
    # If both fitted edge windows contain only air on every view, every
    # virtual ray is identically zero. No interior samples enter that result.
    window = math.ceil(min(d.nu, max(2, d.nu // 16)) * nu / d.nu)
    if oblique is None and not (data[..., :window].any() or data[..., -window:].any()):
        return
    complete = replace(d, nu=3 * d.nu)
    poses = _poses(geometry, views)
    # With a single ray family, virtual and measured rays never intersect
    # inside the object. The extension adds no observations of the measured paths.
    if (
        np.array_equal(poses, np.broadcast_to(poses[:1], poses.shape))
        and cone_parts(geometry, detector) is None
    ):
        return
    unseen = _unseen(geometry, complete, grid, poses)
    if oblique is None:
        unseen[:, d.nu : 2 * d.nu] = False
    if not unseen.any():
        return
    if oblique is None:
        full, mask = sample.completed(geometry, data), unseen[:, None, :]
    else:
        full, mask = oblique.completed(data, d, unseen)
    observed = np.sum(full, axis=-1, where=mask, dtype=np.float64)
    # Positive exterior attenuation, fitted to the integral of the virtual rays
    # that no voxel can reach. The number of missed rays varies by view and can
    # be zero over most of a turn. The second moment of adjacent differences
    # retains the variance of the views that actually contribute.
    uncertainty: Float64Array | float = (
        np.sqrt(np.mean(np.diff(observed, axis=0) ** 2, axis=0) / (2 * views)) if views > 1 else 0.0
    )
    mass: Float64Array = np.maximum(
        np.mean(observed, axis=0, dtype=np.float64)
        - math.sqrt(2 * math.log(2 * views)) * uncertainty,
        0.0,
    )
    if not mass.any():
        return
    # The completed stack includes correlated, synthetic rays. Its residual
    # does not have the measured scan's noise law.
    volume, native = _fit(geometry, wide, complete, full, coarse, native=native, axial=axial)
    predicted: Float64Array = np.zeros(d.nv)
    residual: Float32Array = host_array((views, d.nv, d.nu))
    for part, projected in _project(geometry, complete, wide, poses, volume, coarse, native=native):
        block = numpy_float32_array(
            projected if isinstance(projected, np.ndarray) else projected.get()
        )
        predicted += np.sum(block, axis=(0, 2), where=mask[part], dtype=np.float64) / views
        residual[part] = full[part, :, d.nu : 2 * d.nu] - block[..., d.nu : 2 * d.nu]
    scale = np.divide(mass, predicted, out=np.zeros_like(mass), where=predicted > 0)
    if not scale.any():
        return
    outside = sample.exterior(volume, grid)
    # A coarsened z lattice interpolates the edge row through a half-cell
    # beyond its end centres. At native z sampling the known grid extent
    # already defines the footprint, including diverging cone rays.
    if wide.nz < grid.nz:
        outside = np.concatenate([outside[:1], outside, outside[-1:]])
        wide = replace(wide, nz=wide.nz + 2)
    projected = _project(geometry, detector, wide, poses, outside, coarse, native=native)
    yield from _interpolate(
        projected, residual, scale, detector, native=native, positive=config.method == "mlem"
    )


def _interpolate(
    projected: Iterator[tuple[slice[int, int, None], Float32Array | CuPyValue]],
    residual: Float32Array,
    scale: Float64Array,
    detector: Detector,
    *,
    native: bool,
    positive: bool,
) -> Iterator[tuple[slice[int, int, None], Float32Array | CuPyValue]]:
    """Lift the calibrated coarse prediction residual onto the measured rays."""
    nv, nu = detector.nv, detector.nu
    _, coarse_rows, coarse_columns = residual.shape
    row = (np.arange(nv) + 0.5) * coarse_rows / nv - 0.5
    amplitude = np.interp(row, np.arange(coarse_rows), scale).astype(np.float32)[:, None]
    vertical, horizontal = LinearAxis.cells(coarse_rows, nv), LinearAxis.cells(coarse_columns, nu)
    if native:
        from ._exterior_cuda import interpolate

        yield from interpolate(
            projected, residual, amplitude, horizontal, vertical, positive=positive
        )
        return
    for part, block in projected:
        progress.check_canceled()
        mismatch = residual[part]
        if coarse_columns != nu:
            mismatch = horizontal.interpolate(mismatch)
        if coarse_rows != nv:
            mismatch = vertical.interpolate(mismatch.swapaxes(1, 2)).swapaxes(1, 2)
        corrected = (block + mismatch) * amplitude
        if positive:
            _ = np.maximum(corrected, 0, out=corrected)
        yield part, corrected


def subtract_background(
    data: np.ndarray | jax.Array, background: np.ndarray | jax.Array
) -> np.ndarray | jax.Array:
    """Replace private host scratch with corrected measurements, retaining device placement."""
    if isinstance(data, np.ndarray) and isinstance(background, np.ndarray):
        _ = np.subtract(data, background, out=background)
        return background
    import jax

    assert isinstance(data, jax.Array)
    private = jax.device_put(background, data.sharding)
    # The fitted background is private and can become the corrected stack.
    corrected: jax.Array = jax.jit(_subtract, donate_argnums=(1,))(data, private)
    return corrected


def _subtract(data: jax.Array, private: jax.Array) -> jax.Array:
    return (data - private).astype(np.float32)
