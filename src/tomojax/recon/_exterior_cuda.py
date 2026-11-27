"""Apply the coarse exterior model with native CUDA projections and bounded image blocks."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax._typed_arrays import Float32Array, numpy_float32_array
from tomojax.core.cuda_types import CuPyValue

from ._cgls_cuda import array_module, native_device, update_kernels

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax
    from numpy.typing import NDArray

    from tomojax.geometry import Detector, Geometry, Grid

    from ._exterior import LinearAxis
    from .cgls import CGLSConfig


@cache
def _detector_means() -> CuPyValue:
    # A contiguous NumPy float32 reduction accumulates eight lanes per leaf of
    # up to 128 elements. Changing this order changes the coarse CGLS directions.
    return array_module().ElementwiseKernel(
        "raw float32 samples, int32 width",
        "float32 mean",
        """
        ptrdiff_t start = (ptrdiff_t)i * width;
        float sum = 0.f;
        int j = 0;
        if (width >= 8) {
            float lane[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) lane[k] = samples[start+k];
            for (j = 8; j + 7 < width; j += 8) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) lane[k] += samples[start+j+k];
            }
            sum = ((lane[0]+lane[1])+(lane[2]+lane[3]))
                + ((lane[4]+lane[5])+(lane[6]+lane[7]));
        }
        for (; j < width; ++j) sum += samples[start+j];
        mean = sum == 0.f ? 0.f : __fdiv_rn(sum, (float)width);
        """,
        "exterior_detector_means",
    )


def host_detector_block(block: CuPyValue, columns: int) -> Float32Array:
    """Average single-leaf bins before transfer and leave other widths to the host."""
    width, remainder = divmod(block.shape[-1], columns)
    if remainder or not 2 <= width <= 128:
        return numpy_float32_array(block.get())
    shape = (*block.shape[:-1], columns)
    means = _detector_means()(
        block.astype(np.float32, copy=False), np.int32(width), size=block.size // width
    )
    return numpy_float32_array(means.reshape(shape).get())


class DeviceRows:
    """Keep fine samples and their private background on the input device."""

    def __init__(self, data: jax.Array) -> None:
        self.data = data
        self.cp = array_module()
        self.input = self.cp.from_dlpack(data)
        self.buffer: CuPyValue = None
        self.removed = self.cp.zeros((), np.float64)

    @classmethod
    def on_selected_device(cls, data: jax.Array) -> DeviceRows | None:
        (device,) = data.devices()
        cp = array_module()
        with native_device(cp) as current:
            if device.platform == "gpu" and device.local_hardware_id == current.id:
                return cls(data)
        return None

    def _zero_buffer(self) -> CuPyValue:
        from tomojax.core.host_transfer import allocate_device

        (device,) = self.data.devices()
        stats = device.memory_stats() or {}
        dtype = np.dtype(np.float32)
        # Reuse a reserved block if it fits. Growing the JAX pool for this
        # short-lived stack can retain several times the slab's size.
        if stats.get("largest_free_block_bytes", 0) < self.data.size * dtype.itemsize:
            try:
                with self.cp.cuda.using_allocator(None):
                    return self.cp.zeros(self.data.shape, dtype=dtype)
            except (
                self.cp.cuda.memory.OutOfMemoryError,
                self.cp.cuda.runtime.CUDARuntimeError,
            ) as err:
                if isinstance(err, self.cp.cuda.runtime.CUDARuntimeError) and err.status != 2:
                    raise  # Only cudaErrorMemoryAllocation can use the reserved pool instead.
        storage = allocate_device(self.data.shape, dtype=dtype, device=device)
        return self.cp.from_dlpack(storage)

    def write(self, part: slice, rows: slice, block: CuPyValue) -> None:
        with self.cp.cuda.Device(self.input.device.id):
            if self.buffer is None:
                self.buffer = self._zero_buffer()
            self.buffer[part, rows] = block
            self.removed += self.cp.sum(self.cp.asarray(block), dtype=np.float64)

    def result(self) -> tuple[jax.Array | None, float]:
        if self.buffer is None:
            return None, 0.0
        from jax.dlpack import from_dlpack

        with self.cp.cuda.Device(self.input.device.id):
            measured = float(self.input.sum(dtype=np.float64))
            return from_dlpack(self.buffer), float(self.removed) / measured if measured else 0.0


def project_cone(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    poses: Float32Array,
    volume: CuPyValue,
    batch: int,
) -> Iterator[tuple[slice[int, int, None], CuPyValue]]:
    """Project the fitted cone volume directly into blocks on the active CUDA stream."""
    from jax.lax import convert_element_type

    from tomojax.core.cone import cone_model, forward_into

    model = cone_model(geometry, detector)
    assert model is not None
    cp = array_module()
    coefficients = cp.from_dlpack(model.coefficients(convert_element_type(poses, np.float32), grid))
    for first in range(0, len(poses), batch):
        part = slice(first, first + batch)
        coeff = coefficients[part]
        out = cp.empty((len(coeff), detector.nv, detector.nu), dtype=np.float32)
        forward_into(out, coeff, volume, grid=grid, det=detector, z_fast=False)
        yield part, out


class _Planes:
    """Matched projections and independent BLAS inner products on aligned detector rows."""

    def __init__(
        self, poses: Float32Array, grid: Grid, detector: Detector, config: CGLSConfig
    ) -> None:
        from tomojax.core.joseph_cuda import JosephCUDA

        self.cp = array_module()
        self.ops = JosephCUDA(
            poses, grid, detector, views_per_batch=config.views_per_batch, z_fast=True
        )
        self.rows = grid.nz

    def squared(self, values: CuPyValue, axis: int) -> CuPyValue:
        planes = self.cp.ascontiguousarray(self.cp.moveaxis(values, axis, 0)).reshape(self.rows, -1)
        return self.cp.stack([self.cp.dot(plane, plane) for plane in planes])

    def verify(
        self,
        data: CuPyValue,
        x: CuPyValue,
        residual: CuPyValue,
        gradient: CuPyValue,
        check: NDArray[np.bool_],
    ) -> tuple[CuPyValue, CuPyValue, Float32Array, Float32Array, NDArray[np.bool_]]:
        """Recompute against the observations for planes near convergence or roundoff."""
        cp = self.cp
        fitted = self.ops.forward(x)
        exact_residual = data - fitted
        exact = self.ops.adjoint(exact_residual)
        _ = cp.abs(fitted, out=fitted)
        fitted += abs(data)
        bound = self.ops.adjoint(fitted)
        del fitted
        bounded = (abs(exact) <= (8 * np.finfo(np.float32).eps) * bound).all(axis=(0, 1))
        norm, gap, at_precision = numpy_float32_array(
            cp.stack(
                (
                    self.squared(exact, 2),
                    self.squared(exact - gradient, 2),
                    bounded,
                )
            ).get()
        )
        if check.all():
            residual, gradient = exact_residual, exact
        else:
            selected = cp.asarray(check)
            gradient = cp.where(selected, exact, gradient)
            residual = cp.where(selected[None, :, None], exact_residual, residual)
        return residual, gradient, norm, gap, at_precision.astype(bool)


def axial_fit(
    poses: Float32Array, grid: Grid, detector: Detector, full: Float32Array, config: CGLSConfig
) -> Float32Array:
    """Solve the independent normal equations in parallel, one Krylov direction per row."""
    cp = array_module()
    with native_device(cp):
        planes = _Planes(poses, grid, detector, config)
        data = cp.asarray(full, dtype=np.float32, order="C")
        residual = data.copy()
        direction = planes.ops.adjoint(residual)
        x = cp.zeros_like(direction)
        kernels = update_kernels(cp)
        gamma = initial = numpy_float32_array(planes.squared(direction, 2).get())
        eps = np.finfo(np.float32).eps
        threshold = (np.float32(1e-6) * np.sqrt(initial)) ** 2
        restart = verified = np.ones(grid.nz, dtype=bool)
        active = gamma > threshold
        assert config.iterations is not None
        for count in range(1, config.iterations + 1):
            progress.check_canceled()
            prediction = planes.ops.forward(direction)
            denominator = numpy_float32_array(planes.squared(prediction, 1).get())
            active &= denominator > 0
            alpha = np.divide(gamma, denominator, out=np.zeros_like(gamma), where=active)
            step = cp.asarray(alpha)
            kernels["axpy"](step, direction, x)
            kernels["axpy"](-step[None, :, None], prediction, residual)
            del prediction
            gradient = planes.ops.adjoint(residual)
            tiny = (
                abs(step * direction) <= eps * cp.maximum(abs(x), np.finfo(np.float32).tiny)
            ).all(axis=(0, 1))
            # Only vectors with one number per plane cross the bus. Keeping step
            # decisions on the host avoids dozens of scalar GPU dispatches per step.
            next_gamma, nd, nx, tiny_value = numpy_float32_array(
                cp.stack(
                    (
                        planes.squared(gradient, 2),
                        planes.squared(direction, 2),
                        planes.squared(x, 2),
                        tiny,
                    )
                ).get()
            )
            small = alpha**2 * nd <= eps**2 * nx
            stalled = tiny_value.astype(bool)
            candidate = next_gamma <= threshold
            periodic = ((count % 16 == 0) | restart) & (next_gamma <= eps * initial)
            check = active & (small | stalled | candidate | periodic)
            restart = np.zeros_like(active)
            if check.any():
                residual, gradient, exact, gap, bounded = planes.verify(
                    data, x, residual, gradient, check
                )
                restart = check & (candidate | stalled | bounded | (gap > 0.01 * next_gamma))
                active &= ~(check & (stalled | (bounded & verified & (exact >= gamma))))
                next_gamma = np.where(check, exact, next_gamma)
            active &= next_gamma > threshold
            if not active.any():
                break
            beta = np.divide(next_gamma, gamma, out=np.zeros_like(gamma), where=active & ~restart)
            kernels["xpay"](gradient, cp.asarray(beta), direction)
            gamma, verified = next_gamma, check
        return np.ascontiguousarray(x.get().transpose(2, 1, 0))


@cache
def _lift() -> CuPyValue:
    cp = array_module()
    return cp.ElementwiseKernel(
        "float32 projected, raw float32 residual, raw float32 amplitude, "
        "raw int64 x, raw float32 wx, raw int64 y, raw float32 wy, "
        "int32 nu, int32 nv, int32 cu, int32 cv, bool positive",
        "float32 value",
        """
        int u = i % nu, v = i / nu % nv, view = i / (nu * nv);
        int lo = x[u], hi = min(lo + 1, cu - 1), bottom = y[v];
        int a = (view * cv + bottom) * cu, b = (view * cv + min(bottom + 1, cv - 1)) * cu;
        float left = residual[a+lo] + (residual[a+hi] - residual[a+lo]) * wx[u];
        float right = residual[b+lo] + (residual[b+hi] - residual[b+lo]) * wx[u];
        value = (projected + (left + (right - left) * wy[v])) * amplitude[v];
        if (positive) value = fmaxf(0.f, value);
        """,
        "exterior_interpolate",
        options=("--fmad=false",),
    )


def interpolate(
    projected: Iterator[tuple[slice[int, int, None], CuPyValue]],
    residual: Float32Array,
    amplitude: Float32Array,
    horizontal: LinearAxis,
    vertical: LinearAxis,
    *,
    positive: bool,
) -> Iterator[tuple[slice[int, int, None], CuPyValue]]:
    """Interpolate the coarse mismatch and scale the modeled rays on the active device."""
    cp = array_module()
    with native_device(cp):
        weights = tuple(
            cp.asarray(a)
            for a in (
                amplitude,
                horizontal.lower,
                horizontal.fraction,
                vertical.lower,
                vertical.fraction,
            )
        )
        dims = tuple(
            map(
                np.int32,
                (len(horizontal.lower), len(vertical.lower), residual.shape[2], residual.shape[1]),
            )
        )
        for part, block in projected:
            progress.check_canceled()
            value = _lift()(block, cp.asarray(residual[part]), *weights, *dims, positive)
            yield part, value
