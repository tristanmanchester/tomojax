"""Apply the coarse exterior model with native CUDA projections and bounded image blocks."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax._typed_arrays import Float32Array
from tomojax.core import backend_policy, checkpoint
from tomojax.core.cuda_types import CuPyValue
from tomojax.core.geometry.views import stack_view_poses

from ._cgls_cuda import array_module, iterate, native_device, update_kernels
from .cgls import CGLSConfig, resident_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax

    from tomojax.geometry import Detector, Geometry, Grid

    from ._exterior import LinearAxis


class DeviceRows:
    """Keep fine samples and their private background on the input device."""

    def __init__(self, data: jax.Array) -> None:
        self.data = data
        self.cp = array_module()
        self.input = self.cp.from_dlpack(data)
        self.output: jax.Array | None = None
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

    def write(self, part: slice, rows: slice, block: CuPyValue) -> None:
        from tomojax.core.host_transfer import allocate_device

        (device,) = self.data.devices()
        with self.cp.cuda.Device(device.local_hardware_id):
            if self.output is None:
                self.output = allocate_device(
                    self.data.shape, dtype=np.dtype(np.float32), device=device
                )
                self.buffer = self.cp.from_dlpack(self.output)
            self.buffer[part, rows] = block
            self.removed += self.cp.sum(self.cp.asarray(block), dtype=np.float64)

    def result(self) -> tuple[jax.Array | None, float]:
        if self.output is None:
            return None, 0.0
        with self.cp.cuda.Device(self.input.device.id):
            measured = float(self.input.sum(dtype=np.float64))
            return self.output, float(self.removed) / measured if measured else 0.0


class ConeOperator:
    """The same cone kernels as resident CGLS, called on the current CUDA stream."""

    def __init__(
        self,
        geometry: Geometry,
        grid: Grid,
        detector: Detector,
        poses: CuPyValue,
        *,
        batch: int,
        z_fast: bool,
    ) -> None:
        from tomojax.core.cone import cone_model

        model = cone_model(geometry, detector)
        assert model is not None
        self.cp = array_module()
        self.grid, self.detector, self.batch, self.z_fast = grid, detector, batch, z_fast
        self.coeff = self.cp.from_dlpack(model.coefficients(poses, grid))
        self.views = int(self.coeff.shape[0])

    def project(self, volume: CuPyValue, part: slice) -> CuPyValue:
        from tomojax.core.cone import forward_into

        coeff = self.coeff[part]
        out = self.cp.empty((len(coeff), self.detector.nv, self.detector.nu), np.float32)
        forward_into(out, coeff, volume, grid=self.grid, det=self.detector, z_fast=self.z_fast)
        return out

    def forward(self, volume: CuPyValue) -> CuPyValue:
        return self.project(volume, slice(None))

    def adjoint(self, images: CuPyValue) -> CuPyValue:
        from tomojax.core.cone import adjoint_into

        out = self.cp.zeros(self.grid.shape[::-1] if self.z_fast else self.grid.shape, np.float32)
        for first in range(0, self.views, self.batch):
            part = slice(first, first + self.batch)
            adjoint_into(
                out,
                self.coeff[part],
                images[part],
                out,
                grid=self.grid,
                det=self.detector,
                z_fast=self.z_fast,
            )
        return out


def axial_fit(
    poses: Float32Array, grid: Grid, detector: Detector, full: Float32Array, config: CGLSConfig
) -> Float32Array:
    """Solve the independent normal equations in parallel, one Krylov direction per row."""
    from tomojax.core.joseph_cuda import JosephCUDA

    cp = array_module()
    with native_device(cp):
        ops = JosephCUDA(poses, grid, detector, views_per_batch=config.views_per_batch, z_fast=True)
        residual = cp.array(full, dtype=np.float32, order="C")
        direction = ops.adjoint(residual)
        x = cp.zeros_like(direction)

        # Match the scalar CGLS dot products and fused updates. Summing a
        # multi-row volume as a single vector would couple its step lengths.
        def norm2(values: CuPyValue, axis: int) -> CuPyValue:
            planes = cp.ascontiguousarray(cp.moveaxis(values, axis, 0)).reshape(grid.nz, -1)
            return cp.stack([cp.dot(plane, plane) for plane in planes])

        kernels = update_kernels(cp)
        gamma = initial = norm2(direction, 2)
        eps = np.finfo(np.float32).eps
        threshold = (np.float32(1e-6) * cp.sqrt(initial)) ** 2
        restart = verified = cp.ones(grid.nz, dtype=bool)
        active = gamma > threshold
        data = residual.copy()
        assert config.iterations is not None
        for count in range(1, config.iterations + 1):
            progress.check_canceled()
            prediction = ops.forward(direction)
            denominator = norm2(prediction, 1)
            active &= denominator > 0
            alpha = cp.where(active, gamma / cp.where(active, denominator, 1), 0)
            kernels["axpy"](alpha, direction, x)
            kernels["axpy"](-alpha[None, :, None], prediction, residual)
            gradient = ops.adjoint(residual)
            next_gamma = norm2(gradient, 2)
            # Confirm small updates against the measured residual just as in
            # scalar CGLS. Each plane keeps its own restart and convergence state.
            small = alpha**2 * norm2(direction, 2) <= eps**2 * norm2(x, 2)
            tiny = (
                abs(alpha * direction) <= eps * cp.maximum(abs(x), np.finfo(np.float32).tiny)
            ).all(axis=(0, 1))
            candidate = next_gamma <= threshold
            periodic = ((count % 16 == 0) | restart) & (next_gamma <= eps * initial)
            check = active & (small | tiny | candidate | periodic)
            restart = cp.zeros_like(active)
            if bool(cp.any(check)):
                fitted = ops.forward(x)
                exact_residual = data - fitted
                exact = ops.adjoint(exact_residual)
                bound = ops.adjoint(abs(data) + abs(fitted))
                at_precision = (abs(exact) <= 8 * eps * bound).all(axis=(0, 1))
                exact_gamma = norm2(exact, 2)
                restart = check & (
                    candidate
                    | tiny
                    | at_precision
                    | (norm2(exact - gradient, 2) > 0.01 * next_gamma)
                )
                active &= ~(check & (tiny | (at_precision & verified & (exact_gamma >= gamma))))
                gradient = cp.where(check, exact, gradient)
                residual = cp.where(check[None, :, None], exact_residual, residual)
                next_gamma = cp.where(check, exact_gamma, next_gamma)
            active &= next_gamma > threshold
            if not bool(cp.any(active)):
                break
            beta = cp.where(active & ~restart, next_gamma / cp.where(active, gamma, 1), 0)
            kernels["xpay"](gradient, beta, direction)
            gamma, verified = next_gamma, check
        return np.ascontiguousarray(x.get().transpose(2, 1, 0))


def cone_fit(
    geometry: Geometry, grid: Grid, detector: Detector, data: Float32Array, config: CGLSConfig
) -> Float32Array | None:
    """Run the coarse matched CGLS solve without compiling a second XLA solver."""
    cp = array_module()
    with native_device(cp):
        if resident_bytes(grid, data.shape) > backend_policy.cupy_available_bytes():
            return None
        ops = ConeOperator(
            geometry,
            grid,
            detector,
            stack_view_poses(geometry, len(data)),
            batch=config.views_per_batch,
            z_fast=True,
        )
        with checkpoint.loop() as saves:
            x, _ = iterate(
                cp,
                ops,
                cp.asarray(data),
                cp.zeros(grid.shape[::-1], np.float32),
                grid,
                replace(config, tolerance=1e-6),
                saves,
                zero_start=True,
                target=None,
            )
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
