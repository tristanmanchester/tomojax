"""Slab FFT execution and runtime selection of available CUDA support."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import operator
from threading import Lock
from typing import Any, Literal, cast

import numpy as np

from tomojax.geometry import Detector, Grid, grid_volume_origin

from ._fourier_grid import interpolate_numpy, next_fast_len, radial_coefficients, transform_grid
from ._fourier_rows import ProjectionRows, interpolate_projection_rows


def resolve_fourier_backend(backend: str) -> Literal["numpy", "cupy"]:
    """Resolve auto once, without falling back after reconstruction has started."""
    if backend in {"numpy", "cupy"}:
        return "cupy" if backend == "cupy" else "numpy"
    if backend != "auto":
        raise ValueError("fourier_reconstruct: backend must be 'auto', 'numpy' or 'cupy'")
    import jax

    device = jax.config.jax_default_device
    if device is None or isinstance(device, str):
        device = jax.devices(device)[0]
    if device.platform != "gpu" or not device.client.platform_version.lower().startswith("cuda"):
        return "numpy"
    try:
        import cupy
    except ImportError:
        return "numpy"
    if not cupy.cuda.is_available():
        return "numpy"
    # Never change the caller's current CUDA device or mix it with another JAX device.
    if cupy.cuda.runtime.getDevice() != device.local_hardware_id:
        return "numpy"
    return "cupy"


def fourier_slab_depth(grid: Grid, detector: Detector, requested: int | None, backend: str) -> int:
    """Bound automatic CUDA FFT planes before allocation; explicit budgets stay fixed."""
    if requested is not None:
        depth = operator.index(requested)
    elif backend == "cupy":
        shape, _ = transform_grid(grid, detector)
        # Several transform buffers, workspaces and pending transfers coexist.
        # Bounding one padded FP32 plane stack avoids expensive retry cascades.
        planes = max(1, (256 * 1024**2) // (4 * shape[0] * shape[1]))
        depth = min(16, 1 << (planes.bit_length() - 1))
    else:
        depth = 16
    if depth < 1:
        raise ValueError("fourier_reconstruct: slices_per_batch must be positive")
    return min(depth, grid.nz)


def is_cuda_oom(error: Exception) -> bool:
    """Recognize allocator exhaustion without swallowing unrelated failures."""
    if not isinstance(error, MemoryError):
        return False
    try:
        import cupy
    except ImportError:
        return False
    return isinstance(error, cupy.cuda.memory.OutOfMemoryError)


class _Geometry:
    def __init__(self, grid, detector, nviews, flipped, angles, backend) -> None:
        self.cuda = backend == "cupy"
        if self.cuda:
            try:
                import cupy as xp
            except ImportError as exc:
                raise ImportError(
                    "fourier_reconstruct: backend='cupy' requires tomojax[fourier-cuda12]"
                ) from exc
            from ._fourier_cuda import interpolation_kernel, mass_deapod_kernel

            self.mass_kernel = mass_deapod_kernel()
        else:
            xp = np
        self.xp = xp
        self.grid, self.detector = grid, detector
        angles, starts, counts = np.unique(angles, return_index=True, return_counts=True)
        self.nviews = len(angles)
        self.groups = None if self.nviews == nviews else np.append(starts, nviews).astype(np.int32)
        self.device_groups = xp.asarray([] if self.groups is None else self.groups, dtype=xp.int32)
        gaps = np.diff(np.append(angles, angles[0] + np.pi)).astype(np.float64)
        self.view_weights = xp.asarray(
            np.repeat((gaps + np.roll(gaps, 1)) / (2 * np.pi * counts), counts)
        )
        self.uniform = np.allclose(gaps, np.pi / self.nviews, rtol=1e-5, atol=np.deg2rad(2e-5))
        self.nfft = max(64, next_fast_len(2 * detector.nu))
        self.shape, self.crop = transform_grid(grid, detector)
        self.flipped = xp.asarray(flipped, dtype=xp.int8)
        self.period_phase = complex(np.exp(2j * np.pi * (detector.nu - 1) / 2))
        real = xp.float32 if self.cuda else xp.float64
        complex_dtype = xp.complex64 if self.cuda else xp.complex128
        phase, deapod, table = radial_coefficients(detector.nu, self.nfft, detector.du)
        self.projection_phase = xp.asarray(phase, dtype=complex_dtype)
        self.deapod = xp.asarray(deapod, dtype=real)
        self.table = xp.asarray(table, dtype=real)
        px, py = self.shape
        qy = py // 2 + 1
        self.coordinates = self._coordinates(angles)
        self.x_boundary = None
        stop = qy - 1 if py % 2 == 0 else qy
        if px % 2 == 0 and grid.vx > detector.du and stop > 1:
            # At x Nyquist, -kx maps to the same stored bin. If its line is
            # inside the detector's disk, conjugate symmetry needs the missing
            # opposite-y sample, not just conjugation of the current sample.
            coordinates = self._coordinates(angles, mirror=True)
            self.x_boundary = (coordinates, stop)
        if self.cuda:
            self.kernel = interpolation_kernel(
                x_boundary=self.x_boundary is not None, repeats=self.groups is not None
            )

    def _coordinates(self, angles, *, mirror=False):
        if self.cuda:
            return self._cuda_coordinates(angles, mirror=mirror)
        xp, grid, detector = self.xp, self.grid, self.detector
        px, py = self.shape
        fx = xp.fft.fftfreq(px, d=grid.vx)[:, None]
        # Retain the full transform's negative y Nyquist frequency, including
        # the opposite-y samples needed at the x boundary of an anisotropic grid.
        fy = xp.fft.fftfreq(py, d=grid.vy)[None, : py // 2 + 1]
        if mirror:
            stop = py // 2 if py % 2 == 0 else py // 2 + 1
            fx, fy = fx[px // 2 : px // 2 + 1], -fy[:, 1:stop]
        real = xp.float32 if self.cuda else xp.float64
        complex_dtype = xp.complex64 if self.cuda else xp.complex128
        radial = xp.sqrt(fx * fx + fy * fy) * (self.nfft * detector.du)
        direction = xp.mod(xp.arctan2(-fy, fx) - angles[0], 2 * np.pi)
        if self.uniform:
            angle = direction * (self.nviews / np.pi)
        else:
            # Interpolate in measured angle, not view index. Include the opposing
            # half turn and periodic endpoint, retaining FP64 angular fractions.
            knots = np.asarray(angles) - angles[0]
            knots = xp.asarray(np.concatenate((knots, knots + np.pi, [2 * np.pi])))
            angle = xp.interp(direction, knots, xp.arange(2 * self.nviews + 1))
        detector_phase = xp.exp(
            -2j * np.pi * radial * detector.center[0] / (self.nfft * detector.du)
        ).astype(complex_dtype)
        ox, oy, _ = grid_volume_origin(grid)
        phase = xp.exp(
            2j * np.pi * (fx * (ox - self.crop[0] * grid.vx) + fy * (oy - self.crop[1] * grid.vy))
        ).astype(complex_dtype)
        return angle, radial.astype(real), phase, detector_phase

    def _cuda_coordinates(self, angles, *, mirror):
        from ._fourier_cuda import coordinates_kernel

        xp, grid, detector = self.xp, self.grid, self.detector
        px, py = self.shape
        stop = py // 2 if py % 2 == 0 else py // 2 + 1
        shape = (1, stop - 1) if mirror else (px, py // 2 + 1)
        arrays = (
            xp.empty(shape, xp.float64),
            xp.empty(shape, xp.float32),
            xp.empty(shape, xp.complex64),
            xp.empty(shape, xp.complex64),
        )
        # Repeated source angles were combined before these coordinates were built.
        knots = np.asarray(angles) - angles[0]
        knots = xp.asarray(np.concatenate((knots, knots + np.pi, [2 * np.pi])))
        ox, oy, _ = grid_volume_origin(grid)
        coordinates_kernel()(
            (-(-arrays[0].size // 256),),
            (256,),
            (
                *arrays,
                knots,
                np.int32(self.nviews),
                np.int32(self.uniform),
                np.int32(px),
                np.int32(py),
                np.int32(mirror),
                np.float64(grid.vx),
                np.float64(grid.vy),
                np.float64(ox - self.crop[0] * grid.vx),
                np.float64(oy - self.crop[1] * grid.vy),
                np.float64(detector.du),
                np.float64(detector.center[0]),
                np.int32(self.nfft),
                np.float64(angles[0]),
                np.int64(arrays[0].size),
            ),
        )
        return arrays


@lru_cache(maxsize=8)
def _cached_geometry(grid, detector, nviews, flipped, angles, device) -> tuple[_Geometry, Any]:
    """Retain small immutable geometry arrays; no mutable slab buffers."""
    import cupy

    with cupy.cuda.Device(device):
        geometry = _Geometry(grid, detector, nviews, flipped, angles, "cupy")
        ready = cupy.cuda.Event()
        ready.record()
    return geometry, ready


_STREAMS_LOCK = Lock()


@lru_cache(maxsize=8)
def _cached_transfer_streams(device):
    """Reuse device queues so repeated calls reuse CuPy's stream-local arenas."""
    import cupy

    with cupy.cuda.Device(device):
        return tuple(cupy.cuda.Stream(non_blocking=True) for _ in range(3))


def _transfer_streams(device):
    # lru_cache alone can create duplicate queues on simultaneous first calls.
    with _STREAMS_LOCK:
        return _cached_transfer_streams(device)


@dataclass
class PendingSlab:
    output: np.ndarray
    event: Any
    owners: tuple[Any, ...]

    def wait(self) -> np.ndarray:
        self.event.synchronize()
        return self.output


class FourierPlan:
    def __init__(
        self,
        grid,
        detector,
        nviews,
        flipped,
        angles,
        depth,
        backend,
        *,
        async_transfers=False,
        transpose_download=False,
        shifts=None,
    ) -> None:
        self.cuda = backend == "cupy"
        if self.cuda:
            try:
                import cupy as xp
            except ImportError as exc:
                raise ImportError(
                    "fourier_reconstruct: backend='cupy' requires tomojax[fourier-cuda12]"
                ) from exc
        else:
            xp = np
        self.xp = xp
        self.transpose_download = transpose_download
        self.grid, self.detector = grid, detector
        self.source_views, self.depth = nviews, depth
        self.upload = self.compute = self.download = None
        if self.cuda and async_transfers:
            self.upload, self.compute, self.download = _transfer_streams(
                xp.cuda.runtime.getDevice()
            )
        # Construct geometry on the compute stream too, so its temporary
        # allocations can be reused by the FFT rather than held in another arena.
        with self.compute if self.compute is not None else nullcontext():
            shape, _ = transform_grid(grid, detector)
            if self.cuda and shape[0] * shape[1] <= 131072:
                geometry, ready = _cached_geometry(
                    grid,
                    detector,
                    nviews,
                    tuple(map(int, flipped)),
                    angles,
                    xp.cuda.runtime.getDevice(),
                )
                xp.cuda.get_current_stream().wait_event(ready)
            else:
                geometry = _Geometry(grid, detector, nviews, flipped, angles, backend)
            self.shifts = None if shifts is None else xp.asarray(shifts, dtype=xp.float32)
            self.nviews = geometry.nviews
            self.groups, self.device_groups = geometry.groups, geometry.device_groups
            self.nfft = geometry.nfft
            self.shape, self.crop = geometry.shape, geometry.crop
            self.flipped, self.period_phase = geometry.flipped, geometry.period_phase
            self.view_weights = geometry.view_weights
            self.projection_phase = geometry.projection_phase
            self.deapod, self.table = geometry.deapod, geometry.table
            self.coordinates = geometry.coordinates
            if self.cuda:
                self.kernel = geometry.kernel
                if self.shifts is not None:
                    from ._fourier_cuda import interpolation_kernel

                    self.kernel = interpolation_kernel(
                        x_boundary=geometry.x_boundary is not None,
                        repeats=self.groups is not None,
                        translated=True,
                    )
                self.mass_kernel = geometry.mass_kernel
                self.row_mass = xp.empty((depth, nviews), dtype=xp.float64)
            px, py = self.shape
            qy = py // 2 + 1
            dtype = xp.complex64 if self.cuda else xp.complex128
            self.frequencies = xp.empty((depth, px, qy), dtype=dtype)
            self.x_boundary = geometry.x_boundary
            if self.cuda:
                from cupy.cuda import cufft

                # Own this batched contiguous C2R workspace, not a global-cache
                # entry that could prevent a smaller retry from fitting. Calling
                # cuFFT directly also avoids importing SciPy's FFT adapter.
                cuda_fft = cast("Any", cufft)
                self.inverse_direction = cuda_fft.CUFFT_INVERSE
                self.inverse_plan = cuda_fft.PlanNd(
                    self.shape,
                    (px, qy),
                    1,
                    px * qy,
                    self.shape,
                    1,
                    px * py,
                    cuda_fft.CUFFT_C2R,
                    depth,
                    "C",
                    2,
                    py,
                )

    def _interpolate_cuda(self, spectrum):
        output = self.frequencies
        angle, radial, phase, detector_phase = self.coordinates
        layers = min(self.depth, 65535)
        args = (
            spectrum,
            angle,
            radial,
            phase,
            output,
            np.int64(output.shape[1] * output.shape[2]),
            np.int32(self.depth),
            np.int32(self.nviews),
            np.int32(self.source_views),
            self.device_groups,
            np.int32(spectrum.shape[-1]),
            np.int32(self.nfft),
            np.float32(self.period_phase.real),
            np.float32(self.period_phase.imag),
            self.table,
            detector_phase,
            self.flipped,
            self.row_mass,
            self.view_weights,
            np.float64(self.detector.du),
            np.int32(1),
        )
        if self.shifts is not None:
            args += (self.shifts,)
        if self.x_boundary is not None:
            coordinates, stop = self.x_boundary
            args += (
                *coordinates,
                np.int64(self.shape[0] // 2 * output.shape[2]),
                np.int32(stop),
            )
        self.kernel(
            (
                (output.shape[1] * output.shape[2] + 255) // 256,
                layers,
                (self.depth + layers - 1) // layers,
            ),
            (256,),
            args,
        )

    def _compute_volume(self, raw, count: int) -> Any:
        xp = self.xp
        if self.cuda:
            rows = self.depth * self.source_views
            self.mass_kernel(
                ((rows + 7) // 8,),
                (256,),
                (raw, self.deapod, self.row_mass, np.int32(self.detector.nu), np.int64(rows)),
            )
        else:
            mass = (xp.sum(raw, axis=-1, dtype=xp.float64) @ self.view_weights) * self.detector.du
            raw *= self.deapod
        spectrum = xp.fft.rfft(raw, n=self.nfft, axis=-1)
        xp.multiply(spectrum, self.projection_phase, out=spectrum)
        if self.cuda:
            self._interpolate_cuda(spectrum)
        else:
            self.frequencies = interpolate_numpy(
                spectrum,
                *self.coordinates,
                self.flipped,
                self.table,
                self.nfft,
                self.period_phase,
                self.groups,
                self.shifts,
            )
            self.frequencies[:, 0, 0] = mass
            if self.x_boundary is not None:
                coordinates, stop = self.x_boundary
                mirror = interpolate_numpy(
                    spectrum,
                    *coordinates,
                    self.flipped,
                    self.table,
                    self.nfft,
                    self.period_phase,
                    self.groups,
                    self.shifts,
                )
                row = self.frequencies[:, self.shape[0] // 2, 1:stop]
                row[:] = 0.5 * (row + mirror[:, 0, :].conj())
        # Interpolation has consumed the projection FFT. Its storage can now
        # serve the Cartesian FFT on this same compute stream.
        del spectrum
        cx, cy = self.crop
        normalization = 1
        if self.cuda:
            # cuFFT may destroy the input; the next slab overwrites every
            # frequency. Its inverse is unnormalized: normalize only the crop.
            volume = xp.empty((self.depth, *self.shape), dtype=xp.float32)
            self.inverse_plan.fft(self.frequencies, volume, self.inverse_direction)
            normalization = self.shape[0] * self.shape[1]
        else:
            volume = xp.fft.irfft2(self.frequencies, s=self.shape, axes=(-2, -1))
        # Keep fixed FFT batches, but do not materialize or transfer padded output.
        # Dense scalar arithmetic needs no sparse-operand dispatch or its imports.
        scale = 1.0 / (self.grid.vx * self.grid.vy * normalization)
        output = xp.multiply(volume[:count, cx : cx + self.grid.nx, cy : cy + self.grid.ny], scale)
        # Match plane-contiguous Fortran xyz storage before the device-to-host copy.
        return xp.ascontiguousarray(output.swapaxes(1, 2)) if self.transpose_download else output

    def reconstruct(self, host, *, count: int) -> np.ndarray:
        xp = self.xp
        with self.compute if self.compute is not None else nullcontext():
            # NumPy must also copy: deapodization must not mutate caller storage.
            if isinstance(host, ProjectionRows):
                source = xp.asarray(host.samples)
                raw, _owners = interpolate_projection_rows(host, source, self.depth, xp)
                raw = raw.astype(self.deapod.dtype, copy=False)
            else:
                raw = xp.array(host, dtype=self.deapod.dtype, copy=True)
            volume = self._compute_volume(raw, count)
            return xp.asnumpy(volume) if self.cuda else volume.astype(np.float32)

    def enqueue(self, host, *, count: int) -> PendingSlab:
        xp = self.xp
        assert self.upload is not None
        assert self.compute is not None
        assert self.download is not None
        samples = host.samples if isinstance(host, ProjectionRows) else host
        host_storage = xp.cuda.alloc_pinned_memory(samples.size * 4)
        staged = np.ndarray(samples.shape, dtype=np.float32, buffer=host_storage)
        np.copyto(staged, samples)
        try:
            with self.upload:
                source = xp.asarray(staged)
                uploaded = xp.cuda.Event()
                uploaded.record()
            self.compute.wait_event(uploaded)
            with self.compute:
                owners = ()
                raw = source
                if isinstance(host, ProjectionRows):
                    raw, owners = interpolate_projection_rows(host, source, self.depth, xp)
                volume = self._compute_volume(raw, count)
                computed = xp.cuda.Event()
                computed.record()
            output_storage = xp.cuda.alloc_pinned_memory(volume.size * 4)
            output = np.ndarray(volume.shape, dtype=np.float32, buffer=output_storage)
            self.download.wait_event(computed)
            with self.download:
                volume.get(out=output, stream=self.download, blocking=False)
                downloaded = xp.cuda.Event()
                downloaded.record()
            # Keep the asynchronous upload input and output device storage alive
            # until the final copy event, including every cross-stream use.
            return PendingSlab(
                output,
                downloaded,
                (
                    host,
                    host_storage,
                    staged,
                    source,
                    raw,
                    volume,
                    output_storage,
                    uploaded,
                    computed,
                    *owners,
                ),
            )
        except BaseException:
            self.upload.synchronize()
            self.compute.synchronize()
            self.download.synchronize()
            raise

    def enqueue_rows(self, data, order, positions) -> PendingSlab:
        """Borrow device source rows, making an owned FFT slab and output download."""
        import jax

        xp: Any = self.xp
        assert self.compute is not None
        assert self.download is not None
        try:
            with self.compute:
                if isinstance(data, jax.Array) and all(
                    device.platform == "gpu" for device in data.devices()
                ):
                    source = xp.from_dlpack(data)
                else:
                    source = xp.asarray(np.asarray(data))
                source = xp.ascontiguousarray(xp.take(source.transpose(1, 0, 2), order, axis=1))
                positions = np.clip(positions, -1, data.shape[1])
                indices = np.floor(positions).astype(np.int64)
                fraction = (positions - indices).astype(np.float32)
                lower = np.where((indices >= 0) & (indices < data.shape[1]), indices, -1)
                upper = np.where(
                    (indices + 1 >= 0) & (indices + 1 < data.shape[1]) & (fraction != 0),
                    indices + 1,
                    -1,
                )
                rows = ProjectionRows(source, lower, upper, fraction)
                raw, metadata = interpolate_projection_rows(rows, source, self.depth, xp)
                volume = self._compute_volume(raw, len(positions))
                computed = xp.cuda.Event()
                computed.record()
            storage = xp.cuda.alloc_pinned_memory(volume.size * 4)
            host = np.ndarray(volume.shape, np.float32, buffer=storage)
            self.download.wait_event(computed)
            with self.download:
                volume.get(out=host, stream=self.download, blocking=False)
                downloaded = xp.cuda.Event()
                downloaded.record()
            return PendingSlab(
                host, downloaded, (data, source, raw, volume, storage, computed, *metadata)
            )
        except BaseException:
            self.synchronize()
            raise

    def synchronize(self) -> None:
        """Drain every queue before releasing borrowed producers or retrying a slab."""
        for stream in (self.upload, self.compute, self.download):
            if stream is not None:
                stream.synchronize()
