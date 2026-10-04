"""Slab FFT execution; CuPy is imported only for an explicit CUDA request."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

from tomojax.geometry import grid_volume_origin

from ._fourier_grid import interpolate_numpy, radial_coefficients, transform_grid


class _Geometry:
    def __init__(self, grid, detector, nviews, flipped, theta0, backend) -> None:
        self.cuda = backend == "cupy"
        if self.cuda:
            try:
                import cupy as xp
            except ImportError as exc:
                raise ImportError(
                    "fourier_reconstruct: backend='cupy' requires tomojax[fourier-cuda12]"
                ) from exc
            from ._fourier_cuda import interpolation_kernel

            self.kernel = interpolation_kernel()
        else:
            xp = np
        self.xp = xp
        self.grid, self.detector = grid, detector
        self.nviews = nviews
        self.nfft = max(64, next_fast_len(2 * detector.nu, real=True))
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
        qy = py // 2 + 1 if self.cuda else py
        fx = xp.fft.fftfreq(px, d=grid.vx)[:, None]
        # Retain the full transform's negative y Nyquist frequency. The real
        # inverse FFT supplies the same Hermitian projection as real(ifft2).
        fy = xp.fft.fftfreq(py, d=grid.vy)[None, :qy]
        self.coordinates = self._coordinates(fx, fy, theta0)
        self.x_boundary = None
        stop = qy - 1 if py % 2 == 0 else qy
        if self.cuda and px % 2 == 0 and grid.vx > detector.du and stop > 1:
            # At x Nyquist, -kx maps to the same stored bin. If its line is
            # inside the detector's disk, conjugate symmetry needs the missing
            # opposite-y sample, not just conjugation of the current sample.
            coordinates = self._coordinates(fx[px // 2 : px // 2 + 1], -fy[:, 1:stop], theta0)
            self.x_boundary = (coordinates, stop)

    def _coordinates(self, fx, fy, theta0):
        xp, grid, detector = self.xp, self.grid, self.detector
        real = xp.float32 if self.cuda else xp.float64
        complex_dtype = xp.complex64 if self.cuda else xp.complex128
        radial = xp.sqrt(fx * fx + fy * fy) * (self.nfft * detector.du)
        angle = xp.mod(xp.arctan2(-fy, fx) - theta0, 2 * np.pi) * (self.nviews / np.pi)
        detector_phase = xp.exp(
            -2j * np.pi * radial * detector.det_center[0] / (self.nfft * detector.du)
        ).astype(complex_dtype)
        ox, oy, _ = grid_volume_origin(grid)
        phase = xp.exp(
            2j * np.pi * (fx * (ox - self.crop[0] * grid.vx) + fy * (oy - self.crop[1] * grid.vy))
        ).astype(complex_dtype)
        return angle, radial.astype(real), phase, detector_phase


@lru_cache(maxsize=8)
def _cached_geometry(grid, detector, nviews, flipped, theta0, device) -> tuple[_Geometry, Any]:
    """Retain small immutable geometry arrays; no mutable slab buffers."""
    import cupy

    with cupy.cuda.Device(device):
        geometry = _Geometry(grid, detector, nviews, flipped, theta0, "cupy")
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
class _PendingSlab:
    output: np.ndarray
    event: Any
    owners: tuple[Any, ...]

    def wait(self) -> np.ndarray:
        self.event.synchronize()
        return self.output


class FourierPlan:
    def __init__(
        self, grid, detector, nviews, flipped, theta0, depth, backend, *, async_transfers=False
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
        self.grid, self.detector = grid, detector
        self.nviews, self.depth = nviews, depth
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
                    theta0,
                    xp.cuda.runtime.getDevice(),
                )
                xp.cuda.get_current_stream().wait_event(ready)
            else:
                geometry = _Geometry(grid, detector, nviews, flipped, theta0, backend)
            self.nfft = geometry.nfft
            self.shape, self.crop = geometry.shape, geometry.crop
            self.flipped, self.period_phase = geometry.flipped, geometry.period_phase
            self.projection_phase = geometry.projection_phase
            self.deapod, self.table = geometry.deapod, geometry.table
            self.coordinates = geometry.coordinates
            if self.cuda:
                self.kernel = geometry.kernel
            px, py = self.shape
            qy = py // 2 + 1 if self.cuda else py
            dtype = xp.complex64 if self.cuda else xp.complex128
            self.frequencies = xp.empty((depth, px, qy), dtype=dtype)
            self.x_boundary = None
            if geometry.x_boundary is not None:
                coordinates, stop = geometry.x_boundary
                self.x_boundary = (coordinates, xp.empty((depth, 1, stop - 1), dtype=dtype), stop)

    def _interpolate_cuda(self, spectrum, coordinates, output):
        angle, radial, phase, detector_phase = coordinates
        self.kernel(
            ((output.size + 255) // 256,),
            (256,),
            (
                spectrum,
                angle,
                radial,
                phase,
                output,
                np.int64(output.shape[1] * output.shape[2]),
                np.int32(self.depth),
                np.int32(self.nviews),
                np.int32(spectrum.shape[-1]),
                np.int32(self.nfft),
                np.float32(self.period_phase.real),
                np.float32(self.period_phase.imag),
                self.table,
                detector_phase,
                self.flipped,
            ),
        )

    def _compute_volume(self, raw):
        xp = self.xp
        mass = xp.mean(xp.sum(raw, axis=-1, dtype=xp.float64), axis=1) * self.detector.du
        raw *= self.deapod
        spectrum = xp.fft.rfft(raw, n=self.nfft, axis=-1)
        spectrum *= self.projection_phase
        if self.cuda:
            self._interpolate_cuda(spectrum, self.coordinates, self.frequencies)
            if self.x_boundary is not None:
                coordinates, mirror, stop = self.x_boundary
                self._interpolate_cuda(spectrum, coordinates, mirror)
                row = self.frequencies[:, self.shape[0] // 2, 1:stop]
                row[:] = 0.5 * (row + mirror[:, 0, :].conj())
        else:
            self.frequencies = interpolate_numpy(
                spectrum,
                *self.coordinates,
                self.flipped,
                self.table,
                self.nfft,
                self.period_phase,
            )
        self.frequencies[:, 0, 0] = mass
        cx, cy = self.crop
        volume = (
            xp.fft.irfft2(self.frequencies, s=self.shape, axes=(-2, -1))
            if self.cuda
            else xp.fft.ifft2(self.frequencies, axes=(-2, -1)).real
        )
        return volume[:, cx : cx + self.grid.nx, cy : cy + self.grid.ny] / (
            self.grid.vx * self.grid.vy
        )

    def reconstruct(self, host) -> np.ndarray:
        xp = self.xp
        with self.compute if self.compute is not None else nullcontext():
            # NumPy must also copy: deapodization must not mutate caller storage.
            raw = xp.array(host, dtype=self.deapod.dtype, copy=True)
            volume = self._compute_volume(raw)
            return xp.asnumpy(volume) if self.cuda else volume.astype(np.float32)

    def enqueue(self, host) -> _PendingSlab:
        xp = self.xp
        assert self.upload is not None
        assert self.compute is not None
        assert self.download is not None
        host_storage = xp.cuda.alloc_pinned_memory(host.size * 4)
        staged = np.ndarray(host.shape, dtype=np.float32, buffer=host_storage)
        np.copyto(staged, host)
        try:
            with self.upload:
                raw = xp.asarray(staged)
                uploaded = xp.cuda.Event()
                uploaded.record()
            self.compute.wait_event(uploaded)
            with self.compute:
                volume = self._compute_volume(raw)
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
            return _PendingSlab(
                output,
                downloaded,
                (host_storage, staged, raw, volume, output_storage, uploaded, computed),
            )
        except BaseException:
            self.upload.synchronize()
            self.compute.synchronize()
            self.download.synchronize()
            raise
