"""The forward projector and its exact transpose, for any geometry."""

from __future__ import annotations

import copy
import functools
import math
import os
import pickle
from typing import TYPE_CHECKING

import jax
from jax.core import Tracer
import jax.numpy as jnp

from tomojax.backends import device_free_memory_bytes
from tomojax.core.devices import as_devices, view_split
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.joseph import (
    PrivateJoseph,
    axial_backproject,
    plane_coefficients,
    stages_bands,
)
from tomojax.core.volume_layout import transpose_volume
from tomojax.geometry import ParallelGeometry
from tomojax.geometry.api import detector_grid_from_geometry_inputs
from tomojax.recon._projection import projection_operators, resolve_geometry_projector
from tomojax.recon._volume_storage import resident_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from tomojax._typed_arrays import Device
    from tomojax.geometry import Detector, Grid, ScanGeometry

_BATCH = 32
_AXIAL_STACK_BYTES = 64 * 1024**2


def _view_batch_size(
    grid: Grid,
    detector: Detector,
    n_views: int,
    *,
    axial_cuda: bool,
) -> int:
    """Use one axial CUDA batch only when its whole workspace is bounded.

    Count both the detector images and the vertically resampled stack. Larger
    problems retain the existing cap rather than introduce overlapping large
    tails; CPU and non-axial projectors retain their existing batching too.
    """
    batch = max(1, min(_BATCH, n_views))
    workspace = 4 * n_views * detector.nu * (detector.nv + grid.nz)
    if axial_cuda and workspace <= _AXIAL_STACK_BYTES:
        batch = max(1, n_views)
    return batch


def matched_operators(
    geometry: ScanGeometry,
    grid: Grid | None,
    context: str,
    devices: Device | Sequence[Device] | None = None,
):
    """The grid, the split of the views among ``devices`` and builders of their operators.

    The last item builds the same operators on solver-private z-fast storage
    for a single-device CUDA Joseph projector, else is None. Building computes
    the views' plane coefficients, so callers build only what they use.
    """
    grid = geometry.grid if grid is None else grid
    detector = geometry.detector
    det_grid = detector_grid_from_geometry_inputs(detector, geometry)
    model, backend = resolve_geometry_projector(
        geometry, "auto", "auto", detector=detector, det_grid=det_grid, context=context
    )
    n_views = len(geometry.angles)
    poses = stack_view_poses(geometry, n_views)
    axial_cuda = backend == "cuda" and model == "joseph" and isinstance(geometry, ParallelGeometry)
    batch = _view_batch_size(grid, detector, n_views, axial_cuda=axial_cuda)
    split = view_split(devices, n_views)
    build = functools.partial(projection_operators, poses, grid, detector, det_grid, backend, batch)
    private = resident_model(model, backend)
    if split is not None or not isinstance(private, PrivateJoseph):
        return grid, split, functools.partial(build, model, split=split), None
    return grid, split, functools.partial(build, model), functools.partial(build, private)


class Projector:
    """A fixed-geometry projector and its exact transpose.

    Construct once and call ``operator(volume)`` for forward projection or
    ``operator.adjoint(projections)`` for the transpose. Geometry and grid are
    snapshotted at construction. Each direction compiles on first use and
    reuses its compiled program on subsequent calls. Both directions support
    JAX transformations and differentiation.

    When ``grid.nz == detector.nv == 1``, a volume can also be supplied as
    ``(ny, nx)``: its projections have shape ``(views, nu)``. Conversely,
    ``adjoint`` maps that 2D sinogram to ``(ny, nx)``. Explicit 3D inputs
    preserve their singleton axes; these are the same matched operators.

    Parameters
    ----------
    geometry : ScanGeometry
        Parallel, laminography or cone-beam geometry, including per-view poses.
    grid : Grid or None, optional
        Volume grid; None uses the geometry's grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views. None uses the current default device.

    Attributes
    ----------
    geometry : ScanGeometry
        Snapshot of the supplied scan geometry.
    grid : Grid
        The volume grid used by both directions.
    volume_shape : tuple of int
        Expected volume shape, ``(nz, ny, nx)``.
    projection_shape : tuple of int
        Expected projection shape, ``(views, nv, nu)``.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> operator = tj.Projector(scan.geometry)  # doctest: +SKIP
    >>> simulated = operator(tj.reconstruct(scan).volume)  # doctest: +SKIP
    >>> transpose = operator.adjoint(scan.projections)  # doctest: +SKIP
    """

    def __init__(
        self,
        geometry: ScanGeometry,
        *,
        grid: Grid | None = None,
        devices: Device | Sequence[Device] | None = None,
    ) -> None:
        self.geometry = copy.deepcopy(geometry)
        self.grid, self._split, self._build, self._build_private = matched_operators(
            self.geometry, grid, "Projector", devices
        )
        self.volume_shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        self.projection_shape = (
            len(self.geometry.angles),
            self.geometry.detector.nv,
            self.geometry.detector.nu,
        )
        # Compile the complete batched loop, not individual operations, so its
        # output buffer is updated in place rather than duplicated per batch.
        self._forward = jax.jit(lambda x: self._operators[0](x))
        self._adjoint = jax.jit(lambda y: self._operators[1](y))
        self._single_slice = self.grid.nz == self.geometry.detector.nv == 1

    @functools.cached_property
    def _operators(self):
        """The forward and transpose on public (z, y, x) volumes, built on first use."""
        with jax.ensure_compile_time_eval():  # the geometry is concrete inside any trace
            return self._build()

    @functools.cached_property
    def _private_operators(self):
        """The same pair on solver-private z-fast volumes, built on first use."""
        assert self._build_private is not None
        with jax.ensure_compile_time_eval():
            return self._build_private()

    def __call__(self, volume: jax.Array | np.ndarray) -> jax.Array:
        """Project a volume with shape ``volume_shape`` into line integrals.

        Parameters
        ----------
        volume : numpy.ndarray or jax.Array
            Volume in reciprocal geometry-length units, shape ``(nz, ny, nx)``
            or ``(ny, nx)`` for a single slice.

        Returns
        -------
        jax.Array
            Line integrals with shape ``(views, nv, nu)``, or ``(views, nu)``
            for a squeezed single slice.

        Raises
        ------
        ValueError
            If the volume shape does not match the grid.
        """
        squeezed = self._single_slice and volume.ndim == 2
        if squeezed:
            volume = volume[None, :, :]
        self._validate(volume, self.volume_shape, "project: volume")
        volume = jnp.asarray(volume, jnp.float32)
        private = self._private_call(volume, 0, volume.size + math.prod(self.projection_shape))
        projections = (private or self._forward)(volume)
        if self._split is not None:
            projections = self._split.take(projections)
        return projections[:, 0, :] if squeezed else projections

    def adjoint(self, projections: jax.Array | np.ndarray) -> jax.Array:
        """Apply the exact transpose, not an approximate reconstruction.

        Parameters
        ----------
        projections : numpy.ndarray or jax.Array
            Line integrals with shape ``(views, nv, nu)``, or ``(views, nu)``
            for a squeezed single slice.

        Returns
        -------
        jax.Array
            Adjoint volume with shape ``(nz, ny, nx)``, or ``(ny, nx)``
            for a squeezed single-slice input.

        Raises
        ------
        ValueError
            If the projection shape does not match the geometry.
        """
        squeezed = self._single_slice and projections.ndim == 2
        if squeezed:
            projections = projections[:, None, :]
        self._validate(projections, self.projection_shape, "backproject: projections")
        if self._split is None:
            projections = jnp.asarray(projections, jnp.float32)
            elements = 2 * math.prod(self.volume_shape)
            adjoint = self._band_adjoint or self._private_call(projections, 1, elements)
            volume = (adjoint or self._adjoint)(projections)
        else:
            volume = self._split.gather(self._adjoint(self._split.place(projections)))
        return volume[0] if squeezed else volume

    @functools.cached_property
    def _coefficients(self):
        with jax.ensure_compile_time_eval():  # the geometry is concrete inside any trace
            poses = stack_view_poses(self.geometry, self.projection_shape[0])
            return plane_coefficients(poses, self.grid, self.geometry.detector)

    @functools.cached_property
    def _bands(self) -> tuple[bool, bool]:
        """Whether the forward and the transpose of these views stage their own z-fast bands."""
        coefficients, detector = self._coefficients, self.geometry.detector
        with jax.ensure_compile_time_eval():
            forward = stages_bands(coefficients, self.grid, detector)
            return forward, stages_bands(coefficients, self.grid, detector, adjoint=True)

    # One more volume in z-fast storage lets the kernels read z contiguously;
    # calls take that copy when it fits, with the same result bits either way,
    # except for views whose kernels stage their own z-fast bands.
    @functools.cached_property
    def _private_forward(self):
        if self._bands[0]:
            return None
        return jax.jit(lambda x: self._private_operators[0](transpose_volume(x)))

    @functools.cached_property
    def _private_adjoint(self):
        return jax.jit(lambda y: transpose_volume(self._private_operators[1](y)))

    @functools.cached_property
    def _band_adjoint(self):
        """The transpose of axial views, in bands, batch by batch as the operators add them."""
        if self._build_private is None or not self._bands[1]:
            return None
        views, parallel = self.projection_shape[0], isinstance(self.geometry, ParallelGeometry)
        batch = _view_batch_size(self.grid, self.geometry.detector, views, axial_cuda=parallel)
        return jax.jit(lambda y: axial_backproject(self._coefficients, y, self.grid, batch=batch))

    def _private_call(self, values: jax.Array, direction: int, elements: int):
        """The private-storage call for ``direction`` when ``elements`` more floats fit.

        Traced calls read the public array in place: free memory at trace time
        says nothing about the memory when the compiled program runs.
        """
        if self._build_private is None or isinstance(values, Tracer):
            return None
        free = device_free_memory_bytes()
        if free is None or 4 * elements > free:
            return None
        return self._private_adjoint if direction else self._private_forward

    @staticmethod
    def _validate(values, shape: tuple[int, int, int], context: str) -> None:
        if tuple(values.shape) != shape:
            raise ValueError(f"{context} shape {tuple(values.shape)} does not match {shape}")


def project(
    geometry: ScanGeometry,
    volume: jax.Array | np.ndarray,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> jax.Array:
    """Project a volume through the scan geometry.

    ``volume`` is ``(nz, ny, nx)`` on ``grid`` (the geometry's own by default),
    in the reciprocal of the geometry's length unit. Works for parallel,
    laminography and cone-beam geometries, with per-view poses and a rolled
    detector; :func:`backproject` is its exact transpose. ``devices`` (one or
    several, ``jax.devices()`` for all) shares the views among them, each
    holding the whole volume; the projections come back on the first.

    Parameters
    ----------
    geometry : ScanGeometry
        Parallel, laminography or cone-beam scan geometry, with any per-view poses.
    volume : array-like
        Volume of shape ``(nz, ny, nx)`` (single slice: ``(ny, nx)``),
        in reciprocal geometry-length units.
    grid : Grid or None, optional
        Volume grid. None uses geometry.grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views. None uses the current default device.

    Returns
    -------
    jax.Array
        Line integrals of shape ``(views, nv, nu)`` (single slice: ``(views, nu)``).

    Raises
    ------
    ValueError
        If the array shape does not match the grid or scan geometry.
    """
    return _projector(geometry, grid, devices, volume)(volume)


def backproject(
    geometry: ScanGeometry,
    projections: jax.Array | np.ndarray,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> jax.Array:
    """Apply the exact transpose of the forward projector.

    This is the exact adjoint, not a reconstruction: use
    :func:`tomojax.reconstruct` for a volume. ``devices`` shares the views
    among them, as for :func:`project`, and sums their backprojections.

    Parameters
    ----------
    geometry : ScanGeometry
        Parallel, laminography or cone-beam scan geometry, with any per-view poses.
    projections : array-like
        Projection stack of shape ``(views, nv, nu)`` (single slice: ``(views, nu)``).
    grid : Grid or None, optional
        Volume grid. None uses geometry.grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views. None uses the current default device.

    Returns
    -------
    jax.Array
        Adjoint volume of shape ``(nz, ny, nx)`` (single slice: ``(ny, nx)``),
        not a reconstruction.

    Raises
    ------
    ValueError
        If the array shape does not match the grid or scan geometry.
    """
    return _projector(geometry, grid, devices, projections).adjoint(projections)


def _projector(
    geometry: ScanGeometry,
    grid: Grid | None,
    devices: Device | Sequence[Device] | None,
    values: jax.Array | np.ndarray,
) -> Projector:
    """A Projector for these arguments, shared by calls with equal ones.

    A Projector compiles its programs on first use, so repeated calls reuse
    them instead of compiling again. The key holds everything construction
    reads: the arguments, the default backend and device, and the TOMOJAX_
    settings. A call inside a trace builds one in that trace, and a geometry
    that cannot be pickled gets one of its own; neither is shared.
    """
    if isinstance(values, Tracer):
        return Projector(geometry, grid=grid, devices=devices)
    try:
        snapshot = pickle.dumps((geometry, grid))
    except (pickle.PicklingError, AttributeError, TypeError):
        return Projector(geometry, grid=grid, devices=devices)
    settings = tuple(sorted(item for item in os.environ.items() if item[0].startswith("TOMOJAX_")))
    placement = jax.default_backend(), jax.config.jax_default_device
    return _shared_projector(snapshot, as_devices(devices), placement, settings)


@functools.lru_cache(maxsize=4)
def _shared_projector(
    snapshot: bytes, devices: tuple[Device, ...] | None, *_environment: object
) -> Projector:
    geometry, grid = pickle.loads(snapshot)
    return Projector(geometry, grid=grid, devices=devices)


__all__ = ["Projector", "backproject", "project"]
