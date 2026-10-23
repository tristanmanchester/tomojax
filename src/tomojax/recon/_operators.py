"""The forward projector and its exact transpose, for any geometry."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.devices import view_split
from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry import ParallelGeometry
from tomojax.geometry.api import detector_grid_from_geometry_inputs
from tomojax.recon._projection import projection_operators, resolve_geometry_projector

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
    """The grid, the split of the views among ``devices`` and the operators on them."""
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
    operators = projection_operators(
        poses, grid, detector, det_grid, backend, batch, model, split=split
    )
    return grid, split, operators


class Projector:
    """A fixed-geometry projector and its exact transpose.

    Construct once and call ``operator(volume)`` for forward projection or
    ``operator.adjoint(projections)`` for the transpose. Geometry and grid are
    snapshotted at construction. Each direction compiles on first use and
    reuses its compiled program on subsequent calls. Both directions support
    JAX transformations and differentiation.

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
        Expected volume shape, ``(nx, ny, nz)``.
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
        self.grid, self._split, (forward, adjoint) = matched_operators(
            self.geometry, grid, "Projector", devices
        )
        self.volume_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        self.projection_shape = (
            len(self.geometry.angles),
            self.geometry.detector.nv,
            self.geometry.detector.nu,
        )
        # Compile the complete batched loop, not individual operations, so its
        # output buffer is updated in place rather than duplicated per batch.
        self._forward = jax.jit(forward)
        self._adjoint = jax.jit(adjoint)

    def __call__(self, volume: jax.Array | np.ndarray) -> jax.Array:
        """Project a volume with shape ``volume_shape`` into line integrals.

        Parameters
        ----------
        volume : numpy.ndarray or jax.Array
            Volume in reciprocal geometry-length units, shape ``(nx, ny, nz)``.

        Returns
        -------
        jax.Array
            Line integrals with shape ``(views, nv, nu)``.

        Raises
        ------
        ValueError
            If the volume shape does not match the grid.
        """
        self._validate(volume, self.volume_shape, "project: volume")
        projections = self._forward(jnp.asarray(volume, jnp.float32))
        return projections if self._split is None else self._split.take(projections)

    def adjoint(self, projections: jax.Array | np.ndarray) -> jax.Array:
        """Apply the exact transpose, not an approximate reconstruction.

        Parameters
        ----------
        projections : numpy.ndarray or jax.Array
            Line integrals with shape ``(views, nv, nu)``.

        Returns
        -------
        jax.Array
            Adjoint volume with shape ``(nx, ny, nz)``.

        Raises
        ------
        ValueError
            If the projection shape does not match the geometry.
        """
        self._validate(projections, self.projection_shape, "backproject: projections")
        if self._split is None:
            return self._adjoint(jnp.asarray(projections, jnp.float32))
        return self._split.gather(self._adjoint(self._split.place(projections)))

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

    ``volume`` is ``(nx, ny, nz)`` on ``grid`` (the geometry's own by default),
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
        Volume of shape ``(nx, ny, nz)`` in reciprocal geometry-length units.
    grid : Grid or None, optional
        Volume grid. None uses geometry.grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views. None uses the current default device.

    Returns
    -------
    jax.Array
        Line integrals of shape ``(views, nv, nu)``.

    Raises
    ------
    ValueError
        If the array shape does not match the grid or scan geometry.
    """
    return Projector(geometry, grid=grid, devices=devices)(volume)


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
        Projection stack of shape ``(views, nv, nu)``.
    grid : Grid or None, optional
        Volume grid. None uses geometry.grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views. None uses the current default device.

    Returns
    -------
    jax.Array
        Adjoint volume of shape ``(nx, ny, nz)``, not a reconstruction.

    Raises
    ------
    ValueError
        If the array shape does not match the grid or scan geometry.
    """
    return Projector(geometry, grid=grid, devices=devices).adjoint(projections)


__all__ = ["Projector", "backproject", "project"]
