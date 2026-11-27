"""Feldkamp-Davis-Kress (FDK) reconstruction for cone-beam scans.

Each projection is cosine-weighted, ramp-filtered along detector rows on the
isocenter plane, and backprojected voxel by voxel with the ``(SOD / U)^2``
distance weight, where U is the voxel's distance from the source along the
detector normal. Full turns share angular measure across conjugate views;
shorter arcs of at least a half turn plus the fan angle use Parker weights
with a virtual fan spanning the acquired arc, retaining overscan measurements.
On an offset detector, a ray whose conjugate falls off the panel keeps the
whole weight of its line.

FDK is exact only in the plane through the source orbit; away from it the
reconstruction carries the usual cone-beam artifacts, growing with the cone
angle. It is a fast first reconstruction and initializer for the iterative
solvers, which model the cone geometry exactly.
"""

from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
import math
import operator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import _parameters, progress
from tomojax._parameters import check_backend, normalize_filter
from tomojax.core import checkpoint
from tomojax.core.cone import ConeModel
from tomojax.core.devices import BatchRanges, announce, as_devices, sum_batches, view_split
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.host_transfer import copy_to_host, upload_host
from tomojax.core.validation import validate_projection_stack
from tomojax.recon._backend import reconstruction_backend
from tomojax.recon._continuation import Continuation, cone_axis_distance, continued_columns
from tomojax.recon._fdk_angles import (
    check_angular_coverage,
    column_weights,
    fan_angles,
    grouped_angles,
)
from tomojax.recon._fdk_cuda import backproject_cuda
from tomojax.recon._fdk_filter import (
    batch_cosines,
    continued_rows,
    cosines,
    filter_operator,
    grid_columns,
    measured_and_exterior,
)
from tomojax.recon._fdk_oblique import FanRows, FilteredChords, PermutedRows
from tomojax.recon._host_arrays import validate_host_arrays
from tomojax.recon._row_filter import filter_rows, filtered_profiles
from tomojax.recon.filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from tomojax._parameters import Filter, ProjectorBackend
    from tomojax._typed_arrays import Device
    from tomojax.core.geometry.base import Detector, Geometry, Grid
    from tomojax.core.geometry.cone import ConeBeam
    from tomojax.recon.types import ReconstructionSlab

    from ._fdk_oblique import Parts


# The CUDA path filters rows up to this wide by matrix product (see filter_operator):
# on an RTX 4070 that beats the FFTs 3x at 768 pixels and breaks even near 2500.
_DENSE_FILTER_MAX = 2048


@dataclass(frozen=True, kw_only=True)
class FDKConfig:
    """FDK options.

    ``filter`` is ``ramp``, ``shepp-logan`` or ``hann``. ``backend``
    ``auto`` uses the CUDA kernel when CuPy and a CUDA device are available.
    ``views_per_batch`` bounds the filtered views held on the device at once.
    ``devices`` (one or several JAX devices) shares the views among them: each
    filters and backprojects its own, weighted for the whole scan, and the
    volumes are summed on the first. :func:`fdk_host` gives each its own slabs.

    ``exterior="cylinder"`` (default) continues each detector row past both
    edges after angular weighting as a uniform cylinder about the rotation axis;
    ``exterior="zero"`` assumes zero attenuation beyond the detector.

    Parameters
    ----------
    filter : str, optional
        Frequency filter: "ramp", "shepp-logan" or "hann". Default is 'ramp'.
    backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". Auto uses CUDA when available.
        Default is 'auto'.
    views_per_batch : int, optional
        Maximum views processed together. Batches also cap FP32 input storage at
        64 MiB (or one view) to bound host transfer staging. Default is 64.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
    exterior : {"cylinder", "zero"}, optional
        Model beyond the detector: "cylinder" fits each row at its edges before
        filtering; "zero" assumes zero attenuation beyond the detector. Default
        is 'cylinder'.
    """

    filter: Filter = "ramp"
    backend: ProjectorBackend = "auto"
    views_per_batch: int = 64
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    exterior: _parameters.Exterior = _parameters.DEFAULT_EXTERIOR

    def __post_init__(self) -> None:
        _parameters.check_exterior(self.exterior)
        check_backend(self.backend, "fdk")
        object.__setattr__(self, "filter", normalize_filter(self.filter))
        object.__setattr__(self, "devices", as_devices(self.devices))


def view_weights(geometry: Geometry, detector: Detector, n_views: int) -> np.ndarray:
    """FDK angular weights on physical horizontal detector lines.

    On a row-aligned panel the result is (views, nu). A rolled panel has a
    different measured fan on each rectified row, giving (views, nv, nu) on
    the horizontal filter lattice.
    """
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("view_weights needs a cone-beam geometry")
    fan = FanRows.for_scan(geometry, parts, n_views)
    if fan is None:
        return column_weights(parts, grouped_angles(geometry, n_views))
    check_angular_coverage(geometry, detector, n_views)
    result, first = [], 0
    for index, count in enumerate(fan.counts):
        views = slice(first, first + count)
        result.append(
            np.stack(
                [fan.weights(index, row, views) for row in range(fan.parts[index][2].nv)], axis=1
            )
        )
        first += count
    return np.concatenate(result)


def _virtual_columns(beam: ConeBeam, detector: Detector) -> tuple[int, int]:
    """Columns to add below and above an offset detector, to its conjugate rays' reach.

    An offset detector reconstructs the whole field its long side covers, in a
    full turn or a short scan; voxels on the short side then project past the
    detector, where the filtered rows (zero data, nonzero ramp tail) must still
    be read. A column's conjugate has the opposite fan angle (see
    :func:`fan_angles`), which a laterally offset axis maps nonlinearly to u.
    """
    if not _windowable(beam):
        return 0, 0
    sdd, du = float(beam.source_to_detector), float(detector.du)
    gamma = fan_angles(beam, detector)
    tilt = np.arctan(float(beam.axis_offset) / float(beam.source_to_axis))
    # Column of fan angle g: u = SDD tan(g + tilt), clipped short of the 90-degree pole.
    reach = sdd * np.tan(np.clip(-gamma[[-1, 0]] + tilt, -1.5, 1.5))
    # Fractional column indices of the conjugates of the last and first columns.
    low, high = (reach - float(detector.center[0])) / du + (detector.nu - 1) / 2
    pad = (max(0, math.ceil(-low - 1e-6)), max(0, math.ceil(high - (detector.nu - 1) - 1e-6)))
    return (0, 0) if max(pad) <= 1 else pad


def _backproject_jax(
    filtered: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    scale: float,
    out: jax.Array,
) -> jax.Array:
    """Voxel-driven backprojection of filtered ``(views, nv, nu)`` images."""
    index = [jnp.arange(n, dtype=jnp.float32) for n in (grid.nz, grid.ny, grid.nx)]
    qz, qy, qx = jnp.meshgrid(*index, indexing="ij")

    def step(total: jax.Array, view: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        image, c = view
        d0, d1, d2 = qx - c[0], qy - c[1], qz - c[2]
        lam = c[16] / (c[13] * d0 + c[14] * d1 + c[15] * d2)
        u = c[23] + lam * (c[17] * d0 + c[18] * d1 + c[19] * d2)
        v = c[24] + lam * (c[20] * d0 + c[21] * d1 + c[22] * d2)
        u0, v0 = jnp.floor(u), jnp.floor(v)
        value = jnp.zeros_like(u)
        for du in (0.0, 1.0):
            for dv in (0.0, 1.0):
                ui, vi = u0 + du, v0 + dv
                w = (1 - jnp.abs(u - ui)) * (1 - jnp.abs(v - vi))
                valid = (ui >= 0) & (ui < detector.nu) & (vi >= 0) & (vi < detector.nv)
                sample = image[
                    jnp.clip(vi, 0, detector.nv - 1).astype(jnp.int32),
                    jnp.clip(ui, 0, detector.nu - 1).astype(jnp.int32),
                ]
                value = value + jnp.where(valid, sample * w, 0.0)
        return total + value * (lam * scale) ** 2, None

    total, _ = jax.lax.scan(step, out, (filtered, coeff))
    return total


@partial(
    jax.jit,
    static_argnames=("grid", "detector", "scale", "cuda", "pad_lo", "pad_hi", "distance"),
    donate_argnames=("out",),
)
def _fdk_batch(
    views: jax.Array,
    start: jax.Array | int,
    coeff: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array | None,
    operator: jax.Array | None,
    radius2: jax.Array | None,
    profiles: jax.Array | None,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
    pad_lo: int,
    pad_hi: int,
    distance: tuple[float, ...],
) -> jax.Array:
    # The batch's views of the scan's coefficients and weights.
    coeff = jax.lax.dynamic_slice_in_dim(coeff, start, views.shape[0])
    backproject = backproject_cuda if cuda else _backproject_jax
    if kernel is None:
        return backproject(views, coeff, grid, detector, scale, out)
    weights = jax.lax.dynamic_slice_in_dim(weights, start, views.shape[0])
    nu = views.shape[-1]
    wide = replace(detector, nu=len(distance) or nu)
    cosine = batch_cosines(cosine, start, views.shape[0], wide)
    if profiles is not None and radius2 is not None:
        rows, correction = measured_and_exterior(
            views, weights, cosine, radius2, distance, profiles
        )
        return backproject_cuda(rows, coeff, grid, detector, scale, out, operator, correction)
    rows = continued_rows(views, weights, cosine, radius2, distance)
    if operator is not None:
        return backproject_cuda(rows, coeff, grid, detector, scale, out, operator)
    filtered = filter_rows(rows, kernel, nu, pad_lo, pad_hi)
    return backproject(filtered, coeff, grid, detector, scale, out)


@jax.jit
def _virtual_coefficients(coeff: jax.Array, pad_lo: int) -> jax.Array:
    """Translate a detector corner and its projected column origin together."""
    coeff = coeff.at[:, 3:6].add(-pad_lo * coeff[:, 6:9])
    return coeff.at[:, 23].add(pad_lo)


@dataclass(frozen=True)
class _Prepared:
    """Per-scan FDK weights and filter on the device; see :func:`_prepare`."""

    cuda: bool
    detector: Detector
    weights: jax.Array
    cosine: jax.Array
    kernel: jax.Array
    scale: float
    views_per_batch: int
    pad_lo: int = 0
    pad_hi: int = 0
    # The filter as a matrix (see filter_operator), for the CUDA kernel's rows.
    operator: jax.Array | None = None
    # Distance from the rotation axis of each ray of a continued row.
    distance: tuple[float, ...] = ()

    fan: FanRows | None = None
    first_row: int = 0
    config: FDKConfig = field(default_factory=FDKConfig)

    @property
    def batch(self) -> int:
        """Bound transfers for the current row window, up to the configured view cap."""
        # Prefetch and transfer staging can retain several input buffers.
        # Size these for the measured rows, not the uncropped detector.
        mib = 8 if self.fan is not None and not self.fan.permuted else 64
        transfer_views = max(1, (mib * 2**20) // (4 * self.detector.nu * self.detector.nv))
        return max(1, min(self.views_per_batch, transfer_views))

    def coefficients(
        self, geometry: Geometry, grid: Grid, detector: Detector, poses: jax.Array
    ) -> jax.Array:
        parts = self.fan.parts if self.fan is not None else cone_parts(geometry, detector)
        assert parts is not None
        coeff = ConeModel(parts).coefficients(poses, grid)
        # Virtual columns translate the detector origin in its own plane, even
        # when its u axis has a y component (which Detector.center cannot hold).
        if self.pad_lo:
            coeff = _virtual_coefficients(coeff, self.pad_lo)
        return coeff

    def window(self, first: int, stop: int) -> _Prepared:
        """A physical row window, without moving any calibrated detector plane."""
        cosine = self.cosine
        if cosine.ndim == 2:
            cosine = cosine[first:stop]
        else:
            shift = first + (stop - first - 1) / 2 - (self.detector.nv - 1) / 2
            cosine = cosine.at[:, 1, 0].add(shift * self.detector.dv)
        return replace(
            self,
            detector=replace(self.detector, nv=stop - first),
            first_row=self.first_row + first,
            cosine=cosine,
        )

    def virtual(self, detector: Detector) -> Detector:
        """Pixel count of the filtered images; their placement is in coefficients."""
        return replace(detector, nu=detector.nu + self.pad_lo + self.pad_hi)

    def on(self, device: Device | None, views: range) -> _Prepared:
        """These weights for ``views`` alone, and the filter, on ``device``."""
        put = partial(jax.device_put, device=device)
        operator = None if self.operator is None else put(self.operator)
        weights = put(self.weights[views.start : views.stop])
        cosine = self.cosine if self.cosine.ndim == 2 else self.cosine[views.start : views.stop]
        return replace(
            self,
            weights=weights,
            cosine=put(cosine),
            kernel=put(self.kernel),
            operator=operator,
        )


def _prepare(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    poses: np.ndarray,
    cfg: FDKConfig,
) -> _Prepared:
    """FDK weights, filter and virtual columns for the complete scan."""
    n_views = len(poses)
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    cuda = reconstruction_backend(cfg.backend, cone=True, context="fdk") == "cuda"
    fan = FanRows.for_scan(geometry, parts, n_views)
    if fan is not None:
        check_angular_coverage(geometry, detector, n_views)
        parts = fan.parts
        detector = parts[0][2]
    magnification = parts[0][1].magnification
    du_iso = float(detector.du) / magnification
    padding = [_virtual_columns(beam, own) for _, beam, own in parts]
    if cfg.exterior == "cylinder":
        padding.append(grid_columns(parts, grid, poses))
    pad_lo, pad_hi = np.max(padding, axis=0).tolist()
    if fan is not None and not fan.permuted:
        empty = jnp.empty((0, 0), jnp.float32)
        return _Prepared(
            cuda,
            detector,
            empty,
            empty,
            empty,
            1 / magnification,
            cfg.views_per_batch,
            pad_lo,
            pad_hi,
            fan=fan,
            config=cfg,
        )
    # A shared detector-pitch ramp is linear in magnification. Fold each
    # view's remaining inverse magnification into its angular weight.
    weights = column_weights(parts, fan.angles if fan else grouped_angles(geometry, n_views))
    weights *= magnification / ConeModel(parts).magnification(n_views)[:, None]
    # The kernel spans the continued and the virtual row, so no column the
    # backprojection reads sees either wrap around.
    width = detector.nu + pad_lo + pad_hi
    pad = continued_columns(detector.nu) if cfg.exterior == "cylinder" else 0
    kernel = get_fbp_filter_np(
        cfg.filter, detector.nu + pad + max(pad_lo, pad_hi) if pad else width, du_iso, "float32"
    )
    operator = None
    if cuda and width <= _DENSE_FILTER_MAX:
        operator = jax.device_put(
            filter_operator(kernel, detector.nu, pad_lo, pad_hi, 0 if len(parts) == 1 else pad)
        )
    # Cosine weights of the continued columns too: the same detector, wider.
    continued = tuple(
        (count, beam, replace(own, nu=own.nu + 2 * pad)) for count, beam, own in parts
    )
    return _Prepared(
        cuda=cuda,
        fan=fan,
        detector=detector,
        # device_put, unlike jnp.asarray, compiles nothing for a new shape.
        weights=jax.device_put(np.asarray(weights, np.float32)),
        cosine=jax.device_put(cosines(continued, n_views)),
        kernel=jax.device_put(kernel),
        operator=operator,
        scale=1.0 / magnification,
        views_per_batch=int(cfg.views_per_batch),
        pad_lo=pad_lo,
        pad_hi=pad_hi,
        distance=tuple(cone_axis_distance(parts[0][1], parts[0][2], pad).tolist()) if pad else (),
    )


def _device_views(
    projections: jax.Array | np.ndarray | _RowView | PermutedRows,
    start: int,
    stop: int,
    device: Device,
) -> jax.Array:
    part = projections[start:stop]
    if isinstance(part, jax.Array):
        return jax.device_put(part.astype(jnp.float32), device)
    return upload_host(np.asarray(part, np.float32), device)


@progress.task("reconstruct", "fdk", None, "batches")
def fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: FDKConfig | None = None,
) -> jax.Array:
    """Reconstruct a cone-beam scan with FDK; returns an ``(nz, ny, nx)`` volume.

    ``projections`` are ``(views, nv, nu)`` line integrals; NumPy and memmap
    stacks stream to the device ``views_per_batch`` views at a time. Full turns
    on an offset detector (the axis projecting off its center) get Wang's
    weights from the conjugate views and reconstruct the field the detector's long side covers.
    Each view retains its own calibrated source/detector arrangement; angular
    quadrature covers the whole scan, independently of segment boundaries.
    Noncircular trajectories retain FDK's approximate character.

    With the default ``exterior="cylinder"``, FDK continues rows past both
    detector edges after applying angular weights and before filtering. An
    opposing view that measures an edge's rays takes their weight, so the edge
    continues as zero. Set ``FDKConfig(exterior="zero")`` to assume zero beyond
    the detector. FDK keeps filtered columns as far as the grid projects. Voxels
    outside the field of view hold the reconstruction of the continued scan.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    config : FDKConfig or None, optional
        Solver settings. None uses FDKConfig defaults.

    Returns
    -------
    jax.Array
        Float32 reconstruction of shape ``(nz, ny, nx)``.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    """
    cfg = FDKConfig() if config is None else config
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk projections"
    )
    poses = stack_view_poses(geometry, n_views)
    prep = _prepare(geometry, grid, detector, np.asarray(poses, np.float64), cfg)
    coeff = prep.coefficients(geometry, grid, detector, poses)
    result = _accumulate(grid, projections, prep, coeff, cfg.devices)
    return result.block_until_ready() if progress.for_unit("batches") is not None else result


def _accumulate(
    grid: Grid,
    projections: jax.Array | np.ndarray | _RowView | PermutedRows,
    prep: _Prepared,
    coeff: jax.Array,
    devices: Device | Sequence[Device] | None,
    native_row: int = 0,
) -> jax.Array:
    """Filter and sum validated views using coefficients in this volume's voxel frame."""
    n_views = int(coeff.shape[0])
    virtual = prep.virtual(prep.detector)
    if prep.fan is not None and prep.fan.permuted:
        projections = PermutedRows(
            prep.fan, projections, native_row, prep.first_row, prep.detector.nv
        )
    chords = (
        FilteredChords(
            prep.fan,
            projections,
            native_row,
            prep.first_row,
            prep.first_row + prep.detector.nv,
            prep.pad_lo,
            prep.pad_hi,
            prep.config.filter,
            cylinder=prep.config.exterior == "cylinder",
        )
        if prep.fan is not None and not prep.fan.permuted
        else None
    )
    pad = continued_columns(prep.detector.nu)
    radius2 = (
        Continuation(np.asarray(prep.distance), prep.detector.nu, pad).stack_radius(projections)
        if prep.distance
        else None
    )
    # Physical placement is already in coeff. Equal-sized slabs share the
    # filtering/backprojection executable, rather than compiling every origin.
    kernel_grid = replace(grid, vol_origin=None, vol_center=None)

    def backprojected(
        device: Device | None, views: range, groups: BatchRanges
    ) -> Generator[jax.Array]:
        """The weighted backprojection of ``views``, on ``device`` (JAX's default for None)."""
        here = prep.on(device, views)
        share = jax.device_put(coeff[views.start : views.stop], device)
        radius = jax.device_put(radius2, device)
        profiles = (
            filtered_profiles(
                radius, here.cosine, here.kernel, prep.distance, prep.pad_lo, prep.pad_hi
            )
            if radius is not None and here.operator is not None and here.cosine.ndim == 2
            else None
        )
        out = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32, device=device)
        target = next(iter(out.devices()))
        starts = list(range(0, len(views), prep.batch))

        def read(start: int) -> jax.Array:
            first = views.start + start
            stop = min(first + prep.batch, views.stop)
            if chords is not None:
                return chords.read(first, stop, target)
            return _device_views(projections, first, stop, target)

        # Copy the next batch to the device while the current one is filtered
        # and backprojected.
        with ThreadPoolExecutor(max_workers=1) as copier:
            pending = copier.submit(read, 0)
            for group in groups(prep.batch):
                for index in group:
                    start = starts[index]
                    batch = pending.result()
                    if index + 1 < len(starts):
                        pending = copier.submit(read, starts[index + 1])
                    with progress.preparing("dispatch cone backprojection", index):
                        out = _fdk_batch(
                            batch,
                            start,
                            share,
                            here.cosine,
                            here.weights,
                            here.kernel if chords is None else None,
                            here.operator,
                            radius,
                            profiles,
                            out,
                            grid=kernel_grid,
                            detector=virtual,
                            scale=prep.scale,
                            cuda=prep.cuda,
                            pad_lo=prep.pad_lo,
                            pad_hi=prep.pad_hi,
                            distance=prep.distance,
                        )
                    # Keep only this batch and its prefetched successor live.
                    out.block_until_ready()
                yield out

    split = view_split(devices, n_views)
    if split is None:
        announce()
    return sum_batches(split, n_views, prep.batch, backprojected)


@dataclass(frozen=True, kw_only=True)
class FDKHostConfig:
    """Slab options for :func:`fdk_host`.

    ``slices_per_batch`` z slices are reconstructed per slab; ``None`` sizes slabs
    to about a third of the free device memory. ``fdk`` holds the filter, backend,
    and exterior options.

    Parameters
    ----------
    slices_per_batch : int | None, optional
        Positive slab depth, or None to choose automatically as described above.
    fdk : FDKConfig, optional
        FDK filter, batching, backend, and exterior configuration.
    """

    slices_per_batch: int | None = None
    fdk: FDKConfig = field(default_factory=FDKConfig)


def _slab_rows(
    parts: Parts, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Detector rows that the z-slab ``[z0, z1)`` projects onto in any view."""
    origin = np.asarray(grid_volume_origin(grid))
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    lo = origin - spacing / 2
    hi = origin + (np.asarray([grid.nx, grid.ny, grid.nz]) - 0.5) * spacing
    zlo, zhi = origin[2] + (z0 - 1.5) * grid.vz, origin[2] + (z1 + 0.5) * grid.vz
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (zlo, zhi)]
    )
    # A box projects inside the hull of its projected corners.
    world = np.einsum("nij,kj->nki", poses[:, :3, :3], corners) + poses[:, None, :3, 3]
    low, high, start = math.inf, -math.inf, 0
    for count, beam, own in parts:
        stop = len(poses) if count is None else start + count
        center, u_dir, v_dir = beam.detector_frame(own)
        normal = np.cross(u_dir, v_dir)
        source = beam.source()
        ray = world[start:stop] - source
        hit = source + ray * (((center - source) @ normal) / (ray @ normal))[..., None]
        rows = ((hit - center) @ v_dir) / own.dv + (own.nv - 1) / 2
        low, high, start = min(low, float(rows.min())), max(high, float(rows.max())), stop
    # Keep one measured border row even when the slab is entirely outside.
    # Backprojection still samples zero beyond that row, as with the full
    # detector, without constructing an empty or negative-height detector.
    first = max(0, min(detector.nv - 1, math.floor(low) - 1))
    stop = max(first + 1, min(detector.nv, math.ceil(high) + 2))
    return first, stop


def _windowable(beam: ConeBeam) -> bool:
    """Whether a band of detector rows is itself a detector of ``beam`` (no pitch or yaw)."""
    return float(beam.detector_pitch) == 0.0 and float(beam.detector_yaw) == 0.0


def _slab_grid(grid: Grid, z0: int, z1: int) -> Grid:
    """Grid of z slices ``[z0, z1)``."""
    origin = grid_volume_origin(grid)
    return replace(
        grid,
        nz=z1 - z0,
        vol_origin=(origin[0], origin[1], origin[2] + z0 * grid.vz),
        vol_center=None,
    )


def storage_rows(
    geometry: Geometry, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Bucket a slab's measured rows without changing any physical coordinates."""
    parts = cone_parts(geometry, detector)
    assert parts is not None
    fan = FanRows.for_scan(geometry, parts, len(poses))
    if fan is not None:
        return fan.band(*filter_band(fan.parts, grid, fan.parts[0][2], poses, z0, z1))
    return filter_band(parts, grid, detector, poses, z0, z1)


def filter_band(
    parts: Parts, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    r0, r1 = _slab_rows(parts, grid, detector, poses, z0, z1)
    quantum = min(64, max(1, detector.nv // 8))
    rows = min(detector.nv, -(-(r1 - r0) // quantum) * quantum)
    r0 = min(r0, detector.nv - rows)
    return r0, r0 + rows


def fdk_slabs[T](
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    *,
    config: FDKConfig,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Yield owned FDK slabs from corrected row bands without whole host arrays.

    ``read(first, stop)`` supplies all views of those measured rows and its
    retained context. Weights use the complete acquisition, not the row crop.
    At most 32 output planes are reconstructed together on the first configured
    device. Closing the iterator releases its inputs; completed slabs stay valid.

    Parameters
    ----------
    geometry : Geometry
        Cone-beam acquisition with view angles.
    grid : Grid
        Physical output grid.
    detector : Detector
        Full acquisition detector.
    read : callable
        Corrected row-band reader returning projections and retained context.
    config : FDKConfig
        Filter, batching and backend settings.

    Yields
    ------
    ReconstructionSlab
        Owned host volume with its measured rows and producer context.
    """
    from tomojax.backends import device_free_memory_bytes
    from tomojax.recon.types import ReconstructionSlab

    n_views = len(getattr(geometry, "angles", ()))
    check_angular_coverage(geometry, detector, n_views)
    devices = as_devices(config.devices)
    device = devices[0] if devices else None
    with jax.default_device(device):
        free = device_free_memory_bytes() or 2 * 1024**3
        depth = max(1, min(32, int(free // (3 * 4 * grid.nx * grid.ny))))
        poses = np.asarray(stack_view_poses(geometry, n_views), np.float64)
        prep = _prepare(geometry, grid, detector, poses, replace(config, devices=device))
        coeff = prep.coefficients(geometry, grid, detector, jnp.asarray(poses, jnp.float32))
        with checkpoint.slabs() as finished:
            for start in range(finished.prefix, grid.nz, depth):
                stop = min(start + depth, grid.nz)
                r0, r1 = storage_rows(geometry, grid, detector, poses, start, stop)
                projections, context = read(r0, r1)
                f0, f1 = (
                    (r0, r1)
                    if prep.fan is None
                    else filter_band(prep.fan.parts, grid, prep.detector, poses, start, stop)
                )
                slab = _slab_grid(grid, start, stop)
                volume = _accumulate(
                    slab,
                    projections,
                    prep.window(f0, f1),
                    _window_coefficients(coeff, f0, start),
                    None,
                    native_row=r0,
                )
                owned = np.empty((stop - start, grid.ny, grid.nx), np.float32, order="C")
                copy_to_host(volume, owned)
                del volume
                yield ReconstructionSlab(start, stop, slice(r0, r1), projections, owned, context)
                del projections, owned, context
                finished.add(start, stop)


def host_slab_ranges(
    grid: Grid, devices: Sequence[Device | None], depth: int | None
) -> list[tuple[int, int]]:
    """Size FDK slabs for the least free memory among their workers."""
    from tomojax.backends import device_free_memory_bytes

    if depth is None:
        free = min(device_free_memory_bytes(device=d) or 2 * 1024**3 for d in devices)
        depth = max(1, int(free // (3 * 4 * grid.nx * grid.ny)))
    return [(z, min(z + depth, grid.nz)) for z in range(0, grid.nz, depth)]


@progress.task("reconstruct", "fdk", None, "slabs")
def fdk_host(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    config: FDKHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """FDK with host input and output, reconstructed in z slabs on the device.

    Accepts NumPy arrays and memmaps; ``out`` may be a writable ``(nz, ny, nx)``
    FP32 memmap. Each slab filters a bounded band containing the rows it projects
    onto in any calibrated view, so projections and volume can both exceed device
    memory. With several ``config.fdk.devices``, each device
    reconstructs its own slabs, so the volume may exceed every one of them.

    Input and output storage must not overlap. Input storage must remain
    unchanged until the call finishes; sampled projections must be finite in
    FP32. Completed slabs are written immediately, so a later failure can
    leave a partially written output. This host-returning routine is not
    differentiable; use :func:`fdk` for device-resident output.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    config : FDKHostConfig or None, optional
        Solver settings. None uses FDKHostConfig defaults.
    out : numpy.ndarray or None, optional
        Writable float32 host storage of shape ``(nz, ny, nx)``, including a memmap.
        Must not overlap input storage. None allocates a new C-order output.

    Returns
    -------
    numpy.ndarray
        Float32 reconstruction of shape ``(nz, ny, nx)``. Returns out itself when supplied.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    """
    cfg = FDKHostConfig() if config is None else config
    if cone_parts(geometry, detector) is None:
        raise ValueError("fdk_host needs a cone-beam geometry; use fbp_host for parallel beams")
    depth = cfg.slices_per_batch
    if depth is not None:
        if isinstance(depth, bool | np.bool_):
            raise ValueError("fdk_host: slices_per_batch must be a positive integer or None")
        try:
            depth = operator.index(depth)
        except TypeError as error:
            raise ValueError(
                "fdk_host: slices_per_batch must be a positive integer or None"
            ) from error
        if depth < 1:
            raise ValueError("fdk_host: slices_per_batch must be a positive integer or None")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk_host projections"
    )
    check_angular_coverage(geometry, detector, n_views)
    result = validate_host_arrays(projections, out, (nz, ny, nx), "fdk_host")
    devices = as_devices(cfg.fdk.devices) or (None,)
    slabs = host_slab_ranges(grid, devices, depth)
    progress.remaining("slabs", len(slabs))
    poses = np.asarray(stack_view_poses(geometry, n_views), np.float64)
    work = _Slabs(geometry, grid, detector, projections, poses, result)
    if progress.tracking_placement():
        announce(
            tuple(d for d in devices[: len(slabs)] if d is not None) or jax.local_devices()[:1]
        )
    with checkpoint.slabs(result) as finished:
        slabs = [(z0, z1) for z0, z1 in slabs if not finished.finished(z0, z1)]
        work.write_all(slabs, devices, cfg.fdk, finished)
    return result


@jax.jit
def _window_coefficients(coeff: jax.Array, row: int, z: int) -> jax.Array:
    """Translate full-grid coefficients by integer voxel/pixel offsets only."""
    coeff = coeff.at[:, 2].add(-z)
    coeff = coeff.at[:, 5].add(-z)
    coeff = coeff.at[:, 3:6].add(row * coeff[:, 9:12])
    return coeff.at[:, 24].add(-row)


@dataclass(frozen=True)
class _Slabs:
    """A scan's FDK in z slabs, into ``result``; see :func:`fdk_host`."""

    geometry: Geometry
    grid: Grid
    detector: Detector
    projections: np.ndarray
    poses: np.ndarray
    result: np.ndarray

    def reconstruct(self, z0: int, z1: int, prep: _Prepared, coeff: jax.Array) -> jax.Array:
        """Slices ``[z0, z1)``, filtering only their physical detector-row window."""
        slab = _slab_grid(self.grid, z0, z1)
        r0, r1 = storage_rows(self.geometry, self.grid, self.detector, self.poses, z0, z1)
        f0, f1 = (
            (r0, r1)
            if prep.fan is None
            else filter_band(prep.fan.parts, self.grid, prep.detector, self.poses, z0, z1)
        )
        views = _RowView(self.projections, r0, r1)
        return _accumulate(
            slab,
            views,
            prep.window(f0, f1),
            _window_coefficients(coeff, f0, z0),
            None,
            native_row=r0,
        )

    def write_all(
        self,
        slabs: list[tuple[int, int]],
        devices: Sequence[Device | None],
        fdk: FDKConfig,
        finished: checkpoint.Slabs,
    ) -> None:
        """Schedule one slab per device and publish successful writes on the owner."""

        def land(write: tuple[int, int, Future[None]]) -> None:
            z0, z1, stored = write
            stored.result()
            finished.add(z0, z1)

        with ThreadPoolExecutor(len(devices)) as writer, ThreadPoolExecutor(len(devices)) as pool:
            workers = [
                self.run(slabs[k :: len(devices)], device, fdk=fdk, writer=writer)
                for k, device in enumerate(devices)
            ]
            written = []
            for first in range(0, len(slabs), len(devices)):
                progress.check_canceled()
                jobs = [pool.submit(next, worker) for worker in workers[: len(slabs) - first]]
                for previous in written:
                    land(previous)
                written = [job.result() for job in jobs]
                if progress.for_unit("slabs") is not None:
                    for write in written:
                        progress.check_canceled()
                        land(write)
                        progress.step("slabs")
                    written.clear()
            for previous in written:
                land(previous)

    def run(
        self,
        slabs: list[tuple[int, int]],
        device: Device | None,
        *,
        fdk: FDKConfig,
        writer: Executor,
    ) -> Generator[tuple[int, int, Future[None]], None, None]:
        """Reconstruct ``slabs`` in turn on ``device``, each stored by ``writer`` as the next runs.

        All of a slab's work stays on ``device``, its weights and coefficients
        too: another thread's on the same GPU changes the order of its sums.
        """
        if not slabs:
            return
        if device is not None:
            fdk = replace(fdk, devices=device)
        with jax.default_device(device):
            prep = _prepare(self.geometry, self.grid, self.detector, self.poses, fdk)
            coeff = prep.coefficients(
                self.geometry, self.grid, self.detector, jnp.asarray(self.poses, jnp.float32)
            )
        for z0, z1 in slabs:
            with jax.default_device(device):
                volume = self.reconstruct(z0, z1, prep, coeff)
            written = writer.submit(self.store, z0, z1, volume)
            del volume
            yield z0, z1, written

    def store(self, z0: int, z1: int, volume: jax.Array) -> None:
        copy_to_host(volume, self.result[z0:z1])


class _RowView:
    """Lazy ``projections[:, r0:r1]`` that slices views first, so memmaps stay on disk."""

    def __init__(self, projections: np.ndarray, r0: int, r1: int) -> None:
        self.projections, self.r0, self.r1 = projections, r0, r1
        self.shape = (projections.shape[0], r1 - r0, projections.shape[2])
        self.dtype = np.dtype(np.float32)
        self.ndim = 3

    def __getitem__(self, views: slice) -> np.ndarray:
        # Conversion can overflow in the copier's worker thread. Report it as
        # the same explicit input error as a measured NaN/Inf, not a warning.
        with np.errstate(over="ignore"):
            data = np.asarray(self.projections[views, self.r0 : self.r1], np.float32)
        if not np.isfinite(data).all():
            raise ValueError("fdk_host: sampled projections must be finite in FP32")
        return data

    def __len__(self) -> int:
        return self.shape[0]


__all__ = ["FDKConfig", "FDKHostConfig", "fdk", "fdk_host", "fdk_slabs", "view_weights"]
