"""Feldkamp-Davis-Kress (FDK) reconstruction for cone-beam scans.

Each projection is cosine-weighted, ramp-filtered along detector rows on the
isocentre plane, and backprojected voxel by voxel with the ``(SOD / U)^2``
distance weight, where U is the voxel's distance from the source along the
detector normal. Full turns share angular measure across conjugate views;
shorter arcs of at least a half turn plus the fan angle use Parker weights
with a virtual fan spanning the acquired arc, retaining overscan measurements.

FDK is exact only in the plane through the source orbit; away from it the
reconstruction carries the usual cone-beam artefacts, growing with the cone
angle. It is a fast first reconstruction and initialiser for the iterative
solvers, which model the cone geometry exactly.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
import logging
import math
import operator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.cone import ConeModel, use_cuda_cone
from tomojax.core.devices import as_devices, view_split
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.host_transfer import copy_to_host, upload_host
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.recon._fdk_angles import grouped_angles, ordered_arc
from tomojax.recon._fdk_coverage import full_turn_weights
from tomojax.recon._fdk_cuda import backproject_cuda
from tomojax.recon._host_arrays import validate_host_arrays
from tomojax.recon.filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from tomojax._typed_arrays import Device
    from tomojax.core.geometry.base import Detector, Geometry, Grid
    from tomojax.core.geometry.cone import ConeBeam
    from tomojax.recon.types import ReconstructionSlab

LOG = logging.getLogger(__name__)

# The CUDA path filters rows up to this wide by matrix product (see _filter_operator):
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

    Parameters
    ----------
    filter : str, optional
        Frequency filter: "ramp", "shepp-logan" or "hann". Default is 'ramp'.
    backend : str, optional
        Execution backend: "auto", "jax" or "cuda". Auto uses CUDA when available.
        Default is 'auto'.
    views_per_batch : int, optional
        Maximum views processed together. Batches also cap FP32 input storage at
        64 MiB (or one view) to bound host transfer staging. Default is 64.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
    """

    filter: str = "ramp"
    backend: str = "auto"
    views_per_batch: int = 64
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", as_devices(self.devices))


def _fan_angles(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """Fan angle of each detector column from the ray through the rotation axis."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.center[0]
    return np.arctan(u / float(beam.source_to_detector)) - np.arctan(
        float(beam.axis_offset) / float(beam.source_to_axis)
    )


def _full_turn(geometry: Geometry, n_views: int) -> bool:
    angles, _, _ = grouped_angles(geometry, n_views)
    return ordered_arc(angles)[2]


def _virtual_columns(
    geometry: Geometry,
    beam: ConeBeam,
    detector: Detector,
    n_views: int,
    *,
    full_turn: bool | None = None,
) -> tuple[int, int]:
    """Columns to add below and above an offset detector, mirroring it about the axis.

    In a full turn an offset detector reconstructs the whole field its long
    side covers; voxels on the short side then project past the detector, where
    the filtered rows (zero data, nonzero ramp tail) must still be read.
    """
    if full_turn is None:
        full_turn = _full_turn(geometry, n_views)
    if not full_turn or not _windowable(beam):
        return 0, 0
    axis = (float(beam.axis_offset) * beam.magnification - float(detector.center[0])) / float(
        detector.du
    ) + (detector.nu - 1) / 2
    below, above = axis, detector.nu - 1 - axis
    if abs(above - below) <= 1.0:
        return 0, 0
    pad = min(math.ceil(abs(above - below)), detector.nu)
    return (pad, 0) if above > below else (0, pad)


def view_weights(geometry: Geometry, detector: Detector, n_views: int) -> np.ndarray:
    """Return ``(views, nu)`` FDK angular weights.

    Full turns share each ray's angular measure with its conjugate view's
    detector coverage; short scans use Parker weights. Views repeating
    an angle (several turns, at several heights say) share its measure, so a
    scan of several turns averages their reconstructions. Angle labels may
    wrap at 360 degrees or differ by whole turns without changing the weights.
    """
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("view_weights needs a cone-beam geometry")
    angles, view_angle, repeats = grouped_angles(geometry, n_views)
    ordering = ordered_arc(angles)
    if ordering[2]:
        fans, pixels, counts = [], [], []
        for count, beam, own in parts:
            size = n_views if count is None else count
            counts.append(size)
            fans.append(np.broadcast_to(_fan_angles(beam, own), (size, own.nu)))
            pixels.append(np.full(size, float(own.du) / float(beam.source_to_detector)))
        return full_turn_weights(
            angles, view_angle, repeats, np.concatenate(fans), np.concatenate(pixels), tuple(counts)
        )
    weights, first = [], 0
    for count, beam, own in parts:
        stop = n_views if count is None else first + count
        selected = view_angle[first:stop]
        weights.append(_distinct_view_weights(beam, own, selected, ordering))
        first = stop
    return np.concatenate(weights) / repeats[view_angle][:, None]


def _distinct_view_weights(
    beam: ConeBeam,
    detector: Detector,
    selected: np.ndarray,
    ordering: tuple[np.ndarray, np.ndarray, bool],
) -> np.ndarray:
    """:func:`view_weights` for views at distinct ``angles`` (radians)."""
    order, sorted_angles, _ = ordering
    n_views = len(order)
    gaps = np.diff(sorted_angles)
    step = float(np.median(gaps))
    arc = float(sorted_angles[-1] - sorted_angles[0]) + step
    # Angular measure of each view: half the gap to each neighbour.
    measure = np.empty(n_views)
    padded = np.concatenate([[sorted_angles[0] - step], sorted_angles, [sorted_angles[-1] + step]])
    measure[order] = 0.5 * (padded[2:] - padded[:-2])
    # Parker weights over the fan angle of each detector column.
    gamma = _fan_angles(beam, detector)
    pixel = float(detector.du) / float(beam.source_to_detector)
    if abs(float(gamma.max() + gamma.min())) > 0.05 * float(gamma.max() - gamma.min()):
        LOG.warning(
            "FDK short scan with an offset detector: Parker weights assume the axis "
            "projects near the detector centre (fan %.2f to %.2f degrees); columns beyond "
            "the short side are weighted as if their conjugate rays were measured",
            np.rad2deg(gamma.min() - pixel / 2),
            np.rad2deg(gamma.max() + pixel / 2),
        )
    delta = float(np.max(np.abs(gamma)))
    if arc < np.pi + 2 * delta - 1e-6:
        raise ValueError(
            f"FDK short scans need at least 180 degrees plus the fan angle "
            f"({np.rad2deg(np.pi + 2 * delta):.1f} degrees); got {np.rad2deg(arc):.1f}"
        )
    # Treat the extra acquisition angle as a virtual fan. Otherwise the
    # minimum-scan window assigns zero to valid later views of an overscan.
    delta = max(delta, (arc - np.pi) / 2)
    beta = np.empty((n_views, 1))
    beta[order, 0] = sorted_angles - sorted_angles[0]
    beta = beta[selected]
    g = gamma[None, :]
    weight = np.ones((len(selected), detector.nu))
    rise = beta < 2 * (delta - g)
    weight = np.where(rise, np.sin(np.pi / 4 * beta / np.maximum(delta - g, 1e-12)) ** 2, weight)
    end = np.pi + 2 * delta
    fall = beta > np.pi - 2 * g
    weight = np.where(
        fall, np.sin(np.pi / 4 * (end - beta) / np.maximum(delta + g, 1e-12)) ** 2, weight
    )
    weight = np.where(beta > end, 0.0, weight)
    return weight * measure[selected, None]


def _cosine_weights(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """``(nv, nu)`` cosine of each pixel's ray to the detector normal."""
    centre, u_dir, v_dir = beam.detector_frame(detector)
    normal = np.cross(u_dir, v_dir)
    normal /= np.linalg.norm(normal)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    pixel = centre + u[None, :, None] * u_dir + v[:, None, None] * v_dir
    ray = pixel - beam.source()
    return np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)


def _cosines(parts: tuple[tuple[int | None, ConeBeam, Detector], ...], n_views: int) -> np.ndarray:
    """Shared cosine image, or compact per-view ray coordinates in detector axes."""
    if len(parts) == 1:
        return np.asarray(_cosine_weights(parts[0][1], parts[0][2]), np.float32)
    values = []
    for count, beam, detector in parts:
        centre, u, v = beam.detector_frame(detector)
        ray = centre - beam.source()
        coordinates = np.array([ray @ u, ray @ v, abs(ray @ np.cross(u, v))], np.float32)
        values.append(np.broadcast_to(coordinates, (n_views if count is None else count, 3)))
    return np.concatenate(values)[..., None]


def _batch_cosines(
    cosine: jax.Array, start: jax.Array | int, count: int, detector: Detector
) -> jax.Array:
    if cosine.ndim == 2:
        return cosine
    parameters = jax.lax.dynamic_slice_in_dim(cosine, start, count)[..., 0]
    u = (jnp.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (jnp.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    x = parameters[:, 0, None, None] + u[None, None, :]
    y = parameters[:, 1, None, None] + v[None, :, None]
    z = parameters[:, 2, None, None]
    return z / jnp.sqrt(x * x + y * y + z * z)


def _filter_operator(spectrum: np.ndarray, nu: int, pad_lo: int, pad_hi: int) -> np.ndarray:
    """The ``(nu, pad_lo + nu + pad_hi)`` matrix that ramp-filters rows as ``spectrum`` does.

    Filtering is a circular convolution with the zero-padded row; as a matrix
    product it runs on the GPU's matrix units several times faster than the
    FFTs for rows up to a couple of thousand pixels.
    """
    n_fft = 2 * (spectrum.shape[0] - 1)
    impulse = np.fft.irfft(spectrum.astype(np.float64), n=n_fft)
    columns = np.arange(-pad_lo, nu + pad_hi)
    return impulse[(columns[None, :] - np.arange(nu)[:, None]) % n_fft].astype(np.float32)


def _filter(
    views: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array,
    pad_lo: int = 0,
    pad_hi: int = 0,
) -> jax.Array:
    """Weight and ramp-filter rows, keeping ``pad_lo``/``pad_hi`` columns beyond them.

    Filtered rows extend past the data; an offset detector's backprojection
    needs that tail on its short side (see :func:`_virtual_columns`).
    """
    # Angular (Parker, offset-detector) weights vary along the row, so they
    # apply before the ramp filter, as do the cosine weights.
    rows = views * cosine * weights[:, None, :]
    n_fft = 2 * (int(kernel.shape[0]) - 1)
    nu = views.shape[-1]
    spectrum = jnp.fft.rfft(rows, n=n_fft, axis=-1) * kernel
    out = jnp.fft.irfft(spectrum, n=n_fft, axis=-1)
    if pad_lo:
        return jnp.concatenate([out[..., n_fft - pad_lo :], out[..., : nu + pad_hi]], axis=-1)
    return out[..., : nu + pad_hi]


def _backproject_jax(
    filtered: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    scale: float,
    out: jax.Array,
) -> jax.Array:
    """Voxel-driven backprojection of filtered ``(views, nv, nu)`` images."""
    index = [jnp.arange(n, dtype=jnp.float32) for n in (grid.nx, grid.ny, grid.nz)]
    qx, qy, qz = jnp.meshgrid(*index, indexing="ij")

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
    static_argnames=("grid", "detector", "scale", "cuda", "pad_lo", "pad_hi"),
    donate_argnames=("out",),
)
def _fdk_batch(
    views: jax.Array,
    start: jax.Array | int,
    coeff: jax.Array,
    cosine: jax.Array,
    weights: jax.Array,
    kernel: jax.Array,
    operator: jax.Array | None,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
    pad_lo: int,
    pad_hi: int,
) -> jax.Array:
    # The batch's views of the scan's coefficients and weights.
    coeff = jax.lax.dynamic_slice_in_dim(coeff, start, views.shape[0])
    weights = jax.lax.dynamic_slice_in_dim(weights, start, views.shape[0])
    measured = replace(detector, nu=views.shape[-1])
    cosine = _batch_cosines(cosine, start, views.shape[0], measured)
    if operator is not None:
        rows = views * cosine * weights[:, None, :]
        return backproject_cuda(rows, coeff, grid, detector, scale, out, operator)
    filtered = _filter(views, cosine, weights, kernel, pad_lo, pad_hi)
    backproject = backproject_cuda if cuda else _backproject_jax
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
    # The filter as a matrix (see _filter_operator), for the CUDA kernel's rows.
    operator: jax.Array | None = None

    @property
    def batch(self) -> int:
        """Bound transfers for the current row window, up to the configured view cap."""
        # Prefetch and transfer staging can retain several input buffers.
        # Size these for the measured rows, not the uncropped detector.
        transfer_views = max(1, (64 * 2**20) // (4 * self.detector.nu * self.detector.nv))
        return max(1, min(self.views_per_batch, transfer_views))

    def coefficients(
        self, geometry: Geometry, grid: Grid, detector: Detector, poses: jax.Array
    ) -> jax.Array:
        parts = cone_parts(geometry, detector)
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
    detector: Detector,
    n_views: int,
    cfg: FDKConfig,
    columns: Detector | None = None,
) -> _Prepared:
    """FDK weights, filter and virtual columns for ``detector``.

    ``columns``, a detector with the same columns (the full detector of a band
    of rows), sets the per-column angular weights and virtual columns, so every
    band of a rolled detector gets the same ones.
    """
    columns = detector if columns is None else columns
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    if cfg.backend not in {"auto", "jax", "cuda"}:
        raise ValueError("fdk backend must be 'auto', 'jax' or 'cuda'")
    cuda = use_cuda_cone() if cfg.backend == "auto" else cfg.backend == "cuda"
    if cuda and not use_cuda_cone():
        raise ValueError("fdk: the CUDA kernel needs CuPy on a CUDA device")
    magnification = parts[0][1].magnification
    du_iso = float(detector.du) / magnification
    column_parts = cone_parts(geometry, columns)
    assert column_parts is not None
    full_turn = _full_turn(geometry, n_views)
    padding = [
        _virtual_columns(geometry, beam, own, n_views, full_turn=full_turn)
        for _, beam, own in column_parts
    ]
    pad_lo, pad_hi = np.max(padding, axis=0).tolist()
    # A shared detector-pitch ramp is linear in magnification. Fold each
    # view's remaining inverse magnification into its angular weight.
    weights = view_weights(geometry, columns, n_views)
    weights *= magnification / ConeModel(parts).magnification(n_views)[:, None]
    # The kernel spans the virtual row too, so its tail does not wrap around.
    width = detector.nu + pad_lo + pad_hi
    kernel = get_fbp_filter_np(cfg.filter, width, du_iso, "float32")
    operator = None
    if cuda and width <= _DENSE_FILTER_MAX:
        operator = jax.device_put(_filter_operator(kernel, detector.nu, pad_lo, pad_hi))
    return _Prepared(
        cuda=cuda,
        detector=detector,
        # device_put, unlike jnp.asarray, compiles nothing for a new shape.
        weights=jax.device_put(np.asarray(weights, np.float32)),
        cosine=jax.device_put(_cosines(parts, n_views)),
        kernel=jax.device_put(kernel),
        operator=operator,
        scale=1.0 / magnification,
        views_per_batch=int(cfg.views_per_batch),
        pad_lo=pad_lo,
        pad_hi=pad_hi,
    )


def _device_views(
    projections: jax.Array | np.ndarray | _RowView,
    start: int,
    stop: int,
    device: Device,
) -> jax.Array:
    part = projections[start:stop]
    if isinstance(part, jax.Array):
        return jax.device_put(part.astype(jnp.float32), device)
    return upload_host(np.asarray(part, np.float32), device)


_filter_jit = jax.jit(_filter, static_argnames=("pad_lo", "pad_hi"))


@partial(jax.jit, static_argnames=("grid", "detector", "scale", "cuda"), donate_argnames=("out",))
def _backproject_batch(
    filtered: jax.Array,
    coeff: jax.Array,
    out: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    scale: float,
    cuda: bool,
) -> jax.Array:
    backproject = backproject_cuda if cuda else _backproject_jax
    return backproject(filtered, coeff, grid, detector, scale, out)


def _filter_views(prep: _Prepared, projections: jax.Array | np.ndarray, n_views: int) -> jax.Array:
    """Weighted, ramp-filtered ``(views, nv, nu)`` projections on the device."""
    parts = []
    for start in range(0, n_views, prep.batch):
        stop = min(start + prep.batch, n_views)
        views = _device_views(projections, start, stop, next(iter(prep.weights.devices())))
        weights = prep.weights[start:stop]
        cosine = _batch_cosines(prep.cosine, start, stop - start, prep.detector)
        parts.append(_filter_jit(views, cosine, weights, prep.kernel, prep.pad_lo, prep.pad_hi))
    return jnp.concatenate(parts, axis=0)


def _backproject_filtered(
    geometry: Geometry, grid: Grid, detector: Detector, filtered: jax.Array, prep: _Prepared
) -> jax.Array:
    """Backproject filtered projections from :func:`_filter_views` into ``grid``."""
    n_views = int(filtered.shape[0])
    coeff = prep.coefficients(geometry, grid, detector, stack_view_poses(geometry, n_views))
    detector = prep.virtual(detector)
    out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    for start in range(0, n_views, prep.batch):
        stop = min(start + prep.batch, n_views)
        out = _backproject_batch(
            filtered[start:stop], coeff[start:stop], out,
            grid=grid, detector=detector, scale=prep.scale, cuda=prep.cuda,
        )  # fmt: skip
    return out


def fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: FDKConfig | None = None,
) -> jax.Array:
    """Reconstruct a cone-beam scan with FDK; returns an ``(nx, ny, nz)`` volume.

    ``projections`` are ``(views, nv, nu)`` line integrals; NumPy and memmap
    stacks stream to the device ``views_per_batch`` views at a time. Full turns
    on an offset detector (the axis projecting off its centre) get Wang's
    weights from the conjugate views and reconstruct the field the detector's long side covers.
    Each view retains its own calibrated source/detector arrangement; angular
    quadrature covers the whole scan, independently of segment boundaries.
    Noncircular trajectories retain FDK's approximate character.

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
        Float32 reconstruction of shape ``(nx, ny, nz)``.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    """
    cfg = FDKConfig() if config is None else config
    parts = cone_parts(geometry, detector)
    if parts is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    validate_grid(grid, "fdk grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk projections"
    )
    prep = _prepare(geometry, detector, n_views, cfg)
    coeff = prep.coefficients(geometry, grid, detector, stack_view_poses(geometry, n_views))
    return _accumulate(grid, projections, prep, coeff, cfg.devices)


def _accumulate(
    grid: Grid,
    projections: jax.Array | np.ndarray | _RowView,
    prep: _Prepared,
    coeff: jax.Array,
    devices: Device | Sequence[Device] | None,
) -> jax.Array:
    """Filter and sum validated views using coefficients in this volume's voxel frame."""
    n_views = int(coeff.shape[0])
    virtual = prep.virtual(prep.detector)
    # Physical placement is already in coeff. Equal-sized slabs share the
    # filtering/backprojection executable, rather than compiling every origin.
    kernel_grid = replace(grid, vol_origin=None, vol_center=None)

    def backprojected(device: Device | None, views: range) -> jax.Array:
        """The weighted backprojection of ``views``, on ``device`` (JAX's default for None)."""
        here = prep.on(device, views)
        share = jax.device_put(coeff[views.start : views.stop], device)
        out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32, device=device)
        target = next(iter(out.devices()))
        starts = list(range(0, len(views), prep.batch))

        def read(start: int) -> jax.Array:
            first = views.start + start
            return _device_views(projections, first, min(first + prep.batch, views.stop), target)

        # Copy the next batch to the device while the current one is filtered
        # and backprojected.
        with ThreadPoolExecutor(max_workers=1) as copier:
            pending = copier.submit(read, 0)
            for index, start in enumerate(starts):
                batch = pending.result()
                if index + 1 < len(starts):
                    pending = copier.submit(read, starts[index + 1])
                out = _fdk_batch(
                    batch, start, share, here.cosine, here.weights, here.kernel, here.operator,
                    out, grid=kernel_grid, detector=virtual, scale=prep.scale, cuda=prep.cuda,
                    pad_lo=prep.pad_lo, pad_hi=prep.pad_hi,
                )  # fmt: skip
                # Keep only this batch and its prefetched successor live.
                out.block_until_ready()
        return out

    split = view_split(devices, n_views)
    whole = range(n_views)
    return backprojected(None, whole) if split is None else split.summed(backprojected)


@dataclass(frozen=True, kw_only=True)
class FDKHostConfig:
    """Slab options for :func:`fdk_host`.

    ``slices_per_batch`` z slices are reconstructed per slab; ``None`` sizes slabs
    to about a third of the free device memory. ``fdk`` holds the filter and backend
    options.

    Parameters
    ----------
    slices_per_batch : int | None, optional
        Positive slab depth, or None to choose automatically as described above.
    fdk : FDKConfig, optional
        FDK filter, batching and backend configuration.
    """

    slices_per_batch: int | None = None
    fdk: FDKConfig = field(default_factory=FDKConfig)


def _slab_rows(
    geometry: Geometry, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Detector rows that the z-slab ``[z0, z1)`` projects onto in any view."""
    parts = cone_parts(geometry, detector)
    assert parts is not None
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
        centre, u_dir, v_dir = beam.detector_frame(own)
        normal = np.cross(u_dir, v_dir)
        source = beam.source()
        ray = world[start:stop] - source
        hit = source + ray * (((centre - source) @ normal) / (ray @ normal))[..., None]
        rows = ((hit - centre) @ v_dir) / own.dv + (own.nv - 1) / 2
        low, high, start = min(low, float(rows.min())), max(high, float(rows.max())), stop
    # Keep one measured border row even when the slab is entirely outside.
    # Backprojection still samples zero beyond that row, as with the full
    # detector, without constructing an empty or negative-height detector.
    first = max(0, min(detector.nv - 1, math.floor(low) - 1))
    stop = max(first + 1, min(detector.nv, math.ceil(high) + 2))
    return first, stop


def _windowable(beam: ConeBeam) -> bool:
    """Whether a band of detector rows is itself a detector of ``beam`` (no pitch or yaw)."""
    return float(beam.detector_pitch_deg) == 0.0 and float(beam.detector_yaw_deg) == 0.0


def _detector_window(
    beam: ConeBeam,
    detector: Detector,
    r0: int,
    r1: int,
    c1: int | None = None,
    f: int = 1,
) -> Detector:
    """Rows ``[r0, r1)`` and columns ``[0, c1)`` of ``detector``, binned ``f x f``.

    The window keeps its place on the rolled detector: its centre moves along
    the detector's own u and v. Needs :func:`_windowable` beams, whose detector
    axes stay in the plane of ``detector.center``.
    """
    rows, cols = (r1 - r0) // f, (detector.nu if c1 is None else c1) // f
    su = ((f * cols - 1) / 2 - (detector.nu - 1) / 2) * detector.du
    sv = (r0 + (f * rows - 1) / 2 - (detector.nv - 1) / 2) * detector.dv
    _, u_dir, v_dir = beam.detector_frame(detector)
    shift = su * u_dir + sv * v_dir
    return replace(
        detector,
        nu=cols,
        nv=rows,
        du=detector.du * f,
        dv=detector.dv * f,
        center=(detector.center[0] + shift[0], detector.center[1] + shift[2]),
    )


def _slab_grid(grid: Grid, z0: int, z1: int) -> Grid:
    """Grid of z slices ``[z0, z1)``."""
    origin = grid_volume_origin(grid)
    return replace(
        grid,
        nz=z1 - z0,
        vol_origin=(origin[0], origin[1], origin[2] + z0 * grid.vz),
        vol_center=None,
    )


def _storage_rows(
    geometry: Geometry, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Bucket a slab's measured rows without changing any physical coordinates."""
    r0, r1 = _slab_rows(geometry, grid, detector, poses, z0, z1)
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

    validate_grid(grid, "FDK slab grid")
    n_views = len(getattr(geometry, "angles", ()))
    ordered_arc(grouped_angles(geometry, n_views)[0])
    devices = as_devices(config.devices)
    device = devices[0] if devices else None
    with jax.default_device(device):
        free = device_free_memory_bytes() or 2 * 1024**3
        depth = max(1, min(32, int(free // (3 * 4 * grid.nx * grid.ny))))
        poses = np.asarray(stack_view_poses(geometry, n_views), np.float64)
        prep = _prepare(geometry, detector, n_views, replace(config, devices=device))
        coeff = prep.coefficients(geometry, grid, detector, jnp.asarray(poses, jnp.float32))
        for start in range(0, grid.nz, depth):
            stop = min(start + depth, grid.nz)
            r0, r1 = _storage_rows(geometry, grid, detector, poses, start, stop)
            projections, context = read(r0, r1)
            slab = _slab_grid(grid, start, stop)
            volume = _accumulate(
                slab,
                projections,
                prep.window(r0, r1),
                _window_coefficients(coeff, r0, start),
                None,
            )
            owned = np.empty((grid.nx, grid.ny, stop - start), np.float32, order="F")
            copy_to_host(volume.T, owned.T)
            del volume
            yield ReconstructionSlab(start, stop, slice(r0, r1), projections, owned, context)
            del projections, owned, context


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

    Accepts NumPy arrays and memmaps; ``out`` may be a writable ``(nx, ny, nz)``
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
        Writable float32 host storage of shape ``(nx, ny, nz)``, including a memmap.
        Must not overlap input storage. None allocates a new Fortran-order output.

    Returns
    -------
    numpy.ndarray
        Float32 reconstruction of shape ``(nx, ny, nz)``. Returns out itself when supplied.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    """
    from tomojax.backends import device_free_memory_bytes

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
    shape = validate_grid(grid, "fdk_host grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk_host projections"
    )
    ordered_arc(grouped_angles(geometry, n_views)[0])
    result = validate_host_arrays(projections, out, shape, "fdk_host", order="F")
    if depth is None:
        free = device_free_memory_bytes() or 2 * 1024**3
        depth = max(1, int(free // (3 * max(1, 4 * grid.nx * grid.ny))))
    depth = min(int(depth), grid.nz)
    slabs = [(z0, min(z0 + depth, grid.nz)) for z0 in range(0, grid.nz, depth)]
    poses = np.asarray(stack_view_poses(geometry, n_views), np.float64)
    work = _Slabs(geometry, grid, detector, projections, poses, result)
    devices = as_devices(cfg.fdk.devices) or (None,)
    # Each device's slabs in turn, copied into ``result`` (slow for memmaps)
    # while the next one runs.
    with ThreadPoolExecutor(len(devices)) as writer, ThreadPoolExecutor(len(devices)) as pool:
        shares = [slabs[k :: len(devices)] for k in range(len(devices))]
        list(pool.map(partial(work.run, fdk=cfg.fdk, writer=writer), shares, devices))
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
        r0, r1 = _storage_rows(self.geometry, self.grid, self.detector, self.poses, z0, z1)
        views = _RowView(self.projections, r0, r1)
        return _accumulate(
            slab, views, prep.window(r0, r1), _window_coefficients(coeff, r0, z0), None
        )

    def run(
        self,
        slabs: list[tuple[int, int]],
        device: Device | None,
        *,
        fdk: FDKConfig,
        writer: Executor,
    ) -> None:
        """Reconstruct ``slabs`` in turn on ``device``, each stored by ``writer`` as the next runs.

        All of a slab's work stays on ``device``, its weights and coefficients
        too: another thread's on the same GPU changes the order of its sums.
        """
        if not slabs:
            return
        if device is not None:
            fdk = replace(fdk, devices=device)
        with jax.default_device(device):
            prep = _prepare(self.geometry, self.detector, len(self.projections), fdk)
            coeff = prep.coefficients(
                self.geometry, self.grid, self.detector, jnp.asarray(self.poses, jnp.float32)
            )
        pending: Future[None] | None = None
        for z0, z1 in slabs:
            with jax.default_device(device):
                volume = self.reconstruct(z0, z1, prep, coeff)
            if pending is not None:
                pending.result()
            pending = writer.submit(self.store, z0, z1, volume)
            del volume
        if pending is not None:
            pending.result()

    def store(self, z0: int, z1: int, volume: jax.Array) -> None:
        copy_to_host(volume.T, self.result[:, :, z0:z1].T)


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
