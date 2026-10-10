"""Feldkamp-Davis-Kress (FDK) reconstruction for cone-beam scans.

Each projection is cosine-weighted, ramp-filtered along detector rows on the
isocentre plane, and backprojected voxel by voxel with the ``(SOD / U)^2``
distance weight, where U is the voxel's distance from the source along the
detector normal. Full turns weight each view by half its angular step;
shorter arcs of at least a half turn plus the fan angle use Parker weights.

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

from tomojax.core.cone import cone_coefficients, use_cuda_cone
from tomojax.core.devices import as_devices, view_split
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.recon._fdk_angles import grouped_angles, ordered_arc
from tomojax.recon._fdk_cuda import backproject_cuda
from tomojax.recon._host_arrays import validate_host_arrays
from tomojax.recon.filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax._typed_arrays import Device
    from tomojax.core.geometry.base import Detector, Geometry, Grid
    from tomojax.core.geometry.cone import ConeBeam

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


def _full_turn_column_weights(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """``(nu,)`` weights of a full turn: 1/2, or Wang's for an offset detector.

    A full turn measures each ray twice, once from each side, so each column
    takes half the angular measure. When the detector is offset so that the
    rotation axis projects off its centre, columns beyond the short side's
    reach are measured once and weigh 1, and a sin^2 ramp across the overlap
    (0 at the short edge, 1/2 at the axis, 1 at the mirrored column) keeps the
    weights of each ray's two measurements summing to one.
    """
    gamma = _fan_angles(beam, detector)
    lo, hi = float(gamma.min()), float(gamma.max())
    pixel = float(detector.du) / float(beam.source_to_detector)
    if lo > pixel or hi < -pixel:
        raise ValueError("FDK needs the rotation axis to project onto the detector")
    if abs(hi + lo) <= pixel:
        return np.full(detector.nu, 0.5)
    side = 1.0 if hi + lo > 0 else -1.0
    overlap = max(min(-lo, hi), pixel)
    x = side * gamma
    ramp = np.sin(np.pi / 4 * (1 + np.clip(x / overlap, -1.0, 1.0))) ** 2
    return np.where(x >= overlap, 1.0, ramp)


def _full_turn(geometry: Geometry, n_views: int) -> bool:
    angles, _, _ = grouped_angles(geometry, n_views)
    return ordered_arc(angles)[2]


def _virtual_columns(
    geometry: Geometry, beam: ConeBeam, detector: Detector, n_views: int
) -> tuple[int, int]:
    """Columns to add below and above an offset detector, mirroring it about the axis.

    In a full turn an offset detector reconstructs the whole field its long
    side covers; voxels on the short side then project past the detector, where
    the filtered rows (zero data, nonzero ramp tail) must still be read.
    """
    if not _full_turn(geometry, n_views) or not _windowable(beam):
        return 0, 0
    axis = (float(beam.axis_offset) * beam.magnification - float(detector.center[0])) / float(
        detector.du
    ) + (detector.nu - 1) / 2
    below, above = axis, detector.nu - 1 - axis
    if abs(above - below) <= 1.0:
        return 0, 0
    pad = min(math.ceil(abs(above - below)), detector.nu)
    return (pad, 0) if above > below else (0, pad)


def _extended(beam: ConeBeam, detector: Detector, pad_lo: int, pad_hi: int) -> Detector:
    """``detector`` with ``pad_lo`` columns added below u and ``pad_hi`` above."""
    from dataclasses import replace

    if not (pad_lo or pad_hi):
        return detector
    _, u_dir, _ = beam.detector_frame(detector)
    shift = (pad_hi - pad_lo) / 2 * float(detector.du) * u_dir
    return replace(
        detector,
        nu=detector.nu + pad_lo + pad_hi,
        center=(detector.center[0] + shift[0], detector.center[1] + shift[2]),
    )


def view_weights(geometry: Geometry, detector: Detector, n_views: int) -> np.ndarray:
    """Return ``(views, nu)`` FDK angular weights.

    Full turns weigh each view by its angular measure, halved (Wang's weights
    for an offset detector); short scans use Parker weights. Views repeating
    an angle (several turns, at several heights say) share its measure, so a
    scan of several turns averages their reconstructions. Angle labels may
    wrap at 360 degrees or differ by whole turns without changing the weights.
    """
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("view_weights needs a cone-beam geometry")
    angles, view_angle, repeats = grouped_angles(geometry, n_views)
    weights = _distinct_view_weights(beam, detector, angles)
    return weights[view_angle] / repeats[view_angle][:, None]


def _distinct_view_weights(beam: ConeBeam, detector: Detector, angles: np.ndarray) -> np.ndarray:
    """:func:`view_weights` for views at distinct ``angles`` (radians)."""
    n_views = len(angles)
    order, sorted_angles, full_turn = ordered_arc(angles)
    gaps = np.diff(sorted_angles)
    step = float(np.median(gaps))
    arc = float(sorted_angles[-1] - sorted_angles[0]) + step
    # Angular measure of each view: half the gap to each neighbour.
    measure = np.empty(n_views)
    if full_turn:
        wrapped = np.concatenate(
            [sorted_angles[-1:] - 2 * np.pi, sorted_angles, sorted_angles[:1] + 2 * np.pi]
        )
        measure[order] = 0.5 * (wrapped[2:] - wrapped[:-2])
        return measure[:, None] * _full_turn_column_weights(beam, detector)[None, :]
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
    beta = np.empty((n_views, 1))
    beta[order, 0] = sorted_angles - sorted_angles[0]
    g = gamma[None, :]
    weight = np.ones((n_views, detector.nu))
    rise = beta < 2 * (delta - g)
    weight = np.where(rise, np.sin(np.pi / 4 * beta / np.maximum(delta - g, 1e-12)) ** 2, weight)
    end = np.pi + 2 * delta
    fall = beta > np.pi - 2 * g
    weight = np.where(
        fall, np.sin(np.pi / 4 * (end - beta) / np.maximum(delta + g, 1e-12)) ** 2, weight
    )
    weight = np.where(beta > end, 0.0, weight)
    return weight * measure[:, None]


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
    if operator is not None:
        rows = views * cosine * weights[:, None, :]
        return backproject_cuda(rows, coeff, grid, detector, scale, out, operator)
    filtered = _filter(views, cosine, weights, kernel, pad_lo, pad_hi)
    backproject = backproject_cuda if cuda else _backproject_jax
    return backproject(filtered, coeff, grid, detector, scale, out)


@dataclass(frozen=True)
class _Prepared:
    """Per-scan FDK weights and filter on the device; see :func:`_prepare`."""

    cuda: bool
    weights: jax.Array
    cosine: jax.Array
    kernel: jax.Array
    scale: float
    batch: int
    pad_lo: int = 0
    pad_hi: int = 0
    # The filter as a matrix (see _filter_operator), for the CUDA kernel's rows.
    operator: jax.Array | None = None

    def backprojected(self, beam: ConeBeam, detector: Detector) -> Detector:
        """The detector the filtered rows cover, with its virtual columns."""
        return _extended(beam, detector, self.pad_lo, self.pad_hi)

    def on(self, device: Device | None, views: range) -> _Prepared:
        """These weights for ``views`` alone, and the filter, on ``device``."""
        put = partial(jax.device_put, device=device)
        operator = None if self.operator is None else put(self.operator)
        weights = put(self.weights[views.start : views.stop])
        return replace(
            self,
            weights=weights,
            cosine=put(self.cosine),
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
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    if cfg.backend not in {"auto", "jax", "cuda"}:
        raise ValueError("fdk backend must be 'auto', 'jax' or 'cuda'")
    cuda = use_cuda_cone() if cfg.backend == "auto" else cfg.backend == "cuda"
    if cuda and not use_cuda_cone():
        raise ValueError("fdk: the CUDA kernel needs CuPy on a CUDA device")
    du_iso = float(detector.du) / beam.magnification
    pad_lo, pad_hi = _virtual_columns(geometry, beam, columns, n_views)
    # The kernel spans the virtual row too, so its tail does not wrap around.
    width = detector.nu + pad_lo + pad_hi
    kernel = get_fbp_filter_np(cfg.filter, width, du_iso, "float32")
    operator = None
    if cuda and width <= _DENSE_FILTER_MAX:
        operator = jax.device_put(_filter_operator(kernel, detector.nu, pad_lo, pad_hi))
    return _Prepared(
        cuda=cuda,
        # device_put, unlike jnp.asarray, compiles nothing for a new shape.
        weights=jax.device_put(np.asarray(view_weights(geometry, columns, n_views), np.float32)),
        cosine=jax.device_put(np.asarray(_cosine_weights(beam, detector), np.float32)),
        kernel=jax.device_put(kernel),
        operator=operator,
        scale=1.0 / beam.magnification,
        batch=max(1, int(cfg.views_per_batch)),
        pad_lo=pad_lo,
        pad_hi=pad_hi,
    )


def _device_views(
    projections: jax.Array | np.ndarray, start: int, stop: int, device: Device | None = None
) -> jax.Array:
    part = projections[start:stop]
    if isinstance(part, jax.Array):
        return jax.device_put(part.astype(jnp.float32), device)
    return jax.device_put(np.asarray(part, np.float32), device)


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
        views = _device_views(projections, start, stop)
        weights = prep.weights[start:stop]
        parts.append(
            _filter_jit(views, prep.cosine, weights, prep.kernel, prep.pad_lo, prep.pad_hi)
        )
    return jnp.concatenate(parts, axis=0)


def _backproject_filtered(
    geometry: Geometry, grid: Grid, detector: Detector, filtered: jax.Array, prep: _Prepared
) -> jax.Array:
    """Backproject filtered projections from :func:`_filter_views` into ``grid``."""
    beam = beam_of(geometry)
    assert beam is not None
    n_views = int(filtered.shape[0])
    detector = prep.backprojected(beam, detector)
    coeff = cone_coefficients(stack_view_poses(geometry, n_views), grid, detector, beam)
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
    weights and reconstruct the field the detector's long side covers.
    """
    return _fdk(geometry, grid, detector, projections, config=config)


def _fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: FDKConfig | None = None,
    columns: Detector | None = None,
) -> jax.Array:
    cfg = FDKConfig() if config is None else config
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("fdk needs a cone-beam geometry; use fbp for parallel beams")
    validate_grid(grid, "fdk grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fdk projections"
    )
    prep = _prepare(geometry, detector, n_views, cfg, columns)
    virtual = prep.backprojected(beam, detector)
    coeff = cone_coefficients(stack_view_poses(geometry, n_views), grid, virtual, beam)

    def backprojected(device: Device | None, views: range) -> jax.Array:
        """The weighted backprojection of ``views``, on ``device`` (JAX's default for None)."""
        here = prep.on(device, views)
        share = jax.device_put(coeff[views.start : views.stop], device)
        out = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32, device=device)
        starts = list(range(0, len(views), prep.batch))

        def read(start: int) -> jax.Array:
            first = views.start + start
            return _device_views(projections, first, min(first + prep.batch, views.stop), device)

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
                    out, grid=grid, detector=virtual, scale=prep.scale, cuda=prep.cuda,
                    pad_lo=prep.pad_lo, pad_hi=prep.pad_hi,
                )  # fmt: skip
        return out

    split = view_split(cfg.devices, n_views)
    whole = range(n_views)
    return backprojected(None, whole) if split is None else split.summed(backprojected)


@dataclass(frozen=True, kw_only=True)
class FDKHostConfig:
    """Slab options for :func:`fdk_host`.

    ``slices_per_batch`` z slices are reconstructed per slab; ``None`` sizes slabs
    to about a third of the free device memory. ``fdk`` holds the filter and backend
    options.
    """

    slices_per_batch: int | None = None
    fdk: FDKConfig = field(default_factory=FDKConfig)


def _slab_rows(
    geometry: Geometry, grid: Grid, detector: Detector, poses: np.ndarray, z0: int, z1: int
) -> tuple[int, int]:
    """Detector rows that the z-slab ``[z0, z1)`` projects onto in any view."""
    beam = beam_of(geometry)
    assert beam is not None
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
    centre, u_dir, v_dir = beam.detector_frame(detector)
    normal = np.cross(u_dir, v_dir)
    source = beam.source()
    ray = world - source
    hit = source + ray * (((centre - source) @ normal) / (ray @ normal))[..., None]
    rows = ((hit - centre) @ v_dir) / detector.dv + (detector.nv - 1) / 2
    # Keep one measured border row even when the slab is entirely outside.
    # Backprojection still samples zero beyond that row, as with the full
    # detector, without constructing an empty or negative-height detector.
    first = max(0, min(detector.nv - 1, int(np.floor(rows.min())) - 1))
    stop = max(first + 1, min(detector.nv, int(np.ceil(rows.max())) + 2))
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
    from dataclasses import replace

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
    from dataclasses import replace

    origin = grid_volume_origin(grid)
    return replace(
        grid,
        nz=z1 - z0,
        vol_origin=(origin[0], origin[1], origin[2] + z0 * grid.vz),
        vol_center=None,
    )


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
    FP32 memmap. Each slab filters only the detector rows it projects onto (all
    rows for a pitched or yawed detector), so projections and volume can both
    exceed device memory. With several ``config.fdk.devices``, each device
    reconstructs its own slabs, so the volume may exceed every one of them.

    Input and output storage must not overlap. Input storage must remain
    unchanged until the call finishes; sampled projections must be finite in
    FP32. Completed slabs are written immediately, so a later failure can
    leave a partially written output. This host-returning routine is not
    differentiable; use :func:`fdk` for device-resident output.
    """
    from tomojax.backends import device_free_memory_bytes

    cfg = FDKHostConfig() if config is None else config
    beam = beam_of(geometry)
    if beam is None:
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
    result = validate_host_arrays(projections, out, shape, "fdk_host")
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


@dataclass(frozen=True)
class _Slabs:
    """A scan's FDK in z slabs, into ``result``; see :func:`fdk_host`."""

    geometry: Geometry
    grid: Grid
    detector: Detector
    projections: np.ndarray
    poses: np.ndarray
    result: np.ndarray

    def reconstruct(self, z0: int, z1: int, fdk: FDKConfig) -> np.ndarray:
        """Slices ``[z0, z1)``, from the detector rows they project onto where it can tell."""
        beam = beam_of(self.geometry)
        assert beam is not None
        slab = _slab_grid(self.grid, z0, z1)
        if _windowable(beam):
            r0, r1 = _slab_rows(self.geometry, slab, self.detector, self.poses, 0, z1 - z0)
            rows = _detector_window(beam, self.detector, r0, r1)
        else:
            r0, r1, rows = 0, self.detector.nv, self.detector
        # The full detector sets per-column weights, the same for every slab.
        views = _RowView(self.projections, r0, r1)
        volume = _fdk(self.geometry, slab, rows, views, config=fdk, columns=self.detector)
        return np.asarray(volume)

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
        if device is not None:
            fdk = replace(fdk, devices=device)
        pending: Future[None] | None = None
        for z0, z1 in slabs:
            with jax.default_device(device):
                volume = self.reconstruct(z0, z1, fdk)
            if pending is not None:
                pending.result()
            pending = writer.submit(self.store, z0, z1, volume)
        if pending is not None:
            pending.result()

    def store(self, z0: int, z1: int, volume: np.ndarray) -> None:
        self.result[:, :, z0:z1] = volume


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


__all__ = ["FDKConfig", "FDKHostConfig", "fdk", "fdk_host", "view_weights"]
