"""Cone-beam Joseph projection: rays from a point source sampled on voxel planes.

Each ray is sampled once per voxel-centre plane along the view's dominant axis
and interpolated bilinearly within the plane, as in the parallel Joseph model;
here the in-plane coordinates follow the ray from the source to its pixel, so
magnification changes from plane to plane. Line integrals are in physical
units.

Each ray is sampled along its own dominant index-space axis. The quadrature is
piecewise smooth in poses: switching that axis can change the sample locations
and introduce a discontinuity in 3D. Per-view geometry is a
25-value coefficient row in voxel-index units of the object frame (source,
detector origin and pixel steps, a separable-view flag, and the detector-plane
projection used to bound voxel footprints). Rows come from
world_from_object poses and a :class:`~tomojax.core.geometry.cone.ConeBeam`,
in JAX, so they are differentiable with respect to the poses.

Two implementations agree to FP32 rounding:

- JAX: one code path for every view through flat-index gathers; the forward
  supports autodiff in the volume and the poses, and :func:`cone_backproject`
  is its explicit transpose (a plane-by-plane scatter, never a stored tape).
- CUDA (CuPy, launched on XLA's stream with ``buffer_callback``): a per-ray
  forward for every view, and a plane-tile scatter transpose plus a two-pass
  separable one for views whose detector v axis is parallel to the volume's z
  axis (unperturbed turntable scans), where the bilinear weight factors into a
  column and a z part. Forward and transpose use identical coordinate arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache, partial
from importlib.resources import files
import os
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core._cuda_joseph import xla_stream
from tomojax.core.geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid
    from tomojax.core.geometry.cone import ConeBeam

NC = 25
_HI = jax.lax.Precision.HIGHEST
_VIEW_CHUNK = 32  # adjoint views per launch, so their images stay in L2 (<= 32: VMAX)


def cone_coefficients(
    poses: jax.Array, grid: Grid, detector: Detector, beam: ConeBeam
) -> jax.Array:
    """Return ``(views, 25)`` FP32 per-view coefficients for the cone kernels."""
    return frame_coefficients(poses, beam_frame(beam, detector), grid, detector)


def beam_frame(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """``(4, 3)`` lab detector centre, unit u and v directions and source of ``beam``."""
    centre, u_dir, v_dir = beam.detector_frame(detector)
    return np.stack([centre, u_dir, v_dir, beam.source()]).astype(np.float32)


def frame_coefficients(
    poses: jax.Array, frames: jax.Array | np.ndarray, grid: Grid, detector: Detector
) -> jax.Array:
    """:func:`cone_coefficients` from lab frames (see :func:`beam_frame`), one per view.

    ``frames`` is ``(4, 3)``, shared by every view, or ``(views, 4, 3)``;
    ``detector`` gives the pixel count and pitch.
    """
    poses = jnp.asarray(poses, jnp.float32)
    frames = jnp.broadcast_to(jnp.asarray(frames, jnp.float32), (poses.shape[0], 4, 3))
    # A physical slab origin is numerical input, not a new kernel shape.
    origin = jnp.asarray(grid_volume_origin(grid), jnp.float32)
    kernel_grid = replace(grid, vol_origin=None, vol_center=None)
    return _coefficients(poses, frames, origin, grid=kernel_grid, detector=detector)


@dataclass(frozen=True)
class ConeModel:
    """The source-detector arrangements of a cone-beam scan's views.

    ``parts`` holds ``(views, beam, detector)`` for each run of views (see
    :func:`cone_model`); hashable, so it can configure compiled programs.
    """

    parts: tuple[tuple[int | None, ConeBeam, Detector], ...]

    def backproject(
        self,
        images: jax.Array,
        coeff: jax.Array,
        *,
        grid: Grid,
        detector: Detector,
        backend: str,
        accumulate: jax.Array,
    ) -> jax.Array:
        """Accumulate a batch through the exact cone transpose."""
        return cone_backproject(
            images, coeff, grid, detector, backend=backend, accumulate=accumulate
        )

    def backproject_pair(
        self,
        images: tuple[jax.Array, jax.Array],
        coeff: jax.Array,
        *,
        grid: Grid,
        detector: Detector,
        backend: str,
        accumulate: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        """Accumulate two batch transposes sharing ray geometry and image workspace."""
        return cone_backproject_pair(
            images, coeff, grid, detector, backend=backend, accumulate=accumulate
        )

    @property
    def detector(self) -> Detector:
        """The detector's pixel count and pitch (the parts differ in centre only)."""
        return self.parts[0][2]

    def frames(self, n_views: int) -> np.ndarray:
        """``(n_views, 4, 3)`` lab frame of every view (see :func:`beam_frame`)."""
        frames = [
            np.broadcast_to(beam_frame(beam, detector), (n_views if views is None else views, 4, 3))
            for views, beam, detector in self.parts
        ]
        stacked = np.concatenate(frames)
        if stacked.shape[0] != n_views:
            raise ValueError(
                f"the scan's arrangements cover {stacked.shape[0]} views, not {n_views}"
            )
        return stacked

    def magnification(self, n_views: int) -> np.ndarray:
        """Each view's magnification at the rotation axis."""
        values = [
            np.full(n_views if views is None else views, beam.magnification)
            for views, beam, _ in self.parts
        ]
        return np.concatenate(values)

    def coefficients(self, poses: jax.Array, grid: Grid) -> jax.Array:
        """Per-view coefficients of ``poses``, each view in its own arrangement."""
        return frame_coefficients(poses, self.frames(int(poses.shape[0])), grid, self.detector)


def cone_model(geometry: object, detector: Detector | None = None) -> ConeModel | None:
    """The cone arrangements of ``geometry``'s views, or None for a parallel beam.

    ``detector`` is the scan's detector, binned perhaps (default the
    geometry's); see :func:`~tomojax.core.geometry.cone.cone_parts`.
    """
    from tomojax.core.geometry.cone import cone_parts

    parts = cone_parts(geometry, detector)
    if parts is None:
        return None
    return ConeModel(parts)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _coefficients(
    poses: jax.Array, frames: jax.Array, origin: jax.Array, *, grid: Grid, detector: Detector
) -> jax.Array:
    """:func:`frame_coefficients`: ``frames`` are ``(views, 4, 3)``."""
    rot, trans = poses[:, :3, :3], poses[:, :3, 3]
    spacing = jnp.asarray([grid.vx, grid.vy, grid.vz], jnp.float32)
    centre, u_dir, v_dir, source = frames[:, 0], frames[:, 1], frames[:, 2], frames[:, 3]
    corner = (
        centre
        - (detector.nu - 1) / 2 * float(detector.du) * u_dir
        - (detector.nv - 1) / 2 * float(detector.dv) * v_dir
    )

    def to_object(point: jax.Array) -> jax.Array:  # lab points -> object index coordinates
        local = jnp.einsum("nji,nj->ni", rot, point - trans, precision=_HI)
        return (local - origin) / spacing

    def direction(vector: jax.Array) -> jax.Array:
        return jnp.einsum("nji,nj->ni", rot, vector, precision=_HI) / spacing

    S = to_object(source)
    D0 = to_object(corner)
    DU = direction(float(detector.du) * u_dir)
    DV = direction(float(detector.dv) * v_dir)
    # Views of an unperturbed turntable (detector v parallel to z, no ray steeper
    # than 45 degrees in z) let the CUDA kernels separate columns from z.
    u = jnp.arange(detector.nu, dtype=jnp.float32)
    row0 = D0[:, None, :] + u[None, :, None] * DU[:, None, :] - S[:, None, :]
    rows = jnp.stack([row0, row0 + (detector.nv - 1) * DV[:, None, :]], axis=1)
    lateral = jnp.min(jnp.maximum(jnp.abs(rows[..., 0]), jnp.abs(rows[..., 1])), axis=(1, 2))
    steep = jnp.max(jnp.abs(rows[..., 2]), axis=(1, 2))
    flag = (DV[:, 0] == 0) & (DV[:, 1] == 0) & (DV[:, 2] > 0) & (steep < lateral)
    axis = jax.lax.stop_gradient(flag.astype(jnp.float32))
    n = jnp.cross(DU, DV)
    h = jnp.sum(n * (D0 - S), axis=1)
    dual_u = jnp.cross(DV, n)
    Au = dual_u / jnp.sum(DU * dual_u, axis=1, keepdims=True)
    dual_v = jnp.cross(n, DU)
    Av = dual_v / jnp.sum(DV * dual_v, axis=1, keepdims=True)
    cu = jnp.sum((S - D0) * Au, axis=1)
    cv = jnp.sum((S - D0) * Av, axis=1)
    return jnp.concatenate(
        [S, D0, DU, DV, axis[:, None], n, h[:, None], Au, Av, cu[:, None], cv[:, None]], axis=1
    )


# ----------------------------------------------------------------------------- JAX


def _ray_frame(ray: jax.Array, grid: Grid) -> dict[str, jax.Array]:
    """Per-ray plane axis a (largest index-space component) and in-plane axes b, c."""
    a = jnp.argmax(jnp.abs(ray), axis=-1).astype(jnp.int32)
    b = jnp.where(a == 0, 1, 0)
    c = jnp.where(a == 2, 1, 2)
    sizes = jnp.asarray([grid.nx, grid.ny, grid.nz], jnp.int32)
    strides = jnp.asarray([1, grid.nx, grid.nx * grid.ny], jnp.int32)
    return {
        "a": a, "b": b, "c": c,
        "na": sizes[a], "nb": sizes[b], "nc": sizes[c],
        "sa": strides[a], "sb": strides[b], "sc": strides[c],
    }  # fmt: skip


def _rays(row: jax.Array, detector: Detector) -> jax.Array:
    """``(nv, nu, 3)`` ray vectors (pixel minus source) in index units."""
    u = jnp.arange(detector.nu, dtype=jnp.float32)
    v = jnp.arange(detector.nv, dtype=jnp.float32)
    pixel = row[3:6] + u[None, :, None] * row[6:9] + v[:, None, None] * row[9:12]
    return pixel - row[0:3]


def _component(x: jax.Array, axis: jax.Array) -> jax.Array:
    return jnp.take_along_axis(x, axis[..., None], axis=-1)[..., 0]


def _path_weight(ray: jax.Array, frame: dict[str, jax.Array], grid: Grid) -> jax.Array:
    spacing = jnp.asarray([grid.vx, grid.vy, grid.vz], jnp.float32)
    ra = _component(ray, frame["a"])
    return jnp.linalg.norm(ray * spacing, axis=-1) / jnp.maximum(jnp.abs(ra), 1e-12)


def _plane_samples(
    row: jax.Array, frame: dict[str, jax.Array], ray: jax.Array, k: jax.Array
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Flat indices and weights of the four bilinear taps of every ray on its plane k."""
    S = jnp.broadcast_to(row[0:3], ray.shape)
    ra = _component(ray, frame["a"])
    ra = jnp.where(jnp.abs(ra) < 1e-12, 1e-12, ra)
    t = (k - _component(S, frame["a"])) / ra
    fb = _component(S, frame["b"]) + t * _component(ray, frame["b"])
    fc = _component(S, frame["c"]) + t * _component(ray, frame["c"])
    fl_b, fl_c = jnp.floor(fb), jnp.floor(fc)
    indices, weights = [], []
    # The selected bilinear cell has a partition-of-unity derivative, including
    # at integer coordinates. Differentiating max(1-abs(x), 0) instead gives
    # half a tap at the support endpoint and a spurious gradient on constants.
    wb, wc = fb - fl_b, fc - fl_c
    for db in (0.0, 1.0):
        for dc in (0.0, 1.0):
            jb, jc = fl_b + db, fl_c + dc
            w = (wb if db else 1 - wb) * (wc if dc else 1 - wc)
            jb_i, jc_i = jb.astype(jnp.int32), jc.astype(jnp.int32)
            valid = (
                (jb_i >= 0) & (jb_i < frame["nb"]) & (jc_i >= 0) & (jc_i < frame["nc"])
                & (k < frame["na"])
            )  # fmt: skip
            index = k.astype(jnp.int32) * frame["sa"] + jb_i * frame["sb"] + jc_i * frame["sc"]
            indices.append(jnp.where(valid, index, 0))
            weights.append(jnp.where(valid, w, 0.0))
    return indices, weights


def _forward_view_jax(
    volume: jax.Array, row: jax.Array, grid: Grid, detector: Detector
) -> jax.Array:
    ray = _rays(row, detector)
    frame = _ray_frame(jax.lax.stop_gradient(ray), grid)
    flat = volume.reshape(-1)
    planes = max(grid.nx, grid.ny, grid.nz)

    def step(total: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        indices, weights = _plane_samples(row, frame, ray, k)
        for index, weight in zip(indices, weights, strict=True):
            total = total + flat[index] * weight
        return total, None

    total, _ = jax.lax.scan(
        step,
        jnp.zeros(ray.shape[:-1], jnp.float32),
        jnp.arange(planes, dtype=jnp.float32),
    )
    return total * _path_weight(ray, frame, grid)


def _forward_jax(volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    return jax.vmap(lambda row: _forward_view_jax(volume, row, grid, detector))(coeff)


def _adjoint_jax(
    images: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector, accumulate: jax.Array
) -> jax.Array:
    """Explicit transpose of :func:`_forward_jax`: scatter each plane's taps."""
    rays = jax.vmap(lambda row: _rays(row, detector))(coeff)
    frames = _ray_frame(rays, grid)
    values = images * _path_weight(rays, frames, grid)
    flat = accumulate.reshape(-1)

    def step(total: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        indices, weights = jax.vmap(lambda row, frame, ray: _plane_samples(row, frame, ray, k))(
            coeff, frames, rays
        )
        for index, weight in zip(indices, weights, strict=True):
            total = total.at[index.reshape(-1)].add((values * weight).reshape(-1))
        return total, None

    planes = max(grid.nx, grid.ny, grid.nz)
    total, _ = jax.lax.scan(step, flat, jnp.arange(planes, dtype=jnp.float32))
    return total.reshape(grid.nz, grid.ny, grid.nx)


# ----------------------------------------------------------------------------- CUDA


def use_cuda_cone() -> bool:
    """Return whether cone projections run as CUDA kernels (``TOMOJAX_CUDA_KERNELS=0`` disables)."""
    if os.environ.get("TOMOJAX_CUDA_KERNELS", "auto") == "0":
        return False
    from tomojax.core._cuda_joseph import cuda_gather_available

    return cuda_gather_available()


@cache
def cuda_module() -> Any:
    """Shared cone CUDA kernels, compiled lazily on the active device."""
    import cupy as cp

    source = files("tomojax.core").joinpath("cone_kernels.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source)


def forward_into(out: Any, coeff: Any, volume: Any, *, grid: Grid, det: Detector) -> None:
    """Launch the forward kernels on CuPy's current stream: ``out`` is ``(views, nv, nu)``.

    The arrays are CuPy arrays (or CUDA buffers) on the current device;
    ``volume`` is C-ordered ``(nz, ny, nx)``.
    """
    import cupy as cp

    views = int(coeff.shape[0])
    args = (
        cp.asarray(coeff), cp.asarray(volume), cp.asarray(out),
        np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
        np.int32(det.nu), np.int32(det.nv),
        np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
    )  # fmt: skip
    # Views vary fastest, rows slowest: the blocks running at once cover one band
    # of rows (one slab of the volume) in many views, which then shares the L2.
    cuda_module().get_function("cone_forward")(
        (views, -(-det.nu // 32), -(-det.nv // 16)), (512,), args
    )


def _launch_forward(context: Any, out: Any, *args: Any, **options: Any) -> None:
    with xla_stream(context, out):
        forward_into(out, *args, **options)


def box_forward_into(out: Any, coeff: Any, *, grid: Grid, det: Detector) -> None:
    """Launch the uniform-support forward on CuPy's current stream (see :func:`forward_into`)."""
    import cupy as cp

    cuda_module().get_function("cone_box_forward")(
        (int(coeff.shape[0]), -(-(det.nu * det.nv) // 128)),
        (128,),
        (
            cp.asarray(coeff), cp.asarray(out),
            np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
            np.int32(det.nu), np.int32(det.nv),
            np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
        ),
    )  # fmt: skip


def _launch_box_forward(context: Any, out: Any, *args: Any, **options: Any) -> None:
    with xla_stream(context, out):
        box_forward_into(out, *args, **options)


def _separable_adjoint_kernel(grid: Grid, detector: Detector) -> str:
    """Keep the wider tile for densely sampled detector-row stacks.

    The compact tile's two-cell lanes save work on ordinary row stacks, but
    cost more for wide resampling support. Use a conservative stack-size
    threshold, tighter on small volumes where geometry caching amortizes less.
    Both tiles span 64 planes; up to 32 planes use their 32-plane versions
    (one tile either way), so no lanes compute planes outside the volume.
    """
    row_ratio = 2 if grid.nx * grid.ny * grid.nz < 2**21 else 4
    name = "sep_adjoint" if detector.nv > row_ratio * grid.nz else "sep_adjoint_cached"
    return name + "_thin" if grid.nz <= 32 else name


def _weighted_images(
    module: Any, coeff: Any, images: tuple[Any, ...], grid: Grid, det: Detector
) -> tuple[Any, Any]:
    import cupy as cp

    views = int(coeff.shape[0])
    paired = len(images) == 2
    shape = ((2,) if paired else ()) + (views, det.nu, det.nv)
    weighted = cp.empty(shape, cp.float32)  # optional channel, then (view, u, v)
    tiles_u, tiles_v = -(-det.nu // 32), -(-det.nv // 32)
    ntiles = tiles_u * tiles_v
    # The first `views` entries are the final boxes; with several tiles per
    # view the rest holds each tile's bounds. One tile writes its box directly.
    boxes = cp.empty((views * (ntiles + 1 if ntiles > 1 else 1), 3, 4), cp.int32)
    partial = boxes[views:] if ntiles > 1 else boxes
    kernel = "pair_weight_images_boxes_tiled" if paired else "weight_images_boxes_tiled"
    module.get_function(kernel)(
        (views, tiles_u, tiles_v),
        (32, 8),
        (
            coeff, *(cp.asarray(image) for image in images), weighted,
            np.int32(views), np.int32(det.nu), np.int32(det.nv),
            np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz), partial,
        ),
    )  # fmt: skip
    if ntiles > 1:
        module.get_function("reduce_axis_boxes_tiled")(
            (views,),
            (32, 12),
            (partial, boxes, np.int32(ntiles), np.int32(det.nu), np.int32(det.nv)),
        )
    return weighted, boxes[:views]


def adjoint_into(
    out: Any,
    coeff: Any,
    images: Any,
    accumulate: Any,
    *,
    grid: Grid,
    det: Detector,
) -> None:
    """Set ``out`` to ``accumulate`` plus the exact transpose, on CuPy's current stream.

    ``images`` is ``(views, nv, nu)``; ``out`` and ``accumulate`` are
    ``(nz, ny, nx)`` and may be the same array.
    """
    import cupy as cp

    views = int(coeff.shape[0])
    module = cuda_module()
    separable_adjoint = module.get_function(_separable_adjoint_kernel(grid, det))
    target, initial = cp.asarray(out), cp.asarray(accumulate)
    if target.data.ptr != initial.data.ptr:
        target[...] = initial
    coeff, images = cp.asarray(coeff), cp.asarray(images)
    weighted, boxes = _weighted_images(module, coeff, (images,), grid, det)
    sizes = (grid.nx, grid.ny, grid.nz, det.nu, det.nv)
    dims = tuple(np.int32(x) for x in sizes)
    extent = (grid.nx, grid.ny, grid.nz)
    for start in range(0, views, _VIEW_CHUNK):
        count = min(_VIEW_CHUNK, views - start)
        part = (coeff[start : start + count], boxes[start : start + count], np.int32(count))
        i_part = weighted[start : start + count]
        for axis in (0, 1):
            na, nb = extent[axis], extent[1 - axis]
            separable_adjoint(
                (-(-nb // 64), na, -(-grid.nz // 64)),
                (256,),
                (*part, np.int32(axis), i_part, target, *dims),
            )
        for axis in (0, 1, 2):
            # In-plane axes (b, c): the other two, in order.
            nb, nc = (extent[i] for i in range(3) if i != axis)
            wide = axis != 2 and grid.nz >= 128
            tile_c = 64 if wide else 32
            module.get_function("plane_adjoint_wide" if wide else "plane_adjoint")(
                (
                    (-(-extent[axis] // 2), -(-nb // 15), -(-nc // tile_c))
                    if axis == 0
                    else (-(-nb // 15), -(-extent[axis] // 2), -(-nc // tile_c))
                ),
                (128,),
                (*part, np.int32(axis), i_part, target, *dims),
            )


def _launch_adjoint(context: Any, out: Any, *args: Any, **options: Any) -> None:
    with xla_stream(context, out):
        adjoint_into(out, *args, **options)


def adjoint_pair_into(
    out: tuple[Any, Any],
    coeff: Any,
    images0: Any,
    images1: Any,
    initial0: Any,
    initial1: Any,
    *,
    grid: Grid,
    det: Detector,
) -> None:
    """Two exact transposes sharing ray geometry, on CuPy's current stream.

    ``out[i]`` becomes ``initial{i}`` plus the transpose of ``images{i}``.
    """
    import cupy as cp

    module = cuda_module()
    targets = tuple(cp.asarray(value) for value in out)
    for target, value in zip(targets, (initial0, initial1), strict=True):
        initial = cp.asarray(value)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
    coeff = cp.asarray(coeff)
    weighted, boxes = _weighted_images(module, coeff, (images0, images1), grid, det)
    # Channel-major planes also feed the original separable kernels without copies.
    separate = (weighted[0], weighted[1])
    name = _separable_adjoint_kernel(grid, det)
    paired_separable = name.startswith("sep_adjoint_cached")
    separable = module.get_function("pair_sep_adjoint_cached" if paired_separable else name)
    dims = tuple(np.int32(x) for x in (grid.nx, grid.ny, grid.nz, det.nu, det.nv))
    extent = (grid.nx, grid.ny, grid.nz)
    views = int(coeff.shape[0])
    for start in range(0, views, _VIEW_CHUNK):
        count = min(_VIEW_CHUNK, views - start)
        part = (coeff[start : start + count], boxes[start : start + count], np.int32(count))
        for axis in (0, 1):
            depth = 32 if paired_separable else 64
            shape = (-(-extent[1 - axis] // 64), extent[axis], -(-grid.nz // depth))
            if paired_separable:
                separable(
                    shape,
                    (256,),
                    (
                        *part,
                        np.int32(axis),
                        *(image[start : start + count] for image in separate),
                        *targets,
                        *dims,
                    ),
                )
            else:
                for image, target in zip(separate, targets, strict=True):
                    separable(
                        shape,
                        (256,),
                        (*part, np.int32(axis), image[start : start + count], target, *dims),
                    )
        for axis in (0, 1, 2):
            nb, nc = (extent[i] for i in range(3) if i != axis)
            module.get_function("pair_plane_adjoint")(
                (
                    (extent[axis], -(-nb // 32), -(-nc // 32))
                    if axis == 0
                    else (-(-nb // 32), extent[axis], -(-nc // 32))
                ),
                (128,),
                (
                    *part,
                    np.int32(axis),
                    *(image[start : start + count] for image in separate),
                    *targets,
                    *dims,
                ),
            )


def _launch_adjoint_pair(context: Any, out: Any, *args: Any, **options: Any) -> None:
    with xla_stream(context, out[0]):
        adjoint_pair_into(out, *args, **options)


def forward_cuda(
    volume: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
) -> jax.Array:
    """Raw CUDA projection; AD rules wrap this same launch."""
    call = buffer_callback(
        partial(_launch_forward, grid=grid, det=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nv, detector.nu), jnp.float32),
        vmap_method="sequential",
    )
    args = (coeff, volume.astype(jnp.float32))
    return call(*args)


def _launch_pose_columns(
    context: Any, out: Any, coeff: Any, tangent: Any, volume: Any, *, grid: Grid, det: Detector
) -> None:
    import cupy as cp

    with xla_stream(context, out):
        coeff, tangent, volume, target = (cp.asarray(x) for x in (coeff, tangent, volume, out))
        batch = np.broadcast_shapes(coeff.shape[:-2], tangent.shape[:-3], volume.shape[:-3])
        views, width = coeff.shape[-2], tangent.shape[-2]
        coeff = cp.broadcast_to(coeff, (*batch, views, NC))
        tangent = cp.broadcast_to(tangent, (*batch, views, width, 12))
        kernel = cuda_module().get_function("cone_pose_columns")
        shape = (grid.nz, grid.ny, grid.nx)
        dimensions = (
            np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
            np.int32(det.nu), np.int32(det.nv), np.int32(width),
            np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
        )  # fmt: skip

        def launch(c: Any, d: Any, x: Any, y: Any) -> None:
            kernel(
                (c.shape[0], -(-det.nu // 16), -(-det.nv // 8)),
                (128,),
                (c, d, x, y, *dimensions),
            )

        if volume.size == grid.nx * grid.ny * grid.nz:
            # vmap over views or directions shares the volume: concatenate only
            # the tiny geometry arrays and launch all rays together.
            launch(
                coeff.reshape(-1, NC), tangent.reshape(-1, width, 12), volume.reshape(shape), target
            )
        else:
            # Independently mapped volumes must not be broadcast/materialized.
            # Their batches share a callback, retaining ordinary vmap semantics.
            volume = cp.broadcast_to(volume, (*batch, *shape))
            for index in np.ndindex(batch):
                launch(coeff[index], tangent[index], volume[index], target[index])


def cone_pose_columns(
    volume: jax.Array, coeff: jax.Array, *, tangent: jax.Array, grid: Grid, detector: Detector
) -> jax.Array:
    """CUDA value and pose columns from one traversal of each ray.

    ``tangent`` holds ``(views, directions, 12)`` derivatives of the source,
    detector origin and pixel steps. As for the JAX projector, the ray's
    dominant axis is held fixed: its switches are not differentiable.
    Interpolation derivatives use the current bilinear cell.
    """
    call = buffer_callback(
        partial(_launch_pose_columns, grid=grid, det=detector),
        jax.ShapeDtypeStruct(
            (coeff.shape[0], 1 + tangent.shape[1], detector.nv, detector.nu), jnp.float32
        ),
        vmap_method="expand_dims",
    )
    return call(coeff, tangent, volume.astype(jnp.float32))


def _adjoint_cuda(
    images: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    accumulate: jax.Array,
) -> jax.Array:
    call = buffer_callback(
        partial(
            _launch_adjoint,
            grid=grid,
            det=detector,
        ),
        jax.ShapeDtypeStruct((grid.nz, grid.ny, grid.nx), jnp.float32),
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    return call(coeff, images.astype(jnp.float32), accumulate.astype(jnp.float32))


# ----------------------------------------------------------------------------- public


def _resolve(backend: str) -> str:
    if backend not in {"auto", "jax", "cuda"}:
        raise ValueError("cone backend must be 'auto', 'jax' or 'cuda'")
    if backend == "auto":
        return "cuda" if use_cuda_cone() else "jax"
    if backend == "cuda" and not use_cuda_cone():
        raise ValueError("the CUDA cone kernels need CuPy on a CUDA device")
    return backend


def cone_project(
    volume: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
) -> jax.Array:
    """Project a volume through every view's coefficients; returns ``(views, nv, nu)``.

    Both backends differentiate in the volume and coefficients (hence poses).
    CUDA supports first-order AD with the matched volume transpose. Pose
    derivatives hold the dominant ray axis and interpolation cell fixed; use
    JAX for higher derivatives.
    """
    if _resolve(backend) == "cuda":
        from tomojax.core._cone_autodiff import differentiable_forward

        return differentiable_forward(coeff, jnp.asarray(volume, jnp.float32), grid, detector)
    return _forward_jax(jnp.asarray(volume, jnp.float32), coeff, grid, detector)


def cone_project_ones(
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
) -> jax.Array:
    """Evaluate the same projector on uniform unit support without storing a volume.

    CUDA counts fully supported planes and retains ordinary sampling arithmetic
    at the box boundary. This normalization-only shortcut does not differentiate
    in the coefficients; use cone_project on a unit volume for pose derivatives.
    """
    if _resolve(backend) != "cuda":
        ones = jnp.ones((grid.nz, grid.ny, grid.nx), jnp.float32)
        return cone_project(ones, coeff, grid, detector, backend="jax")
    call = buffer_callback(
        partial(_launch_box_forward, grid=grid, det=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nv, detector.nu), jnp.float32),
        vmap_method="sequential",
    )
    return call(jax.lax.stop_gradient(coeff).astype(jnp.float32))


def cone_backproject(
    images: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
    accumulate: jax.Array | None = None,
) -> jax.Array:
    """Return ``accumulate`` plus the exact transpose of :func:`cone_project`."""
    initial = (
        jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
        if accumulate is None
        else jnp.asarray(accumulate, jnp.float32)
    )
    images = jnp.asarray(images, jnp.float32)
    if _resolve(backend) == "cuda":
        return _adjoint_cuda(images, coeff, grid, detector, initial)
    return _adjoint_jax(images, coeff, grid, detector, initial)


def cone_backproject_pair(
    images: tuple[jax.Array, jax.Array],
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
    accumulate: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Two exact transposes sharing CUDA ray geometry, with independent accumulators."""
    if accumulate is None:
        zero = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
        initial = zero, zero
    else:
        initial = accumulate
    if _resolve(backend) != "cuda":
        first, second = (
            cone_backproject(image, coeff, grid, detector, backend="jax", accumulate=value)
            for image, value in zip(images, initial, strict=True)
        )
        return first, second
    return _adjoint_pair_cuda(images, coeff, grid, detector, initial)


def _adjoint_pair_cuda(
    images: tuple[jax.Array, jax.Array],
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    initial: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    shape = jax.ShapeDtypeStruct((grid.nz, grid.ny, grid.nx), jnp.float32)
    call = buffer_callback(
        partial(
            _launch_adjoint_pair,
            grid=grid,
            det=detector,
        ),
        (shape, shape),
        input_output_aliases={3: 0, 4: 1},
        vmap_method="sequential",
    )
    return call(coeff, *(jnp.asarray(x, jnp.float32) for x in (*images, *initial)))


__all__ = [
    "cone_backproject",
    "cone_backproject_pair",
    "cone_coefficients",
    "cone_project",
    "use_cuda_cone",
    "xla_stream",
]
