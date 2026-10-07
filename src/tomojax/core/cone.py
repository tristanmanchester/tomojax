"""Cone-beam Joseph projection: rays from a point source sampled on voxel planes.

Each ray is sampled once per voxel-centre plane along the view's dominant axis
and interpolated bilinearly within the plane, as in the parallel Joseph model;
here the in-plane coordinates follow the ray from the source to its pixel, so
magnification changes from plane to plane. Line integrals are in physical
units.

Each ray is sampled along its own dominant axis, so a projection changes
continuously as a pose turns a ray through 45 degrees. Per-view geometry is a
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

from dataclasses import dataclass
from functools import cache, partial
from importlib.resources import files
import os
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid
    from tomojax.core.geometry.cone import ConeBeam

NC = 25
_HI = jax.lax.Precision.HIGHEST
_VIEW_CHUNK = 32  # adjoint views per launch, so their images stay in L2 (<= 128)


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
    return _coefficients(poses, frames, grid=grid, detector=detector)


@dataclass(frozen=True)
class ConeModel:
    """The source-detector arrangements of a cone-beam scan's views.

    ``parts`` holds ``(views, beam, detector)`` for each run of views (see
    :func:`cone_model`); hashable, so it can configure compiled programs.
    """

    parts: tuple[tuple[int | None, ConeBeam, Detector], ...]

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
    return None if parts is None else ConeModel(parts)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _coefficients(
    poses: jax.Array, frames: jax.Array, *, grid: Grid, detector: Detector
) -> jax.Array:
    """:func:`frame_coefficients`: ``frames`` are ``(views, 4, 3)``."""
    rot, trans = poses[:, :3, :3], poses[:, :3, 3]
    origin = jnp.asarray(grid_volume_origin(grid), jnp.float32)
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
    strides = jnp.asarray([grid.ny * grid.nz, grid.nz, 1], jnp.int32)
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
    for db in (0.0, 1.0):
        for dc in (0.0, 1.0):
            jb, jc = fl_b + db, fl_c + dc
            w = jnp.maximum(1 - jnp.abs(fb - jb), 0) * jnp.maximum(1 - jnp.abs(fc - jc), 0)
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
    return total.reshape(grid.nx, grid.ny, grid.nz)


# ----------------------------------------------------------------------------- CUDA


def use_cuda_cone() -> bool:
    """Return whether cone projections run as CUDA kernels (``TOMOJAX_CUDA_KERNELS=0`` disables)."""
    if os.environ.get("TOMOJAX_CUDA_KERNELS", "auto") == "0":
        return False
    from tomojax.core._cuda_joseph import cuda_gather_available

    return cuda_gather_available()


@cache
def _module() -> Any:
    import cupy as cp

    source = files("tomojax.core").joinpath("cone_kernels.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source)


def xla_stream(context: Any) -> Any:
    """Return XLA's CUDA stream for a ``buffer_callback`` context, as a CuPy stream."""
    import cupy as cp

    from tomojax.core._cuda_joseph import _XlaStream

    return cp.cuda.Stream.from_external(_XlaStream(int(context.stream)))


def _launch_forward(
    context: Any, out: Any, coeff: Any, volume: Any, *, grid: Grid, det: Detector
) -> None:
    import cupy as cp

    views = int(coeff.shape[0])
    with xla_stream(context):
        args = (
            cp.asarray(coeff), cp.asarray(volume), cp.asarray(out),
            np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
            np.int32(det.nu), np.int32(det.nv),
            np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
        )  # fmt: skip
        # Views vary fastest, rows slowest: the blocks running at once cover one band
        # of rows (one slab of the volume) in many views, which then shares the L2.
        _module().get_function("cone_forward")(
            (views, -(-det.nu // 32), -(-det.nv // 16)), (512,), args
        )


def _launch_adjoint(
    context: Any, out: Any, coeff: Any, images: Any, accumulate: Any, *, grid: Grid, det: Detector
) -> None:
    import cupy as cp

    views = int(coeff.shape[0])
    module = _module()
    with xla_stream(context):
        target, initial = cp.asarray(out), cp.asarray(accumulate)
        if target.data.ptr != initial.data.ptr:
            target[...] = initial
        coeff, images = cp.asarray(coeff), cp.asarray(images)
        weighted = cp.empty((views, det.nu, det.nv), cp.float32)  # (view, u, v)
        total = views * det.nu * det.nv
        module.get_function("weight_images")(
            (-(-total // 256),),
            (256,),
            (
                coeff, images, weighted, np.int32(views), np.int32(det.nu), np.int32(det.nv),
                np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
            ),
        )  # fmt: skip
        boxes = cp.empty((views, 3, 4), cp.int32)
        module.get_function("axis_boxes")(
            (views,), (256,), (coeff, boxes, np.int32(det.nu), np.int32(det.nv))
        )
        sizes = (grid.nx, grid.ny, grid.nz, det.nu, det.nv)
        dims = tuple(np.int32(x) for x in sizes)
        extent = (grid.nx, grid.ny, grid.nz)
        for start in range(0, views, _VIEW_CHUNK):
            count = min(_VIEW_CHUNK, views - start)
            part = (coeff[start : start + count], boxes[start : start + count], np.int32(count))
            i_part = weighted[start : start + count]
            for axis in (0, 1):
                na, nb = extent[axis], extent[1 - axis]
                module.get_function("sep_adjoint")(
                    (-(-grid.nz // 64), -(-nb // 64), na),
                    (256,),
                    (*part, np.int32(axis), i_part, target, *dims),
                )
            for axis in (0, 1, 2):
                # In-plane axes (b, c): the other two, in order.
                nb, nc = (extent[i] for i in range(3) if i != axis)
                module.get_function("plane_adjoint")(
                    (-(-nc // 32), -(-nb // 32), extent[axis]),
                    (128,),
                    (*part, np.int32(axis), i_part, target, *dims),
                )


def _forward_cuda(volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    call = buffer_callback(
        partial(_launch_forward, grid=grid, det=detector),
        jax.ShapeDtypeStruct((coeff.shape[0], detector.nv, detector.nu), jnp.float32),
        vmap_method="sequential",
    )
    return call(coeff, volume.astype(jnp.float32))


def _adjoint_cuda(
    images: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector, accumulate: jax.Array
) -> jax.Array:
    call = buffer_callback(
        partial(_launch_adjoint, grid=grid, det=detector),
        jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32),
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


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _forward_cuda_vjp(
    volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector
) -> jax.Array:
    return _forward_cuda(volume, coeff, grid, detector)


def _forward_cuda_fwd(
    volume: jax.Array, coeff: jax.Array, grid: Grid, detector: Detector
) -> tuple[jax.Array, jax.Array]:
    return _forward_cuda(volume, coeff, grid, detector), coeff


def _forward_cuda_bwd(
    grid: Grid, detector: Detector, coeff: jax.Array, cotangent: jax.Array
) -> tuple[jax.Array, jax.Array]:
    zeros = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    return _adjoint_cuda(cotangent, coeff, grid, detector, zeros), jnp.zeros_like(coeff)


_forward_cuda_vjp.defvjp(_forward_cuda_fwd, _forward_cuda_bwd)


def cone_project(
    volume: jax.Array,
    coeff: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str = "auto",
) -> jax.Array:
    """Project a volume through every view's coefficients; returns ``(views, nv, nu)``.

    The JAX backend differentiates in the volume and the coefficients (hence
    the poses). The CUDA backend differentiates in the volume only, through
    its matched transpose.
    """
    if _resolve(backend) == "cuda":
        return _forward_cuda_vjp(jnp.asarray(volume, jnp.float32), coeff, grid, detector)
    return _forward_jax(jnp.asarray(volume, jnp.float32), coeff, grid, detector)


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
        jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
        if accumulate is None
        else jnp.asarray(accumulate, jnp.float32)
    )
    images = jnp.asarray(images, jnp.float32)
    if _resolve(backend) == "cuda":
        return _adjoint_cuda(images, coeff, grid, detector, initial)
    return _adjoint_jax(images, coeff, grid, detector, initial)


__all__ = [
    "cone_backproject",
    "cone_coefficients",
    "cone_project",
    "use_cuda_cone",
    "xla_stream",
]
