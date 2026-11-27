"""Batched image and active-geometry operators for the joint fit.

Only selected local parameters and shared calibration parameters have columns.
The default has two translations per view; cone translations also include
beam depth. Local rotations, when selected, compose in the sample frame and
use geometry-scaled pixel units. Global center, tilt and roll columns are
shared by the whole scan through the parameter layout's broadcast and summed
score. Coefficients and exact active-coordinate derivatives are computed from
the physical geometry on the device.

Views are processed in fixed batches padded with zero-weight views. Only
per-batch residuals and raw pose columns are temporary; frozen IRLS weights and
the whitened FP32 pose columns stay resident, so one weighted forward pass is
reused within each Schur matvec. Scores, Gram blocks and matvecs use those same
columns. The data loss is a quadratic core with a log tail.
"""

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING, NamedTuple

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.core.registration import length_unit, planes, pose_columns
from tomojax.core.volume_layout import transpose_volume
from tomojax.wire.arrays import Axis, Uniform

from . import _common
from ._parameters import Parameters
from ._projection import (
    Coefficients,
    Projection,
    cone_parameter_scales,
)

if TYPE_CHECKING:
    from tomojax.alignment._config import Interpolation
    from tomojax.core.geometry.base import Detector, Grid

    from ._stacks import ArrayStack, StackSpec


class AffineGeometry(NamedTuple):
    """Fixed parallel-ray coefficients and tangents of the two changing offsets."""

    rays: jax.Array
    offsets: jax.Array


type ViewBatch = tuple[
    jax.Array | AffineGeometry, jax.Array, jax.Array, jax.Array, jax.Array | None
]
type IndexedViewBatch = tuple[jax.Array, *ViewBatch]
type LinearBuffers = tuple[jax.Array, jax.Array, ArrayStack, ArrayStack, ArrayStack]
type _LinearCarry = tuple[jax.Array, jax.Array, tuple[ArrayStack, ArrayStack, ArrayStack]]
type _BatchLinearization = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]


def robust_value(residual: jax.Array, precision: jax.Array, delta: float | jax.Array) -> jax.Array:
    """Masked quadratic-core, log-tail loss; FP64 sum."""
    r = jnp.where(precision > 0, residual, 0).astype(jnp.float64)
    d = jnp.asarray(delta, jnp.float64)
    a = jnp.abs(r)
    loss = jnp.where(a <= d, 0.5 * r * r, d * d * (0.5 + jnp.log(jnp.maximum(a, d) / d)))
    return jnp.sum(precision.astype(jnp.float64) * loss, dtype=jnp.float64)


def robust_batch(
    residual: jax.Array,
    columns: jax.Array,
    precision: jax.Array,
    delta: float | jax.Array,
    *,
    half: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """IRLS square-root weights, and weighted residuals and columns (view, k, v, u).

    ``half`` rounds the weights and weighted columns to FP16 values (5e-4
    relative) for storage; the weighted residual uses the rounded weights.
    """
    r = jnp.where(precision > 0, residual, 0)
    a = jnp.abs(r.astype(jnp.float64))
    d = jnp.asarray(delta, jnp.float64)
    sw = jnp.sqrt(precision.astype(jnp.float64) * jnp.minimum(1.0, d / jnp.maximum(a, d)) ** 2)
    # reduce_precision rounds even where XLA would keep excess precision.
    rounded = partial(jax.lax.reduce_precision, exponent_bits=5, mantissa_bits=10)
    sw = rounded(sw.astype(jnp.float32)) if half else sw.astype(jnp.float32)
    j = (columns * sw[:, None]).astype(jnp.float32)
    return (sw * r).astype(jnp.float32), sw, rounded(j) if half else j


def _offset_and_ramp(
    images: jax.Array, s: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per view (and leading column): the s-whitened fit coefficients of [s, s * v] to images."""
    v = _common.rows(s.shape[-2])
    s = s.astype(jnp.float64)
    extra = (slice(None),) + (None,) * (images.ndim - s.ndim)
    w = s * s
    g00, g01, g11 = _common.view_grams(w)
    det = jnp.where(_common.profiled_views(w), g00 * g11 - g01 * g01, jnp.inf)
    ss, sv = s[extra], (s * v)[extra]
    b0 = jnp.sum(ss * images, axis=(-2, -1))
    b1 = jnp.sum(sv * images, axis=(-2, -1))
    c0 = (g11[extra] * b0 - g01[extra] * b1) / det[extra]
    c1 = (g00[extra] * b1 - g01[extra] * b0) / det[extra]
    return c0, c1, ss, sv


def profile(images: jax.Array, s: jax.Array) -> jax.Array:
    """Whitened images less each view's weighted fit by a constant and a ramp down the rows.

    ``images`` are (view, ..., v, u) and already whitened by ``s`` (view, v, u):
    this projects out the span of ``s`` and ``s * v`` per view, v running from
    -1 to 1 down the rows. An additive offset (flux) and vertical ramp (phase
    retrieval, flat drift) are nuisance the line integrals do not describe;
    eliminating them inside each solve leaves the poses only what they cannot
    explain. Views without weight are returned unchanged.
    """
    c0, c1, ss, sv = _offset_and_ramp(images, s)
    return (images - ss * c0[..., None, None] - sv * c1[..., None, None]).astype(images.dtype)


def profiled_residual(residual: jax.Array, precision: jax.Array) -> jax.Array:
    """Residuals less each view's precision-weighted constant and row ramp."""
    root = jnp.sqrt(precision.astype(jnp.float64))
    whitened = profile(root * residual, root)
    return jnp.where(precision > 0, whitened / jnp.where(precision > 0, root, 1), 0).astype(
        residual.dtype
    )


def column_score(columns: jax.Array, images: jax.Array) -> jax.Array:
    """Per-view J^T r, FP64."""
    return jnp.sum(columns.astype(jnp.float64) * images[:, None].astype(jnp.float64), axis=(-2, -1))


def column_gram(columns: jax.Array) -> jax.Array:
    """Per-view J^T J, FP64."""
    flat = columns.reshape((*columns.shape[:2], -1)).astype(jnp.float64)
    return jnp.einsum("nik,njk->nij", flat, flat)


def _apply_columns(columns: jax.Array, p: jax.Array) -> jax.Array:
    columns, p = columns.astype(jnp.float32), p.astype(jnp.float32)
    image = columns[:, 0] * p[:, 0, None, None]
    # These few directions are an image expression, not a batched GEMM. Keep
    # products visible to XLA so whitening/residual arithmetic can fuse with it.
    for direction in range(1, columns.shape[1]):
        image = image + columns[:, direction] * p[:, direction, None, None]
    return image


class Operators:
    """Projector, transpose, pose columns and the solver's whole-scan reductions.

    ``nominal`` are the (views, 4, 4) nominal poses; ``native`` the native
    detector pitches; ``lever`` native pixels per radian of rotation.
    """

    # FP16 storage of the step's weights and columns; tests turn it off to check the calculus.
    _half = True

    def preview(self, x: jax.Array, c: jax.Array) -> dict[str, progress.Sample]:
        """Working image and incremental per-view pose in public units."""
        images = planes(x, self.grid)
        table, _ = self.parameters.physical(np.asarray(c), np.asarray(self.scales[: self.n]))
        table[:, :3] = np.rad2deg(table[:, :3])
        images["poses"] = progress.Sample(
            table,
            (
                Axis("view", "acquisition", "index", Uniform(0, 1)),
                Axis("column", "pose", "index", Uniform(0, 1)),
            ),
            "mixed",
            "poses",
            columns=pose_columns(length_unit.get()),
        )
        return images

    def __init__(
        self,
        grid: Grid,
        detector: Detector,
        nominal: np.ndarray | jax.Array,
        *,
        lever: float,
        native: tuple[float, float],
        batch_size: int = 8,
        frames: np.ndarray | jax.Array | None = None,
        scales: np.ndarray | jax.Array | None = None,
        interpolation: Interpolation | None = "linear",
        parameters: Parameters | None = None,
        detector_roll: float = 0.0,
        detector_pivot: tuple[float, float] | None = None,
        column_attenuation: float = 0.0,
        masked: bool = False,
    ) -> None:
        self.grid, self.detector = grid, detector
        self.shape = (grid.nz, grid.ny, grid.nx)
        self.n = len(nominal)
        self.parameters = parameters or Parameters(
            self.n, ("translations", "rotations"), cone=frames is not None, lever=lever
        )
        self.detector_roll = float(detector_roll)
        pivot_u, pivot_v = detector.center if detector_pivot is None else detector_pivot
        self.detector_pivot = (pivot_u, pivot_v)
        self.projection = Projection(
            grid,
            detector,
            cone=frames is not None,
            interpolation=interpolation,
            column_attenuation=column_attenuation,
            masked=masked,
        )
        self.z_fast = self.projection.backend == "cuda" and not self.projection.cone
        self.project, self.back = self.projection.project, self.projection.back
        self.backend, self.dofs = self.projection.backend, self.parameters.k
        # Bound the temporary value/pose columns to about 32 MiB when more
        # than one view fits.
        column_bytes = detector.nv * detector.nu * (1 + self.dofs) * 4
        self.batch = min(batch_size, self.n, max(1, 32 * 2**20 // column_bytes))
        self.nb = math.ceil(self.n / self.batch)
        self.padded_n = self.nb * self.batch
        self.image_shape = (self.padded_n, detector.nv, detector.nu)
        du, dv = native
        self.lever, self.native = float(lever), (float(du), float(dv))
        self.ratios = (self.native[0] / detector.du, self.native[1] / detector.dv)
        self.default_scales = jnp.asarray([1 / self.lever] * 3 + list(self.native), jnp.float32)
        if scales is None:
            scales = (
                self.default_scales
                if frames is None
                else cone_parameter_scales(nominal, frames, grid, self.native)
            )
        scales = jnp.broadcast_to(
            jnp.asarray(scales, jnp.float32), (self.n, self.parameters.full_dofs)
        )
        extra = self.padded_n - self.n
        nominal = jnp.asarray(nominal, jnp.float32)
        self.nominal = jnp.concatenate([nominal, jnp.repeat(nominal[:1], extra, axis=0)])
        self.scales = jnp.concatenate([scales, jnp.repeat(scales[:1], extra, axis=0)])
        self.frames = None
        if frames is not None:
            frames = jnp.broadcast_to(jnp.asarray(frames, jnp.float32), (self.n, 4, 3))
            self.frames = jnp.concatenate([frames, jnp.repeat(frames[:1], extra, axis=0)])
        self.valid = jnp.arange(self.padded_n) < self.n
        self.affine = None
        if frames is None and not {"tilt", "roll", "rotations"}.intersection(self.parameters.fit):
            # For this requested model only coefficient offsets change. Freeze
            # its exact affine map once; no pose-matrix AD or inverse belongs in
            # later image/pose iterations. The projector and transpose are shared.
            self.affine = jit(self._affine_geometry)(self.nominal, self.scales)
        self._geometry = jit(self._geometry)
        # A level's outputs are written into buffers it owns, donated (``_stacks``); a buffer
        # the program does not read must still be kept as an argument to receive its output.
        owned = partial(jit, keep_unused=True)
        self.sensitivity = owned(self._sensitivity, donate_argnums=(1,))
        self._initial_gradient = jit(self._initial_gradient)
        self._score_inputs = jit(self._score_inputs)
        self.linearize = owned(self._linearize, donate_argnums=(5,))
        self.objective = jit(self._objective)
        self.common_images = owned(self._common_images, donate_argnums=(2,))
        self.shared_data = jit(self._shared_data)
        self.forward_scores = jit(self.forward_scores, donate_argnums=(5,))
        self.view_summary = jit(self._view_summary)
        self.back_rows = jit(self.back_rows)
        self._pose_back = jit(self._pose_back)

    def stack_specs(self, precision: np.ndarray | jax.Array | None) -> dict[str, StackSpec]:
        """Shape and dtype of each stack a level holds (``_stacks``); precision only when given."""
        d = self.detector
        specs: dict[str, StackSpec] = {
            "y": (self.image_shape, np.float32),
            "residual": (self.image_shape, np.float32),
            "sw": (self.image_shape, np.float16 if self._half else np.float32),
            "j": ((self.padded_n, self.dofs, d.nv, d.nu), np.float16 if self._half else np.float32),
            "r": (self.image_shape, np.float32),
        }
        if precision is not None:
            specs["precision"] = (self.image_shape, np.float32)
        return specs

    def stack(self, array: np.ndarray | jax.Array) -> ArrayStack:
        """A padded per-view array as a resident :class:`Stack` (``_stacks``)."""
        from ._stacks import Stack

        array = jnp.asarray(array)
        return Stack(array, host=False, shape=array.shape, dtype=array.dtype, batch=self.batch)

    def batch_of(
        self, stack: ArrayStack | None, i: int | jax.Array, first: jax.Array | None = None
    ) -> jax.Array | None:
        """Batch ``i`` of a stack, or None for uniform precision."""
        return None if stack is None else stack.read(i, first)

    def pad_poses(self, q: jax.Array | np.ndarray) -> jax.Array:
        """Expand a compact state, then pad its per-view rows with zero-weight views."""
        q = self.parameters.expand(q).astype(jnp.float32)
        return jnp.pad(q, ((0, self.padded_n - self.n), (0, 0)))

    def pad_images(self, y: np.ndarray | jax.Array) -> np.ndarray:
        """Pad on the host with zero images, which carry zero precision."""
        y = np.asarray(y, np.float32)
        if self.n == self.padded_n:
            return y
        return np.pad(y, ((0, self.padded_n - self.n), (0, 0), (0, 0)))

    def pad_precision(self, precision: np.ndarray | jax.Array | None) -> np.ndarray | None:
        """Keep uniform precision implicit; pad explicitly supplied weights."""
        return None if precision is None else self.pad_images(precision)

    def batch_precision(self, precision: jax.Array | None, valid: jax.Array) -> jax.Array:
        """One batch's weights, with zero precision for padded views."""
        if precision is not None:
            return precision
        return jnp.broadcast_to(
            valid[:, None, None].astype(jnp.float32),
            (len(valid), self.detector.nv, self.detector.nu),
        )

    def batches[Tree](self, a: Tree) -> Tree:
        """Reshape a padded per-view array tree into (batches, batch, ...)."""

        def batch(x: jax.Array) -> jax.Array:
            return x.reshape((-1, self.batch, *x.shape[1:]))

        return jax.tree.map(batch, a)

    def prepare(self, q: jax.Array | Coefficients) -> Coefficients:
        """Freeze geometry for the image-only loops."""
        if isinstance(q, Coefficients):
            return q
        return self._geometry(q)

    def _geometry(self, q: jax.Array) -> Coefficients:
        """Freeze the small coefficient packet for one linearized image solve."""

        def coefficients(batch: ViewBatch) -> Coefficients:
            return self.coefficients(*batch)

        packets: Coefficients = jax.lax.map(coefficients, self.views(q))
        return Coefficients(packets.rays.reshape(self.padded_n, -1), packets.valid.reshape(-1))

    def _affine_geometry(self, nominal: jax.Array, scales: jax.Array) -> AffineGeometry:
        q = jnp.zeros((self.padded_n, self.dofs), jnp.float32)
        cf = self.coefficients(nominal, q, self.valid, scales).rays
        tangent = jnp.zeros((self.padded_n, self.dofs, 2), jnp.float32)
        angle = jnp.deg2rad(jnp.float32(self.detector_roll))
        c, s = jnp.cos(angle), jnp.sin(angle)
        d = self.detector
        # A lab translation projected onto the fixed detector u/v basis moves
        # the coefficient offsets by minus that many detector samples.
        for column, coordinate in enumerate(self.parameters.local):
            physical = scales[:, coordinate]
            u = physical * (c if coordinate == 3 else s) / d.du
            v = physical * (-s if coordinate == 3 else c) / d.dv
            tangent = tangent.at[:, column, 0].set(-cf[:, 1] * u - cf[:, 2] * v)
            tangent = tangent.at[:, column, 1].set(-cf[:, 5] * u - cf[:, 6] * v)
        if self.parameters.nglobal:
            # Parallel center u is a positive lab-x detector displacement.
            u, v = c * self.native[0] / d.du, -s * self.native[0] / d.dv
            tangent = tangent.at[:, -1, 0].set(cf[:, 1] * u + cf[:, 2] * v)
            tangent = tangent.at[:, -1, 1].set(cf[:, 5] * u + cf[:, 6] * v)
        return AffineGeometry(cf, tangent)

    def initial_gradient(
        self,
        q: jax.Array | Coefficients,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
    ) -> jax.Array:
        """Huber initialization gradient, with fresh or frozen geometry."""
        return self._initial_gradient(self.prepare(q), x, y, precision, delta)

    def weighted_forward(
        self,
        q: jax.Array | Coefficients,
        x: jax.Array,
        sw: ArrayStack,
        j: ArrayStack,
        precision: ArrayStack | None,
        r: ArrayStack,
    ) -> tuple[ArrayStack, jax.Array]:
        """Weighted, profiled image projections written into ``r``, and their pose-column scores.

        ``sw`` are the IRLS square-root weights of the linearization and
        ``precision`` the data precision it used (None: uniform). ``r`` is donated
        and returned holding the projections.
        """
        r, rows = self.forward_scores(self.prepare(q), x, sw, j, precision, r)
        return r, self.parameters.reduce_score(rows[: self.n])

    def weighted_back(
        self,
        q: jax.Array | Coefficients,
        r: ArrayStack,
        sw: ArrayStack,
        j: ArrayStack,
        p: jax.Array,
        active: tuple[jax.Array, ...] | None = None,
        precision: ArrayStack | None = None,
    ) -> jax.Array:
        """Transpose of :meth:`weighted_forward` applied to residuals less the pose columns.

        ``active`` (``projection.active``) may skip voxels the caller discards.
        """
        return self.back_rows(self.prepare(q), r, sw, j, self.pad_poses(p), active, precision)

    def pose_back(
        self,
        q: jax.Array | Coefficients,
        sw: ArrayStack,
        j: ArrayStack,
        p: jax.Array,
        active: tuple[jax.Array, ...] | None = None,
        precision: ArrayStack | None = None,
    ) -> jax.Array:
        """Apply the image/pose cross block to pose increments (``active`` as weighted_back)."""
        return self._pose_back(self.prepare(q), sw, j, p, active, precision)

    def coefficients(
        self,
        nominal: jax.Array | AffineGeometry,
        q: jax.Array,
        valid: jax.Array,
        scales: jax.Array | None = None,
        frames: jax.Array | None = None,
    ) -> Coefficients:
        """One batch's physical coefficients and explicit padded-view mask."""
        if isinstance(nominal, AffineGeometry):
            offsets = jnp.einsum(
                "nk,nkj->nj",
                q,
                nominal.offsets,
                precision=jax.lax.Precision.HIGHEST,
            )
            rays = nominal.rays.at[:, 4].add(offsets[:, 0]).at[:, 8].add(offsets[:, 1])
            return Coefficients(rays, valid)
        scales = self.default_scales if scales is None else scales
        poses, frames = self.parameters.geometry(
            nominal,
            q,
            scales,
            detector=self.detector,
            native=self.native,
            frames=frames,
            detector_roll=self.detector_roll,
            detector_pivot=self.detector_pivot,
        )
        return self.projection.coefficients(poses, valid, frames)

    def _tangent(
        self,
        nominal: jax.Array | AffineGeometry,
        q: jax.Array,
        valid: jax.Array,
        scales: jax.Array | None = None,
        frames: jax.Array | None = None,
    ) -> jax.Array:
        del valid  # the coefficient packet masks predictions and columns together
        if isinstance(nominal, AffineGeometry):
            # Only the two offsets vary. Keep zeros explicit in the compiled
            # derivative of the ray slopes and integration weights.
            tangent = jnp.zeros((*nominal.offsets.shape[:2], 14), jnp.float32)
            return (
                tangent.at[:, :, 4]
                .set(nominal.offsets[:, :, 0])
                .at[:, :, 8]
                .set(nominal.offsets[:, :, 1])
            )
        scales = self.default_scales if scales is None else scales
        scales = jnp.broadcast_to(scales, (len(q), self.parameters.full_dofs))

        def one(n: jax.Array, p: jax.Array, scale: jax.Array, frame: jax.Array | None) -> jax.Array:
            return self.coefficients(
                n[None], p[None], jnp.ones(1, jnp.bool_), scale[None], frame
            ).rays[0]

        tangent = jax.vmap(
            jax.jacfwd(one, argnums=1), in_axes=(0, 0, 0, None if frames is None else 0)
        )(nominal, q, scales, frames)
        return jnp.swapaxes(tangent, 1, 2)

    def columns(
        self,
        nominal: jax.Array | AffineGeometry,
        q: jax.Array,
        valid: jax.Array,
        x: jax.Array,
        *,
        precision: jax.Array | None = None,
        scales: jax.Array | None = None,
        frames: jax.Array | None = None,
    ) -> jax.Array:
        """Projections and pose derivatives, (view, 1 + dofs, v, u)."""
        cf = self.coefficients(nominal, q, valid, scales, frames)
        tangent = self._tangent(nominal, q, valid, scales, frames)
        return self.projection.columns(cf, x, tangent, precision=precision)

    def _zeros(self) -> jax.Array:
        return jnp.zeros(self.shape, jnp.float32)

    def views(self, q: jax.Array) -> ViewBatch:
        """Batched geometry inputs, sharing the fixed affine map when angles are held."""
        return (
            self.batches(self.nominal if self.affine is None else self.affine),
            self.batches(q),
            self.batches(self.valid),
            self.batches(self.scales),
            None if self.frames is None else self.batches(self.frames),
        )

    def _sensitivity(
        self, q: jax.Array, _out: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """The Lipschitz bound and the voxels any ray sees, into ``_out`` (a bool volume)."""
        weights = jnp.ones((self.batch, self.detector.nv, self.detector.nu), jnp.float32)

        def step(a: jax.Array, z: ViewBatch) -> tuple[jax.Array, None]:
            cf = self.coefficients(*z)
            return self.projection.majorizer(cf, weights, a), None

        raw = jax.lax.scan(step, self._zeros(), self.views(q))[0]
        top = jnp.maximum(raw.max(), jnp.float32(1e-8))
        return top, raw > 0

    def _initial_gradient(
        self,
        q: Coefficients,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
    ) -> jax.Array:
        def step(a: jax.Array, z: tuple[jax.Array, Coefficients]) -> tuple[jax.Array, None]:
            i, cf = z
            yb, wb = y.read(i), self.batch_of(precision, i)
            wb = self.batch_precision(wb, cf.valid)
            # The start fits the objective the solver then uses: each view's
            # offset and row ramp profiled out. Huber, not redescending: the
            # initialization never discards data.
            e = profiled_residual(self.project(cf, x, precision=wb) - yb, wb)
            s0 = jnp.sqrt(wb.astype(jnp.float32))
            clipped = jnp.clip(e, -delta, delta)
            return self.back(a, cf, s0 * profile(s0 * clipped, s0), precision=wb), None

        batches = (jnp.arange(self.nb), self.batches(q))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _linearize(
        self,
        q: jax.Array,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
        out: LinearBuffers,
    ) -> tuple[
        jax.Array,
        jax.Array,
        ArrayStack,
        ArrayStack,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        ArrayStack,
        jax.Array,
    ]:
        """The linearization, with ``out`` = (gradient, diagonal, sw, j, residual) donated.

        The stacks are rewritten in place; the gradient and diagonal are written into the
        two volumes, unread.
        """

        def step(
            state: _LinearCarry, z: IndexedViewBatch
        ) -> tuple[_LinearCarry, _BatchLinearization]:
            gx, dd, (sw_out, j_out, residual_out) = state
            i, nominal, qb, valid, scales, frames = z
            yb, wb = y.read(i), self.batch_of(precision, i)
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            tangent = self._tangent(nominal, qb, valid, scales, frames)
            raw = self.projection.columns(cf, x, tangent, precision=wb)
            r = profiled_residual(raw[:, 0] - yb, wb)
            # The step stores the weights and columns in FP16; the score and
            # Gram matrix below use the same rounded values.
            res, sw, _ = robust_batch(r, raw[:, 1:], wb, delta, half=self._half)
            sw = jax.lax.optimization_barrier(sw)  # stored and used: one value (ArrayStack.read)
            # Only what a view's offset and ramp cannot explain informs the step.
            # The objective's projector (under the precision) applies to the
            # columns, then the robust factor: by the chain rule the score, Gram
            # matrix and image gradient are exactly the profiled objective's.
            s0, rho = self._split(sw, wb, valid)
            rounded = partial(jax.lax.reduce_precision, exponent_bits=5, mantissa_bits=10)
            j = rho[:, None] * profile(raw[:, 1:] * s0[:, None], s0)
            j = jax.lax.optimization_barrier(rounded(j) if self._half else j)
            gx = self.back(gx, cf, self._transpose(res, s0, rho), precision=wb)
            dd = self.projection.diagonal(cf, sw, dd)
            valid_px = jnp.sum(wb > 0, dtype=jnp.int64)
            down = jnp.sum((wb > 0) & (sw.astype(jnp.float64) ** 2 < 0.99 * wb), dtype=jnp.int64)
            # The score, Gram matrix and cross term must describe the SAME J.
            # Quantizing J after these reductions makes the Schur model inconsistent.
            score, gram = column_score(j, res), column_gram(j)
            # A stored value is also used here: one value behind a barrier, so its fusion with
            # the FP64 sums cannot depend on where the stack is held (ArrayStack.read; 1 ulp in the
            # moments measured with the residual on the host).
            residual = jax.lax.optimization_barrier(jnp.where(wb > 0, raw[:, 0] - yb, 0))
            stacks = (sw_out.write(i, sw), j_out.write(i, j), residual_out.write(i, residual))
            # The shared moments are two numbers: accumulate them batch by batch.
            out = (score, gram, robust_value(r, wb, delta), _common.moments(residual, wb))
            # The views' shared offset and ramp (the solver holds them) move with each view's pose.
            shared = _common.shared_weights(wb)
            moved = jax.vmap(lambda c, w: jax.vmap(lambda k: _common.moments(k[None], w[None]))(c))(
                raw[:, 1:], shared
            )
            return (gx, dd, stacks), (*out, valid_px, down, moved)

        batches = (jnp.arange(self.nb), *self.views(q))
        (gx, dd, (sw, j, residual)), summaries = jax.lax.scan(
            step, (self._zeros(), self._zeros(), out[2:]), batches
        )
        g, gram, e, moments, valid, down, moved = summaries
        n, k = self.n, self.dofs
        moved = moved.reshape(self.padded_n, k, 2)[:n]
        return (
            gx,
            dd,
            sw,
            j,
            self.parameters.reduce_score(g.reshape(self.padded_n, k)[:n]),
            gram.reshape(self.padded_n, k, k)[:n],
            jnp.sum(e),
            jnp.stack([jnp.sum(valid), jnp.sum(down)]),
            jnp.sum(moments, axis=0),
            residual,
            jnp.stack([self.parameters.reduce_score(moved[..., i]) for i in range(2)]),
        )

    def _score_inputs(
        self,
        sw: ArrayStack,
        j: ArrayStack,
        residual: ArrayStack,
        precision: ArrayStack | None,
        i: int | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Gather and whiten one batch of translation-score operands where the stacks live."""
        w = self.batch_precision(self.batch_of(precision, i), self.batches(self.valid)[i])
        res = sw.read(i).astype(jnp.float32) * profiled_residual(residual.read(i), w)
        return j.read(i)[:, self.parameters.translation_columns].astype(jnp.float32), res

    def translation_scores(
        self, sw: ArrayStack, j: ArrayStack, residual: ArrayStack, precision: ArrayStack | None
    ) -> jax.Array:
        """Per-view scores for the motion scale, including the unpadded final batch.

        Keep the FP32 dot as a separate reduction. Fusing it with the whitening
        lets the GPU compiler change its accumulation order, enough to change
        the fitted motion on a poorly conditioned scan.
        """
        scores = []
        for i in range(self.nb):
            columns, res = self._score_inputs(sw, j, residual, precision, i)
            count = min(self.n - i * self.batch, self.batch)
            scores.append(jnp.einsum("vkij,vij->vk", columns[:count], res[:count]))
        return jnp.concatenate(scores)

    def _shared_data(
        self, y: ArrayStack, precision: ArrayStack | None
    ) -> tuple[jax.Array, jax.Array]:
        """G and the data's shared moments over the profiled views, accumulated batch by batch."""

        def step(
            total: tuple[jax.Array, jax.Array], z: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            i, valid = z
            yb, wb = y.read(i), self.batch_of(precision, i)
            wb = self.batch_precision(wb, valid)
            gram = _common.metric(wb)
            return (total[0] + gram, total[1] + _common.moments(jnp.where(wb > 0, yb, 0), wb)), None

        start = (jnp.zeros((2, 2), jnp.float64), jnp.zeros(2, jnp.float64))
        batches = (jnp.arange(self.nb), self.batches(self.valid))
        return jax.lax.scan(step, start, batches)[0]

    def _objective(
        self,
        q: jax.Array,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """The profiled robust data term, and the residual's shared moments b (``_common``)."""

        def step(
            total: tuple[jax.Array, jax.Array], z: IndexedViewBatch
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            i, nominal, qb, valid, scales, frames = z
            yb, wb = y.read(i), self.batch_of(precision, i)
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            raw = jnp.where(wb > 0, self.project(cf, x, precision=wb) - yb, 0)
            value = robust_value(profiled_residual(raw, wb), wb, delta)
            return (total[0] + value, total[1] + _common.moments(raw, wb)), None

        batches = (jnp.arange(self.nb), *self.views(q))
        start = (jnp.asarray(0.0, jnp.float64), jnp.zeros(2, jnp.float64))
        return jax.lax.scan(step, start, batches)[0]

    def _common_images(
        self, q: jax.Array, precision: ArrayStack | None, _out: jax.Array | None = None
    ) -> jax.Array:
        """B = [A^T w, A^T (w v)] summed over views: the shared image rows, into ``_out``."""

        def step(
            a: tuple[jax.Array, jax.Array], z: IndexedViewBatch
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            i, nominal, qb, valid, scales, frames = z
            wb = self.batch_precision(self.batch_of(precision, i), valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            shared = _common.shared_weights(wb)
            ramp = (shared * _common.rows(wb.shape[-2])).astype(jnp.float32)
            return (
                self.back(a[0], cf, shared.astype(jnp.float32), precision=wb),
                self.back(a[1], cf, ramp, precision=wb),
            ), None

        batches = (jnp.arange(self.nb), *self.views(q))
        return jnp.stack(jax.lax.scan(step, (self._zeros(), self._zeros()), batches)[0])

    def _view_summary(
        self, q: jax.Array, x: jax.Array, y: ArrayStack, precision: ArrayStack | None
    ) -> tuple[jax.Array, jax.Array]:
        """Each view's offset and row ramp, and its mean squared profiled residual.

        The offset and ramp are the fit of the observations less the prediction,
        less the views' shared offset and ramp: the solver holds the shared level
        (``_common``), so only each view's deviation from it is profiled out. The
        residual is what remains of each view, weighted by the precision and
        averaged over the view's measured pixels.
        """

        def step(z: IndexedViewBatch) -> jax.Array:
            i, nominal, qb, valid, scales, frames = z
            yb, wb = y.read(i), self.batch_of(precision, i)
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            root = jnp.sqrt(wb.astype(jnp.float64))
            c0, c1, ss, sv = _offset_and_ramp(root * (yb - self.project(cf, x, precision=wb)), root)
            whitened = root * (yb - self.project(cf, x, precision=wb)) - ss * c0[:, None, None]
            whitened = whitened - sv * c1[:, None, None]
            mass = jnp.maximum(jnp.sum(wb, axis=(-2, -1), dtype=jnp.float64), 1e-300)
            energy = jnp.sum(whitened**2, axis=(-2, -1)) / mass
            gram = jax.vmap(lambda w: _common.metric(w[None]))(wb).reshape(-1, 4)
            return jnp.concatenate([jnp.stack([c0, c1], -1), energy[:, None], gram], -1)

        batches = (jnp.arange(self.nb), *self.views(q))
        out = jax.lax.map(step, batches).reshape(self.padded_n, 7)[: self.n]
        gram = out[:, 3:].reshape(-1, 2, 2)
        offsets = out[:, :2]
        shared = jnp.linalg.solve(jnp.sum(gram, 0), jnp.einsum("nij,nj->i", gram, offsets))
        return offsets - shared, out[:, 2]

    def _split(
        self, sw: jax.Array, precision: jax.Array | None, valid: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Precision square roots s0 and robust factors rho = sw / s0 of one batch.

        The objective profiles each view under the precision, then applies the
        robust loss: the step's columns are rho * profile(s0 * column, s0) and
        their transpose s0 * profile(rho * image, s0), one projector throughout.
        """
        s0 = jnp.sqrt(self.batch_precision(precision, valid).astype(jnp.float32))
        return s0, jnp.where(s0 > 0, sw / jnp.where(s0 > 0, s0, 1), 0).astype(jnp.float32)

    def _transpose(self, image: jax.Array, s0: jax.Array, rho: jax.Array) -> jax.Array:
        return s0 * profile(rho * image, s0)

    def forward_scores(
        self,
        q: Coefficients,
        x: jax.Array,
        sw: ArrayStack,
        j: ArrayStack,
        precision: ArrayStack | None,
        r: ArrayStack,
        first: jax.Array | None = None,
    ) -> tuple[ArrayStack, jax.Array]:
        """Project one image through all batches, storing its weighted rays and pose scores.

        The CUDA plane kernels read adjacent detector rows along z. One conversion to
        z-contiguous storage serves all the batches in this product.
        """
        x = transpose_volume(x) if self.z_fast else x

        def step(r: ArrayStack, z: tuple[jax.Array, Coefficients]) -> tuple[ArrayStack, jax.Array]:
            i, cf = z
            s, jb, wb = sw.read(i, first), j.read(i, first), self.batch_of(precision, i, first)
            s0, rho = self._split(s, wb, cf.valid)
            image = self.project(cf, x, precision=wb, z_fast=self.z_fast)
            out = jax.lax.optimization_barrier(rho * profile(image * s0, s0))
            return r.write(i, out), column_score(jb, out)

        batches = self.batches(q)
        r, p = jax.lax.scan(step, r, (jnp.arange(len(batches.valid)), batches))
        return r, p.reshape(-1, self.dofs)

    def back_rows(
        self,
        q: Coefficients,
        r: ArrayStack,
        sw: ArrayStack,
        j: ArrayStack,
        rows: jax.Array,
        active: tuple[jax.Array, ...] | None,
        precision: ArrayStack | None,
        accumulator: jax.Array | None = None,
        first: jax.Array | None = None,
    ) -> jax.Array:
        """Apply the weighted image transpose to a share of views and their expanded pose rows."""

        def step(
            a: jax.Array, z: tuple[jax.Array, Coefficients, jax.Array]
        ) -> tuple[jax.Array, None]:
            i, cf, pb = z
            rb, s, jb, wb = (
                r.read(i),
                sw.read(i, first),
                j.read(i, first),
                self.batch_of(precision, i, first),
            )
            s0, rho = self._split(s, wb, cf.valid)
            image = self._transpose(rb - _apply_columns(jb, pb), s0, rho)
            return self.projection.back(a, cf, image, precision=wb, active=active), None

        packet = self.batches(q)
        batches = (jnp.arange(len(packet.valid)), packet, self.batches(rows))
        return jax.lax.scan(step, self._zeros() if accumulator is None else accumulator, batches)[0]

    def _pose_back(
        self,
        q: Coefficients,
        sw: ArrayStack,
        j: ArrayStack,
        p: jax.Array,
        active: tuple[jax.Array, ...] | None,
        precision: ArrayStack | None,
    ) -> jax.Array:
        def step(
            a: jax.Array, z: tuple[jax.Array, Coefficients, jax.Array]
        ) -> tuple[jax.Array, None]:
            i, cf, pb = z
            s, jb, wb = sw.read(i), j.read(i), self.batch_of(precision, i)
            s0, rho = self._split(s, wb, cf.valid)
            image = self._transpose(_apply_columns(jb, pb), s0, rho)
            return self.projection.back(a, cf, image, precision=wb, active=active), None

        batches = (jnp.arange(self.nb), self.batches(q), self.batches(self.pad_poses(p)))
        return jax.lax.scan(step, self._zeros(), batches)[0]
