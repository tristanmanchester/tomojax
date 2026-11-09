"""Batched image and active-geometry operators for the joint fit.

Only selected local parameters and shared calibration parameters have columns.
The default has two translations per view; cone translations also include
beam depth. Local rotations, when selected, compose in the sample frame and
use geometry-scaled pixel units. Global centre, tilt and roll columns are
shared by the whole scan through the parameter layout's broadcast and summed
score. Coefficients and exact active-coordinate derivatives are computed from
the physical geometry on the device.

Views are processed in fixed batches padded with zero-weight views. Only
per-batch residuals and raw pose columns are temporary; frozen IRLS weights and
the whitened FP32 pose columns stay resident, so one weighted forward pass is
reused within each Schur matvec. Scores, Gram blocks and matvecs use those same
columns. The data loss is a quadratic core with a log tail.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._parameters import Parameters
from ._projection import (
    Coefficients,
    Projection,
    cone_parameter_scales,
    posed as _posed,
)

DOFS = 5


class AffineGeometry(NamedTuple):
    """Fixed parallel-ray coefficients and exact centre/translation tangents."""

    rays: jax.Array
    tangent: jax.Array


def posed(nominal, q, lever, native):
    """World-from-object poses (views, 4, 4) after the pose parameters ``q``."""
    return _posed(nominal, q, jnp.asarray([1 / lever] * 3 + list(native), q.dtype))


def robust_value(residual, precision, delta):
    """Masked quadratic-core, log-tail loss; FP64 sum."""
    r = jnp.where(precision > 0, residual, 0).astype(jnp.float64)
    d = jnp.asarray(delta, jnp.float64)
    a = jnp.abs(r)
    loss = jnp.where(a <= d, 0.5 * r * r, d * d * (0.5 + jnp.log(jnp.maximum(a, d) / d)))
    return jnp.sum(precision.astype(jnp.float64) * loss, dtype=jnp.float64)


def robust_batch(residual, columns, precision, delta):
    """IRLS square-root weights, and weighted residuals and columns (view, k, v, u)."""
    r = jnp.where(precision > 0, residual, 0)
    a = jnp.abs(r.astype(jnp.float64))
    d = jnp.asarray(delta, jnp.float64)
    sw = jnp.sqrt(precision.astype(jnp.float64) * jnp.minimum(1.0, d / jnp.maximum(a, d)) ** 2)
    sw = sw.astype(jnp.float32)
    return (sw * r).astype(jnp.float32), sw, (columns * sw[:, None]).astype(jnp.float32)


def photometric_sums(prediction, observed, precision):
    """Per-view normal equations of the observations on [prediction, 1, v], FP64.

    Returns (views, 3, 4): the weighted Gram matrix of the three basis images
    beside their products with the observations; v runs from -1 to 1 down the rows.
    """
    w, p, y = (a.astype(jnp.float64) for a in (precision, prediction, observed))
    v = jnp.broadcast_to(jnp.linspace(-1, 1, p.shape[-2])[:, None], p.shape[-2:])
    basis = jnp.stack([p, jnp.ones_like(p), jnp.broadcast_to(v, p.shape)], 1)
    gram = jnp.einsum("nkvu,nlvu,nvu->nkl", basis, basis, w)
    return jnp.concatenate([gram, jnp.einsum("nkvu,nvu,nvu->nk", basis, y, w)[..., None]], -1)


def column_score(columns, images):
    """Per-view J^T r, FP64."""
    return jnp.sum(columns.astype(jnp.float64) * images[:, None].astype(jnp.float64), axis=(-2, -1))


def column_gram(columns):
    """Per-view J^T J, FP64."""
    flat = columns.reshape((*columns.shape[:2], -1)).astype(jnp.float64)
    return jnp.einsum("nik,njk->nij", flat, flat)


def _apply_columns(columns, p):
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

    def __init__(
        self,
        grid,
        detector,
        nominal,
        *,
        lever,
        native,
        batch_size=8,
        frames=None,
        scales=None,
        interpolation="linear",
        parameters=None,
        detector_roll_deg=0.0,
        detector_pivot=None,
    ) -> None:
        self.grid, self.detector = grid, detector
        self.shape = (grid.nz, grid.ny, grid.nx)
        self.n = len(nominal)
        self.parameters = parameters or Parameters(
            self.n, ("translations", "rotations"), cone=frames is not None, lever=lever
        )
        self.detector_roll_deg = float(detector_roll_deg)
        self.detector_pivot = detector.center if detector_pivot is None else tuple(detector_pivot)
        self.projection = Projection(
            grid,
            detector,
            cone=frames is not None,
            interpolation=interpolation,
        )
        self.backend, self.dofs = self.projection.backend, self.parameters.k
        # Bound the temporary value/pose columns to about 32 MiB when more
        # than one view fits.
        column_bytes = detector.nv * detector.nu * (1 + self.dofs) * 4
        self.batch = min(batch_size, self.n, max(1, 32 * 2**20 // column_bytes))
        self.nb = math.ceil(self.n / self.batch)
        self.padded_n = self.nb * self.batch
        self.image_shape = (self.padded_n, detector.nv, detector.nu)
        self.lever, self.native = float(lever), tuple(map(float, native))
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
            self.affine = jax.jit(self._affine_geometry)(self.nominal, self.scales)
        self._geometry = jax.jit(self._geometry)
        self.sensitivity = jax.jit(self._sensitivity)
        self._initial_gradient = jax.jit(self._initial_gradient)
        self.linearize = jax.jit(self._linearize)
        self.objective = jax.jit(self._objective)
        self._weighted_forward = jax.jit(self._weighted_forward)
        self._weighted_back = jax.jit(self._weighted_back)
        self._pose_back = jax.jit(self._pose_back)

    def pad_poses(self, q):
        """Expand a compact state, then pad its per-view rows with zero-weight views."""
        q = self.parameters.expand(q).astype(jnp.float32)
        return jnp.pad(q, ((0, self.padded_n - self.n), (0, 0)))

    def pad_images(self, y):
        """Pad with zero images, which carry zero precision."""
        y = jnp.asarray(y, jnp.float32)
        return jnp.pad(y, ((0, self.padded_n - self.n), (0, 0), (0, 0)))

    def pad_precision(self, precision):
        """Keep uniform precision implicit; pad explicitly supplied weights."""
        return None if precision is None else self.pad_images(precision)

    def batch_precision(self, precision, valid):
        """One batch's weights, with zero precision for padded views."""
        if precision is not None:
            return precision
        return jnp.broadcast_to(
            valid[:, None, None].astype(jnp.float32),
            (len(valid), self.detector.nv, self.detector.nu),
        )

    def batches(self, a):
        """Reshape a padded per-view array tree into (batches, batch, ...)."""
        return jax.tree.map(lambda x: x.reshape((self.nb, self.batch, *x.shape[1:])), a)

    def prepare(self, q):
        """Freeze geometry for the image-only loops."""
        if isinstance(q, Coefficients):
            return q
        return self._geometry(q)

    def _geometry(self, q):
        """Freeze the small coefficient packet for one linearized image solve."""
        packets = jax.lax.map(lambda z: self.coefficients(*z), self.views(q))
        return jax.tree.map(lambda a: a.reshape((self.padded_n, *a.shape[2:])), packets)

    def _affine_geometry(self, nominal, scales):
        q = jnp.zeros((self.padded_n, self.dofs), jnp.float32)
        cf = self.coefficients(nominal, q, self.valid, scales).rays
        tangent = jnp.zeros((self.padded_n, self.dofs, 14), jnp.float32)
        angle = jnp.deg2rad(jnp.float32(self.detector_roll_deg))
        c, s = jnp.cos(angle), jnp.sin(angle)
        d = self.detector
        # A lab translation projected onto the fixed detector u/v basis moves
        # the coefficient offsets by minus that many detector samples.
        for column, coordinate in enumerate(self.parameters.local):
            physical = scales[:, coordinate]
            u = physical * (c if coordinate == 3 else s) / d.du
            v = physical * (-s if coordinate == 3 else c) / d.dv
            tangent = tangent.at[:, column, 4].set(-cf[:, 1] * u - cf[:, 2] * v)
            tangent = tangent.at[:, column, 8].set(-cf[:, 5] * u - cf[:, 6] * v)
        if self.parameters.nglobal:
            # Parallel centre u is a positive lab-x detector displacement.
            u, v = c * self.native[0] / d.du, -s * self.native[0] / d.dv
            tangent = tangent.at[:, -1, 4].set(cf[:, 1] * u + cf[:, 2] * v)
            tangent = tangent.at[:, -1, 8].set(cf[:, 5] * u + cf[:, 6] * v)
        return AffineGeometry(cf, tangent)

    def initial_gradient(self, q, x, y, precision, delta):
        """Huber initialization gradient, with fresh or frozen geometry."""
        return self._initial_gradient(self.prepare(q), x, y, precision, delta)

    def weighted_forward(self, q, x, sw, j):
        """Weighted image projections and their pose-column contractions."""
        return self._weighted_forward(self.prepare(q), x, sw, j)

    def weighted_back(self, q, r, sw, j, p):
        """Backproject weighted residuals minus the pose-column correction."""
        return self._weighted_back(self.prepare(q), r, sw, j, p)

    def pose_back(self, q, sw, j, p):
        """Apply the image/pose cross block to pose increments."""
        return self._pose_back(self.prepare(q), sw, j, p)

    def coefficients(self, nominal, q, valid, scales=None, frames=None):
        """One batch's physical coefficients and explicit padded-view mask."""
        if isinstance(nominal, AffineGeometry):
            offsets = jnp.einsum(
                "nk,nkj->nj",
                q,
                nominal.tangent[:, :, jnp.asarray([4, 8])],
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
            detector_roll_deg=self.detector_roll_deg,
            detector_pivot=self.detector_pivot,
        )
        return self.projection.coefficients(poses, valid, frames)

    def _tangent(self, nominal, q, valid, scales=None, frames=None):
        del valid  # the coefficient packet masks predictions and columns together
        if isinstance(nominal, AffineGeometry):
            # Preserve the affine map's structural zeros inside the compiled
            # derivative. Treating the cached full array as dense dynamic data
            # needlessly differentiates ray slopes and integration weights.
            tangent = jnp.zeros_like(nominal.tangent)
            return (
                tangent.at[:, :, 4]
                .set(nominal.tangent[:, :, 4])
                .at[:, :, 8]
                .set(nominal.tangent[:, :, 8])
            )
        scales = self.default_scales if scales is None else scales
        scales = jnp.broadcast_to(scales, (len(q), self.parameters.full_dofs))

        def one(n, p, scale, frame):
            return self.coefficients(
                n[None], p[None], jnp.ones(1, jnp.bool_), scale[None], frame
            ).rays[0]

        tangent = jax.vmap(
            jax.jacfwd(one, argnums=1), in_axes=(0, 0, 0, None if frames is None else 0)
        )(nominal, q, scales, frames)
        return jnp.swapaxes(tangent, 1, 2)

    def project(self, cf, x):
        """Projections of one coefficient batch."""
        return self.projection.project(cf, x)

    def columns(self, nominal, q, valid, x, *, scales=None, frames=None):
        """Projections and pose derivatives, (view, 1 + dofs, v, u)."""
        cf = self.coefficients(nominal, q, valid, scales, frames)
        tangent = self._tangent(nominal, q, valid, scales, frames)
        return self.projection.columns(cf, x, tangent)

    def back(self, accumulate, cf, y):
        """Add the matched transpose of one batch to ``accumulate``."""
        return self.projection.back(accumulate, cf, y)

    def _zeros(self):
        return jnp.zeros(self.shape, jnp.float32)

    def views(self, q):
        """Batched geometry inputs, sharing the fixed affine map when angles are held."""
        return (
            self.batches(self.nominal if self.affine is None else self.affine),
            self.batches(q),
            self.batches(self.valid),
            self.batches(self.scales),
            None if self.frames is None else self.batches(self.frames),
        )

    def _sensitivity(self, q):
        weights = jnp.ones((self.batch, self.detector.nv, self.detector.nu), jnp.float32)

        def step(a, z):
            cf = self.coefficients(*z)
            return self.projection.majorizer(cf, weights, a), None

        raw = jax.lax.scan(step, self._zeros(), self.views(q))[0]
        top = jnp.maximum(raw.max(), jnp.float32(1e-8))
        return top, raw > 0

    def _initial_gradient(self, q, x, y, precision, delta):
        def step(a, z):
            cf, yb, wb = z
            wb = self.batch_precision(wb, cf.valid)
            r = self.project(cf, x) - yb
            # Huber, not redescending: the initialization never discards data.
            q64 = wb.astype(jnp.float64) * jnp.clip(r.astype(jnp.float64), -delta, delta)
            return self.back(a, cf, q64.astype(jnp.float32)), None

        batches = (self.batches(q), self.batches(y), self.batches(precision))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _linearize(self, q, x, y, precision, delta):
        def step(state, z):
            gx, dd = state
            nominal, qb, valid, scales, frames, yb, wb = z
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            tangent = self._tangent(nominal, qb, valid, scales, frames)
            raw = self.projection.columns(cf, x, tangent)
            r = raw[:, 0] - yb
            res, sw, j = robust_batch(r, raw[:, 1:], wb, delta)
            gx = self.back(gx, cf, sw * res)
            dd = self.projection.diagonal(cf, sw, dd)
            valid_px = jnp.sum(wb > 0, dtype=jnp.int64)
            down = jnp.sum((wb > 0) & (sw.astype(jnp.float64) ** 2 < 0.99 * wb), dtype=jnp.int64)
            # The score, Gram matrix and cross term must describe the SAME J.
            # Quantizing J after these reductions makes the Schur model inconsistent.
            score, gram = column_score(j, res), column_gram(j)
            out = (sw, j, score, gram, robust_value(r, wb, delta))
            return (gx, dd), (*out, valid_px, down, photometric_sums(raw[:, 0], yb, wb))

        batches = (*self.views(q), self.batches(y), self.batches(precision))
        state, out = jax.lax.scan(step, (self._zeros(), self._zeros()), batches)
        sw, j, g, gram, e, valid, down, sums = out
        n, d, k = self.n, self.detector, self.dofs
        return (
            *state,
            sw.reshape(self.image_shape),
            j.reshape((self.padded_n, k, d.nv, d.nu)),
            self.parameters.reduce_score(g.reshape(self.padded_n, k)[:n]),
            gram.reshape(self.padded_n, k, k)[:n],
            jnp.sum(e),
            jnp.stack([jnp.sum(valid), jnp.sum(down)]),
            sums.reshape(self.padded_n, 3, 4)[:n],
        )

    def _objective(self, q, x, y, precision, delta):
        def step(total, z):
            nominal, qb, valid, scales, frames, yb, wb = z
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            return total + robust_value(self.project(cf, x) - yb, wb, delta), None

        batches = (*self.views(q), self.batches(y), self.batches(precision))
        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), batches)[0]

    def _weighted_forward(self, q, x, sw, j):
        def step(z):
            cf, s, jb = z
            out = self.project(cf, x) * s
            return out, column_score(jb, out)

        out, p = jax.lax.map(step, (self.batches(q), self.batches(sw), self.batches(j)))
        return out.reshape(self.image_shape), self.parameters.reduce_score(
            p.reshape(self.padded_n, self.dofs)[: self.n]
        )

    def _weighted_back(self, q, r, sw, j, p):
        def step(a, z):
            cf, rb, s, jb, pb = z
            return self.back(a, cf, (rb - _apply_columns(jb, pb)) * s), None

        batches = (self.batches(q), *map(self.batches, (r, sw, j, self.pad_poses(p))))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _pose_back(self, q, sw, j, p):
        def step(a, z):
            cf, s, jb, pb = z
            return self.back(a, cf, _apply_columns(jb, pb) * s), None

        batches = (self.batches(q), *map(self.batches, (sw, j, self.pad_poses(p))))
        return jax.lax.scan(step, self._zeros(), batches)[0]
