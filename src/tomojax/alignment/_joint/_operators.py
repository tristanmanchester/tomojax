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

from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._parameters import Parameters
from ._projection import (
    Coefficients,
    Projection,
    cone_parameter_scales,
)


class AffineGeometry(NamedTuple):
    """Fixed parallel-ray coefficients and exact centre/translation tangents."""

    rays: jax.Array
    tangent: jax.Array


def robust_value(residual, precision, delta):
    """Masked quadratic-core, log-tail loss; FP64 sum."""
    r = jnp.where(precision > 0, residual, 0).astype(jnp.float64)
    d = jnp.asarray(delta, jnp.float64)
    a = jnp.abs(r)
    loss = jnp.where(a <= d, 0.5 * r * r, d * d * (0.5 + jnp.log(jnp.maximum(a, d) / d)))
    return jnp.sum(precision.astype(jnp.float64) * loss, dtype=jnp.float64)


def robust_batch(residual, columns, precision, delta, *, half=False):
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


def _offset_and_ramp(images, s):
    """Per view (and leading column): the s-whitened fit coefficients of [s, s * v] to images."""
    v = jnp.linspace(-1, 1, s.shape[-2], dtype=jnp.float64)[:, None]
    s = s.astype(jnp.float64)
    extra = (slice(None),) + (None,) * (images.ndim - s.ndim)
    w = s * s
    g00, g01, g11 = (jnp.sum(w * v**k, axis=(-2, -1)) for k in (0, 1, 2))
    det = g00 * g11 - g01 * g01
    det = jnp.where(det > 1e-12 * jnp.maximum(g00 * g11, 1e-300), det, jnp.inf)
    ss, sv = s[extra], (s * v)[extra]
    b0 = jnp.sum(ss * images, axis=(-2, -1))
    b1 = jnp.sum(sv * images, axis=(-2, -1))
    c0 = (g11[extra] * b0 - g01[extra] * b1) / det[extra]
    c1 = (g00[extra] * b1 - g01[extra] * b0) / det[extra]
    return c0, c1, ss, sv


def profile(images, s):
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


def profiled_residual(residual, precision):
    """Residuals less each view's precision-weighted constant and row ramp."""
    root = jnp.sqrt(precision.astype(jnp.float64))
    whitened = profile(root * residual, root)
    return jnp.where(precision > 0, whitened / jnp.where(precision > 0, root, 1), 0).astype(
        residual.dtype
    )


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
        self.view_summary = jax.jit(self.view_summary)
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

    def weighted_forward(self, q, x, sw, j, precision=None):
        """Weighted, profiled image projections and their pose-column contractions.

        ``sw`` are the IRLS square-root weights of the linearization and
        ``precision`` the data precision it used (None: uniform).
        """
        return self._weighted_forward(self.prepare(q), x, sw, j, precision)

    def weighted_back(self, q, r, sw, j, p, active=None, precision=None):
        """Transpose of :meth:`weighted_forward` applied to residuals less the pose columns.

        ``active`` (``projection.active``) may skip voxels the caller discards.
        """
        return self._weighted_back(self.prepare(q), r, sw, j, p, active, precision)

    def pose_back(self, q, sw, j, p, active=None, precision=None):
        """Apply the image/pose cross block to pose increments (``active`` as weighted_back)."""
        return self._pose_back(self.prepare(q), sw, j, p, active, precision)

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
            # The start fits the objective the solver then uses: each view's
            # offset and row ramp profiled out. Huber, not redescending: the
            # initialization never discards data.
            e = profiled_residual(self.project(cf, x) - yb, wb)
            s0 = jnp.sqrt(wb.astype(jnp.float32))
            clipped = jnp.clip(e, -delta, delta)
            return self.back(a, cf, s0 * profile(s0 * clipped, s0)), None

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
            r = profiled_residual(raw[:, 0] - yb, wb)
            # The step stores the weights and columns in FP16; the score and
            # Gram matrix below use the same rounded values.
            res, sw, _ = robust_batch(r, raw[:, 1:], wb, delta, half=True)
            # Only what a view's offset and ramp cannot explain informs the step.
            # The objective's projector (under the precision) applies to the
            # columns, then the robust factor: by the chain rule the score, Gram
            # matrix and image gradient are exactly the profiled objective's.
            s0, rho = self._split(sw, wb, valid)
            rounded = partial(jax.lax.reduce_precision, exponent_bits=5, mantissa_bits=10)
            j = rounded(rho[:, None] * profile(raw[:, 1:] * s0[:, None], s0))
            gx = self.back(gx, cf, self._transpose(res, s0, rho))
            dd = self.projection.diagonal(cf, sw, dd)
            valid_px = jnp.sum(wb > 0, dtype=jnp.int64)
            down = jnp.sum((wb > 0) & (sw.astype(jnp.float64) ** 2 < 0.99 * wb), dtype=jnp.int64)
            # The score, Gram matrix and cross term must describe the SAME J.
            # Quantizing J after these reductions makes the Schur model inconsistent.
            score, gram = column_score(j, res), column_gram(j)
            out = (sw.astype(jnp.float16), j.astype(jnp.float16), score, gram)
            out = (*out, robust_value(r, wb, delta), jnp.where(wb > 0, raw[:, 0] - yb, 0))
            return (gx, dd), (*out, valid_px, down)

        batches = (*self.views(q), self.batches(y), self.batches(precision))
        state, out = jax.lax.scan(step, (self._zeros(), self._zeros()), batches)
        sw, j, g, gram, e, residual, valid, down = out
        n, d, k = self.n, self.detector, self.dofs
        return (
            *state,
            sw.reshape(self.image_shape),
            j.reshape((self.padded_n, k, d.nv, d.nu)),
            self.parameters.reduce_score(g.reshape(self.padded_n, k)[:n]),
            gram.reshape(self.padded_n, k, k)[:n],
            jnp.sum(e),
            jnp.stack([jnp.sum(valid), jnp.sum(down)]),
            residual.reshape(self.image_shape),
        )

    def _objective(self, q, x, y, precision, delta):
        def step(total, z):
            nominal, qb, valid, scales, frames, yb, wb = z
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            residual = profiled_residual(self.project(cf, x) - yb, wb)
            return total + robust_value(residual, wb, delta), None

        batches = (*self.views(q), self.batches(y), self.batches(precision))
        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), batches)[0]

    def view_summary(self, q, x, y, precision):
        """Each view's offset and row ramp, and its mean squared profiled residual.

        The offset and ramp are the fit of the observations less the prediction;
        the residual is what remains, weighted by the precision and averaged over
        the view's measured pixels.
        """

        def step(z):
            nominal, qb, valid, scales, frames, yb, wb = z
            wb = self.batch_precision(wb, valid)
            cf = self.coefficients(nominal, qb, valid, scales, frames)
            root = jnp.sqrt(wb.astype(jnp.float64))
            c0, c1, ss, sv = _offset_and_ramp(root * (yb - self.project(cf, x)), root)
            whitened = root * (yb - self.project(cf, x)) - ss * c0[:, None, None]
            whitened = whitened - sv * c1[:, None, None]
            mass = jnp.maximum(jnp.sum(wb, axis=(-2, -1), dtype=jnp.float64), 1e-300)
            energy = jnp.sum(whitened**2, axis=(-2, -1)) / mass
            return jnp.concatenate([jnp.stack([c0, c1], -1), energy[:, None]], -1)

        batches = (*self.views(q), self.batches(y), self.batches(precision))
        out = jax.lax.map(step, batches).reshape(self.padded_n, 3)[: self.n]
        return out[:, :2], out[:, 2]

    def _split(self, sw, precision, valid):
        """Precision square roots s0 and robust factors rho = sw / s0 of one batch.

        The objective profiles each view under the precision, then applies the
        robust loss: the step's columns are rho * profile(s0 * column, s0) and
        their transpose s0 * profile(rho * image, s0), one projector throughout.
        """
        s0 = jnp.sqrt(self.batch_precision(precision, valid).astype(jnp.float32))
        return s0, jnp.where(s0 > 0, sw / jnp.where(s0 > 0, s0, 1), 0).astype(jnp.float32)

    def _transpose(self, image, s0, rho):
        return s0 * profile(rho * image, s0)

    def _weighted_forward(self, q, x, sw, j, precision):
        def step(z):
            cf, s, jb, wb = z
            s0, rho = self._split(s, wb, cf.valid)
            out = rho * profile(self.project(cf, x) * s0, s0)
            return out, column_score(jb, out)

        batches = (self.batches(q), *map(self.batches, (sw, j, precision)))
        out, p = jax.lax.map(step, batches)
        return out.reshape(self.image_shape), self.parameters.reduce_score(
            p.reshape(self.padded_n, self.dofs)[: self.n]
        )

    def _weighted_back(self, q, r, sw, j, p, active, precision):
        def step(a, z):
            cf, rb, s, jb, pb, wb = z
            s0, rho = self._split(s, wb, cf.valid)
            image = self._transpose(rb - _apply_columns(jb, pb), s0, rho)
            return self.projection.back(a, cf, image, active=active), None

        batches = (self.batches(q), *map(self.batches, (r, sw, j, self.pad_poses(p), precision)))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _pose_back(self, q, sw, j, p, active, precision):
        def step(a, z):
            cf, s, jb, pb, wb = z
            s0, rho = self._split(s, wb, cf.valid)
            image = self._transpose(_apply_columns(jb, pb), s0, rho)
            return self.projection.back(a, cf, image, active=active), None

        batches = (self.batches(q), *map(self.batches, (sw, j, self.pad_poses(p), precision)))
        return jax.lax.scan(step, self._zeros(), batches)[0]
