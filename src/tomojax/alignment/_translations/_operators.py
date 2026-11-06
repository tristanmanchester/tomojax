"""Batched device operators for the joint volume and detector-translation fit.

Views are processed in fixed batches padded with zero-weight copies, so a
padded view adds nothing. Only per-batch residuals and raw pose columns are
temporary; frozen IRLS weights and the two whitened translation columns per
view stay resident, so one weighted forward pass is reused within each Schur
matvec. The data loss is a quadratic core with a logarithmic tail.
"""

import math

import jax
import jax.numpy as jnp

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.joseph import (
    forward_project_planes,
    plane_pose_columns,
    plane_weighted_diagonal,
    sum_backproject_planes,
)
from tomojax.recon.fista_tv import probe_gradient_rms


def translated(cf, p, ratios):
    """Coefficients shifted by per-view detector translations ``p`` (native pixels)."""
    p = p.reshape((-1, 2)).astype(jnp.float32)
    p = jnp.pad(p, ((0, cf.shape[0] - p.shape[0]), (0, 0)))
    u, v = ratios[0] * p[:, 0], ratios[1] * p[:, 1]
    out = cf.at[:, 4].add(-cf[:, 1] * u - cf[:, 2] * v)
    return out.at[:, 8].add(-cf[:, 5] * u - cf[:, 6] * v)


def robust_value(residual, precision, delta):
    """Masked quadratic-core, log-tail loss; FP64 sum."""
    r = jnp.where(precision > 0, residual, 0).astype(jnp.float64)
    d = jnp.asarray(delta, jnp.float64)
    a = jnp.abs(r)
    loss = jnp.where(a <= d, 0.5 * r * r, d * d * (0.5 + jnp.log(jnp.maximum(a, d) / d)))
    return jnp.sum(precision.astype(jnp.float64) * loss, dtype=jnp.float64)


def robust_batch(residual, columns, precision, delta):
    """IRLS square-root weights, and weighted residuals and columns (view, 2, v, u)."""
    r = jnp.where(precision > 0, residual, 0)
    a = jnp.abs(r.astype(jnp.float64))
    d = jnp.asarray(delta, jnp.float64)
    sw = jnp.sqrt(precision.astype(jnp.float64) * jnp.minimum(1.0, d / jnp.maximum(a, d)) ** 2)
    sw = sw.astype(jnp.float32)
    return (sw * r).astype(jnp.float32), sw, (columns * sw[:, None]).astype(jnp.float32)


def column_score(columns, images):
    """Per-view J^T r, FP64."""
    return jnp.sum(columns.astype(jnp.float64) * images[:, None].astype(jnp.float64), axis=(-2, -1))


def column_gram(columns):
    """Per-view 2x2 J^T J, FP64."""
    flat = columns.reshape((len(columns), 2, -1)).astype(jnp.float64)
    return jnp.einsum("nik,njk->nij", flat, flat)


def _apply_columns(columns, p):
    p = p.astype(jnp.float32)
    return columns[:, 0] * p[:, 0, None, None] + columns[:, 1] * p[:, 1, None, None]


class Operators:
    """Projector, transpose, pose columns and the solver's whole-scan reductions."""

    def __init__(self, grid, detector, views, *, batch_size=8, ratios=(1.0, 1.0)) -> None:
        self.grid, self.detector = grid, detector
        self.shape = (grid.nz, grid.ny, grid.nx)
        self.n = views
        self.batch = min(batch_size, views)
        self.nb = math.ceil(views / self.batch)
        self.padded_n = self.nb * self.batch
        self.image_shape = (self.padded_n, detector.nv, detector.nu)
        self.ratios = tuple(float(r) for r in ratios)
        self.backend = "cuda" if cuda_backend() else "jax"
        self.sensitivity = jax.jit(self._sensitivity)
        self.noise_rms = jax.jit(self._noise_rms)
        self.initial_gradient = jax.jit(self._initial_gradient)
        self.linearize = jax.jit(self._linearize)
        self.objective = jax.jit(self._objective)
        self.weighted_forward = jax.jit(self._weighted_forward)
        self.weighted_back = jax.jit(self._weighted_back)
        self.pose_back = jax.jit(self._pose_back)

    def project(self, cf, x):
        """Projections of one coefficient batch."""
        return forward_project_planes(cf, x, self.grid, self.detector, backend=self.backend)

    def columns(self, cf, x):
        """Projections and their two detector-translation derivatives, (view, 3, v, u)."""
        return plane_pose_columns(
            cf, x, self.tangent(cf), self.grid, self.detector, backend=self.backend
        )

    def back(self, accumulate, cf, y):
        """Add the matched transpose of one batch to ``accumulate``."""
        return sum_backproject_planes(
            cf, y, self.grid, self.detector, backend=self.backend, accumulate=accumulate
        )

    def pad_coefficients(self, cf):
        """Pad with zero-weight copies of the first view."""
        cf = jnp.asarray(cf, jnp.float32)
        tail = jnp.repeat(cf[:1], self.padded_n - self.n, axis=0).at[:, 9].set(0.0)
        return jnp.concatenate([cf, tail], axis=0)

    def pad_images(self, y):
        """Pad with zero images, which carry zero precision."""
        y = jnp.asarray(y, jnp.float32)
        return jnp.pad(y, ((0, self.padded_n - self.n), (0, 0), (0, 0)))

    def batches(self, a):
        """Reshape a padded per-view array into (batches, batch, ...)."""
        return a.reshape((self.nb, self.batch, *a.shape[1:]))

    def tangent(self, cf):
        """Coefficient derivatives for one native pixel of detector u and v translation."""
        t = jnp.zeros((len(cf), 2, 14), jnp.float32)
        t = (
            t.at[:, 0, 4]
            .set(-self.ratios[0] * cf[:, 1])
            .at[:, 0, 8]
            .set(-self.ratios[0] * cf[:, 5])
        )
        return (
            t.at[:, 1, 4]
            .set(-self.ratios[1] * cf[:, 2])
            .at[:, 1, 8]
            .set(-self.ratios[1] * cf[:, 6])
        )

    def _zeros(self):
        return jnp.zeros(self.shape, jnp.float32)

    def _sensitivity(self, cf):
        ones = jnp.ones(self.shape, jnp.float32)
        raw = jax.lax.scan(
            lambda a, c: (self.back(a, c, self.project(c, ones)), None),
            self._zeros(),
            self.batches(cf),
        )[0]
        top = jnp.maximum(raw.max(), jnp.float32(1e-8))
        return jnp.maximum(raw, jnp.float32(0.01) * top), top

    def _noise_rms(self, cf, precision):
        sqrtw = jnp.sqrt(precision)
        return probe_gradient_rms(
            cf[: self.n],
            self.grid,
            self.detector,
            lambda images, _start, c, total: self.back(total, c, images),
            batch_size=self.batch,
            read_weights=lambda start, size: jax.lax.dynamic_slice_in_dim(sqrtw, start, size),
        )

    def _initial_gradient(self, cf, x, y, precision, delta):
        def step(a, z):
            c, yb, wb = z
            r = self.project(c, x) - yb
            # Huber, not redescending: the initialization never discards data.
            q = (wb.astype(jnp.float64) * jnp.clip(r.astype(jnp.float64), -delta, delta)).astype(
                jnp.float32
            )
            return self.back(a, c, q), None

        batches = (self.batches(cf), self.batches(y), self.batches(precision))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _linearize(self, cf, x, y, precision, delta):
        def step(state, z):
            gx, dd = state
            c, yb, wb = z
            raw = self.columns(c, x)
            r = raw[:, 0] - yb
            q, sw, j = robust_batch(r, raw[:, 1:], wb, delta)
            gx = self.back(gx, c, sw * q)
            dd = plane_weighted_diagonal(c, sw, dd, self.grid, self.detector, backend=self.backend)
            valid = jnp.sum(wb > 0, dtype=jnp.int64)
            down = jnp.sum((wb > 0) & (sw.astype(jnp.float64) ** 2 < 0.99 * wb), dtype=jnp.int64)
            mass = jnp.sum(sw.astype(jnp.float64) ** 2)
            out = (sw, j, column_score(j, q), column_gram(j), robust_value(r, wb, delta))
            return (gx, dd), (*out, valid, down, mass)

        batches = (self.batches(cf), self.batches(y), self.batches(precision))
        state, out = jax.lax.scan(step, (self._zeros(), self._zeros()), batches)
        sw, j, g, gram, e, valid, down, mass = out
        n, d = self.n, self.detector
        return (
            *state,
            sw.reshape(self.image_shape),
            j.reshape((self.padded_n, 2, d.nv, d.nu)),
            g.reshape(self.padded_n, 2)[:n].reshape(-1),
            gram.reshape(self.padded_n, 2, 2)[:n],
            jnp.sum(e),
            jnp.stack([jnp.sum(valid), jnp.sum(down), jnp.sum(mass)]),
        )

    def _objective(self, cf, x, y, precision, delta):
        def step(total, z):
            c, yb, wb = z
            return total + robust_value(self.project(c, x) - yb, wb, delta), None

        batches = (self.batches(cf), self.batches(y), self.batches(precision))
        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), batches)[0]

    def _weighted_forward(self, cf, x, sw, j):
        def step(z):
            c, s, jb = z
            q = self.project(c, x) * s
            return q, column_score(jb, q)

        q, p = jax.lax.map(step, (self.batches(cf), self.batches(sw), self.batches(j)))
        return q.reshape(self.image_shape), p.reshape(self.padded_n, 2)[: self.n].reshape(-1)

    def _padded_poses(self, p):
        return jnp.pad(p.reshape(self.n, 2), ((0, self.padded_n - self.n), (0, 0)))

    def _weighted_back(self, cf, q, sw, j, p):
        def step(a, z):
            c, qb, s, jb, pb = z
            return self.back(a, c, (qb - _apply_columns(jb, pb)) * s), None

        batches = tuple(map(self.batches, (cf, q, sw, j, self._padded_poses(p))))
        return jax.lax.scan(step, self._zeros(), batches)[0]

    def _pose_back(self, cf, sw, j, p):
        def step(a, z):
            c, s, jb, pb = z
            return self.back(a, c, _apply_columns(jb, pb) * s), None

        batches = tuple(map(self.batches, (cf, sw, j, self._padded_poses(p))))
        return jax.lax.scan(step, self._zeros(), batches)[0]
