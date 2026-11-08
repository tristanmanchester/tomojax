"""Exact bordered pose solves with translation drift/jumps and rotation size priors.

The local normal matrix is F + R + Q K Q: per-view data blocks F, physical
rotation precision R, and a translation difference Laplacian K outside the
drift spline U (Q = I - U U.T). Block cyclic reduction plus a spline Woodbury
correction solves it with the rigid gauge fixed. Shared calibration columns
form a small Schur border around that complete local system.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


def _shift(a, offset):
    """out[i] = a[i + offset], zero filled."""
    n = a.shape[0]
    if abs(offset) >= n:
        return jnp.zeros_like(a)
    pads = [(0, 0)] * a.ndim
    if offset > 0:
        pads[0] = (0, offset)
        return jnp.pad(a[offset:], pads)
    pads[0] = (-offset, 0)
    return jnp.pad(a[:offset], pads)


def _inverse(blocks):
    """Small SPD block inverse, retaining the two-translation fit's direct arithmetic."""
    if blocks.shape[-1] == 1:
        return 1 / blocks
    if blocks.shape[-1] == 2:
        a, b, c, d = blocks[..., 0, 0], blocks[..., 0, 1], blocks[..., 1, 0], blocks[..., 1, 1]
        return (
            jnp.stack([d, -b, -c, a], axis=-1).reshape(blocks.shape)
            / (a * d - b * c)[..., None, None]
        )
    return jnp.linalg.inv(blocks)


def _pcr_factor(diag, off):
    """Parallel cyclic reduction of an SPD block-tridiagonal matrix (k x k blocks)."""
    upper = (
        off[:, None, None] * jnp.eye(diag.shape[-1], dtype=jnp.float64)
        if off.ndim == 1
        else jax.vmap(jnp.diag)(off)
    )
    lower = _shift(upper, -1)
    d, factors = diag, []
    for level in range(math.ceil(math.log2(len(diag)))):
        step = 1 << level
        inverse = _inverse(d)
        left = -lower @ _shift(inverse, -step)
        right = -upper @ _shift(inverse, step)
        factors.append((left, right))
        d = d + left @ _shift(upper, -step) + right @ _shift(lower, step)
        lower = left @ _shift(lower, -step)
        upper = right @ _shift(upper, step)
    return tuple(factors), _inverse(d)


def _pcr_solve(factor, rhs):
    factors, inverse = factor
    vector = rhs.ndim == 2
    if vector:
        rhs = rhs[..., None]
    for level, (left, right) in enumerate(factors):
        step = 1 << level
        rhs = rhs + left @ _shift(rhs, -step) + right @ _shift(rhs, step)
    result = inverse @ rhs
    return result[..., 0] if vector else result


def _tridiagonal(blocks, w, mask=None):
    """Factor F + K, K the first-difference Laplacian with edge weights ``w``."""
    kdiag = jnp.pad(w, (0, 1)) + jnp.pad(w, (1, 0))
    mask = jnp.ones(blocks.shape[-1], jnp.float64) if mask is None else mask
    diagonal = jax.vmap(jnp.diag)(kdiag[:, None] * mask)
    return _pcr_factor(blocks + diagonal, -jnp.pad(w, (0, 1))[:, None] * mask)


def _spline_design(views, spacing):
    """Cubic B-spline design matrix (views, knots) with a knot every ``spacing`` views."""
    t = np.linspace(-1, 1, views)
    knots = list(range(0, views, spacing))
    if knots[-1] != views - 1:
        knots.append(views - 1)
    k = min(3, len(knots) - 1)
    spline = make_interp_spline(t[knots], np.zeros(len(knots)), k=k)
    return BSpline.design_matrix(t, spline.t, k).toarray()


def _orthonormal_columns(matrix):
    """An orthonormal basis without null or repeated constraint columns."""
    matrix = np.asarray(matrix, np.float64)
    if not matrix.size:
        return np.zeros((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(singular.max(initial=0), 1)
    return u[:, singular > tolerance]


class MotionPrior:
    """Translation drift/jumps, a physical rotation prior, and an exact global border.

    Local translations have a free cubic drift spline and Student-t innovations.
    Rotations have a zero-mean Gaussian size prior with ``rotation_sigma_deg``
    standard deviation, in physical radians rather than numerical pixel units.
    Multiplication by ``noise_sigma**2`` puts that prior in the unnormalised
    data loss's units. Shared centre and angle parameters are not penalised.

    ``blocks`` contains every view's local/global Gram block. The local system
    is solved by block cyclic reduction and a spline Woodbury correction, with
    rigid gauge constraints. A small Schur border then couples all views to the
    single copy of each global parameter. No view-wise global approximation is
    made. ``innovation_solve`` holds that border exactly fixed for refinement.
    """

    def __init__(
        self,
        views: int,
        gauge: np.ndarray,
        *,
        knot_spacing: tuple[int, ...],
        delta: float,
        parameters=None,
        rotation_scales=None,
        noise_sigma=0.0,
        rotation_sigma_deg=0.5,
    ) -> None:
        self.parameters = parameters
        self.n = views
        self.nlocal = len(knot_spacing) if parameters is None else parameters.nlocal
        self.nglobal = 0 if parameters is None else parameters.nglobal
        self.k, self.size = self.nlocal + self.nglobal, views * self.nlocal + self.nglobal
        self.local_size, self.delta = views * self.nlocal, float(delta)
        local = tuple(range(self.nlocal)) if parameters is None else parameters.local
        spacing = tuple(knot_spacing[i] for i in local)
        translated = (
            np.ones(self.nlocal)
            if parameters is None
            else np.asarray([i >= 3 for i in local], np.float64)
        )
        self.translation_mask = jnp.asarray(translated, jnp.float64)
        columns = [
            np.kron(_spline_design(views, step), np.eye(self.nlocal)[:, [i]])
            for i, step in enumerate(spacing)
            if translated[i]
        ]
        u = (
            np.linalg.qr(np.concatenate(columns, axis=1), mode="reduced")[0]
            if columns
            else np.zeros((self.local_size, 0))
        )
        gauge = np.asarray(gauge, np.float64)
        if parameters is not None:
            gauge = gauge.reshape(views, parameters.full_dofs, -1)[:, local, :].reshape(
                self.local_size, gauge.shape[-1]
            )
            # A constant parallel lab-x shift is indistinguishable from a global
            # detector-u centre shift. Fix its local mean and let centre own it.
            # Cone centre and object translation differ with depth, so imposing
            # the same constraint there would remove a measurable motion.
            if not parameters.cone and "centre_u" in parameters.globals and 3 in local:
                constant = np.zeros((views, self.nlocal))
                constant[:, local.index(3)] = 1
                gauge = np.column_stack([gauge, constant.reshape(-1)])
        g = _orthonormal_columns(gauge)
        self.U, self.G = jnp.asarray(u), jnp.asarray(g)
        outside = _orthonormal_columns(g - u @ (u.T @ g))
        self.C = jnp.asarray(np.column_stack([u, outside]))
        rotation_precision = np.zeros((views, self.nlocal))
        if parameters is not None and "rotations" in parameters.fit:
            scales = np.broadcast_to(
                np.asarray(rotation_scales, np.float64), (views, parameters.full_dofs)
            )[:, local]
            radians = np.deg2rad(rotation_sigma_deg)
            rotation_precision = (float(noise_sigma) * scales / radians) ** 2 * (1 - translated)
        self.rotation_sigma_deg = float(rotation_sigma_deg)
        self.rotation_precision = jnp.asarray(rotation_precision.reshape(-1), jnp.float64)
        self.value = jax.jit(self._value)
        self.weights = jax.jit(
            lambda p, weight: weight * self.delta / (self.delta**2 + self._squared_jumps(p))
        )
        self.apply = jax.jit(self._apply)
        self.factor = jax.jit(self._factor)
        self.solve = jax.jit(self._solve)
        self.innovation_solve = jax.jit(self._innovation_solve)
        self.project_gauge = jax.jit(self._project_gauge)

    def expand(self, state):
        """Projector rows with one shared value per global coordinate."""
        if self.parameters is None:
            return jnp.asarray(state).reshape(self.n, self.k)
        return self.parameters.expand(state)

    def reduce_score(self, rows):
        """Transpose of the parameter broadcast."""
        if self.parameters is None:
            return jnp.asarray(rows).reshape(-1)
        return self.parameters.reduce_score(rows)

    def _project_gauge(self, state):
        local = state[: self.local_size]
        local = local - self.G @ (self.G.T @ local)
        return jnp.concatenate([local, state[self.local_size :]])

    def _residual(self, p):
        return p - self.U @ (self.U.T @ p)

    def _squared_jumps(self, p):
        residual = self._residual(p[: self.local_size]).reshape(self.n, self.nlocal)
        jumps = jnp.diff(residual, axis=0) * self.translation_mask
        return jnp.sum(jumps**2, axis=1)

    def _value(self, p, weight):
        jumps = (
            0.5 * weight * self.delta * jnp.sum(jnp.log1p(self._squared_jumps(p) / self.delta**2))
        )
        return jumps + 0.5 * jnp.sum(self.rotation_precision * p[: self.local_size] ** 2)

    def _laplacian(self, p, w):
        shape = p.shape
        p = p.reshape((self.n, self.nlocal, *shape[1:]))
        weights = w[:, None] * self.translation_mask[None]
        weights = weights.reshape((*weights.shape, *((1,) * (p.ndim - 2))))
        dif = jnp.diff(p, axis=0) * weights
        pad = [(0, 0)] * p.ndim
        pad[0] = (0, 1)
        left = jnp.pad(dif, pad)
        pad[0] = (1, 0)
        return (jnp.pad(dif, pad) - left).reshape(shape)

    def _apply(self, p, w):
        local = p[: self.local_size]
        jump = self._residual(self._laplacian(self._residual(local), w))
        return jnp.concatenate([jump + self.rotation_precision * local, jnp.zeros(self.nglobal)])

    def _local_factor(self, blocks, w):
        blocks = blocks + jax.vmap(jnp.diag)(self.rotation_precision.reshape(self.n, self.nlocal))
        tri = _tridiagonal(blocks, w, self.translation_mask)
        ku = self._laplacian(self.U, w)
        b = jnp.concatenate([self.U, ku], axis=1)
        k = self.U.shape[1]
        if k:
            zero, eye = jnp.zeros((k, k), jnp.float64), jnp.eye(k, dtype=jnp.float64)
            small = jnp.block([[zero, -eye], [-eye, -self.U.T @ ku]])
            ab = _pcr_solve(tri, b.reshape(self.n, self.nlocal, -1)).reshape(self.local_size, -1)
            small = small + b.T @ ab
            fac = lu_factor((small + small.T) * 0.5)
        else:
            ab, fac = b, None
        ag = _pcr_solve(tri, self.G.reshape(self.n, self.nlocal, -1)).reshape(self.G.shape)
        hg = ag - ab @ lu_solve(fac, b.T @ ag) if k else ag
        gauge = self.G.T @ hg
        chol = jnp.linalg.cholesky((gauge + gauge.T) * 0.5) if self.G.shape[1] else None
        return tri, b, ab, fac, hg, chol

    def _local_solve(self, factor, rhs):
        tri, b, ab, fac, hg, chol = factor
        ar = _pcr_solve(tri, rhs.reshape((self.n, self.nlocal, *rhs.shape[1:]))).reshape(rhs.shape)
        result = ar - ab @ lu_solve(fac, b.T @ ar) if self.U.shape[1] else ar
        if self.G.shape[1]:
            result = result - hg @ cho_solve((chol, True), self.G.T @ result)
        return result

    def _factor(self, blocks, w):
        local = (
            self._local_factor(blocks[:, : self.nlocal, : self.nlocal], w) if self.nlocal else None
        )
        border = blocks[:, : self.nlocal, self.nlocal :].reshape(self.local_size, self.nglobal)
        solved = self._local_solve(local, border) if self.nlocal and self.nglobal else border
        schur = jnp.sum(blocks[:, self.nlocal :, self.nlocal :], axis=0) - border.T @ solved
        chol = jnp.linalg.cholesky((schur + schur.T) * 0.5) if self.nglobal else None
        return local, border, solved, chol

    def _solve(self, factor, rhs):
        local, border, solved, chol = factor
        direct = self._local_solve(local, rhs[: self.local_size]) if self.nlocal else rhs[:0]
        if not self.nglobal:
            return direct
        shared = cho_solve((chol, True), rhs[self.local_size :] - border.T @ direct)
        return jnp.concatenate([direct - solved @ shared, shared])

    def _innovation_solve(self, blocks, w, rhs):
        """Exact local innovation step with drift, rigid gauge and globals held."""
        if not self.nlocal:
            return jnp.zeros_like(rhs)
        blocks = blocks[:, : self.nlocal, : self.nlocal]
        blocks = blocks + jax.vmap(jnp.diag)(self.rotation_precision.reshape(self.n, self.nlocal))
        tri = _tridiagonal(blocks, w, self.translation_mask)
        ar = _pcr_solve(tri, rhs[: self.local_size].reshape(self.n, self.nlocal)).reshape(-1)
        if self.C.shape[1]:
            ac = _pcr_solve(tri, self.C.reshape(self.n, self.nlocal, -1)).reshape(self.C.shape)
            small = self.C.T @ ac
            chol = jnp.linalg.cholesky(0.5 * (small + small.T))
            ar = ar - ac @ cho_solve((chol, True), self.C.T @ ar)
        return jnp.concatenate([ar, jnp.zeros(self.nglobal)])
