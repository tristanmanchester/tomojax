"""Smooth-plus-jump detector-motion prior and its exact global pose solves.

Per-view translations p (native detector pixels, two per view) split into a
cubic spline component S a, knots every ``knot_spacing`` views, and a residual
e = (I - U U^T) p, U an orthonormal basis of S. The prior is a group Huber on
the view-to-view differences of e, so steps and slips cost one jump while the
spline part is free. The common rigid translation of the object (gauge G) is
removed by a small constraint.

H = F + Q K Q with F the 2x2 data blocks, K the reweighted difference Laplacian
and Q = I - U U^T. A = F + K is 2x2-block tridiagonal, solved by parallel cyclic
reduction; Woodbury with B = [U, K U] reduces Q K Q to a dense system of four
times the knot count. All of this is FP64 and never splits views into batches.
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


def _inv2(a):
    det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    inverse = jnp.stack([a[..., 1, 1], -a[..., 0, 1], -a[..., 1, 0], a[..., 0, 0]], -1)
    return inverse.reshape(a.shape) / det[..., None, None]


def _pcr_factor(diag, off):
    """Parallel cyclic reduction of an SPD 2x2-block tridiagonal matrix."""
    upper = off[:, None, None] * jnp.eye(2, dtype=jnp.float64)
    lower = _shift(upper, -1)
    d, factors = diag, []
    for level in range(math.ceil(math.log2(len(diag)))):
        step = 1 << level
        inverse = _inv2(d)
        left = -lower @ _shift(inverse, -step)
        right = -upper @ _shift(inverse, step)
        factors.append((left, right))
        d = d + left @ _shift(upper, -step) + right @ _shift(lower, step)
        lower = left @ _shift(lower, -step)
        upper = right @ _shift(upper, step)
    return tuple(factors), _inv2(d)


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


def _tridiagonal(blocks, w):
    """Factor F + K, K the first-difference Laplacian with edge weights ``w``."""
    kdiag = jnp.pad(w, (0, 1)) + jnp.pad(w, (1, 0))
    return _pcr_factor(
        blocks + kdiag[:, None, None] * jnp.eye(2, dtype=jnp.float64), -jnp.pad(w, (0, 1))
    )


class MotionPrior:
    """The prior, its IRLS weights and the global Gauss-Newton pose solves.

    ``gauge`` is (2 views, 3): the native-pixel detector response of each view
    to a unit object translation. ``delta`` is the Huber width in pixels.
    """

    def __init__(self, views: int, gauge: np.ndarray, *, knot_spacing: int, delta: float) -> None:
        t = np.linspace(-1, 1, views)
        knots = list(range(0, views, knot_spacing))
        if knots[-1] != views - 1:
            knots.append(views - 1)
        k = min(3, len(knots) - 1)
        spline = make_interp_spline(t[knots], np.zeros(len(knots)), k=k)
        design = BSpline.design_matrix(t, spline.t, k).toarray()
        u, _ = np.linalg.qr(np.kron(design, np.eye(2)), mode="reduced")
        g, s, _ = np.linalg.svd(np.asarray(gauge, np.float64), full_matrices=False)
        g = g[:, s > max(gauge.shape) * np.finfo(float).eps * max(s.max(initial=0.0), 1.0)]
        # Innovation constraints: the spline space and the gauge outside it.
        residual = g - u @ (u.T @ g)
        left, r, _ = np.linalg.svd(residual, full_matrices=False)
        keep = r > max(residual.shape) * np.finfo(float).eps * max(np.linalg.norm(g, 2), 1.0)
        c, _ = np.linalg.qr(np.column_stack((u, left[:, keep])), mode="reduced")
        self.n, self.delta = views, float(delta)
        self.U = jnp.asarray(u, jnp.float64)
        self.G = jnp.asarray(g, jnp.float64)
        self.C = jnp.asarray(c, jnp.float64)
        self.value = jax.jit(self._value)
        self.weights = jax.jit(lambda p, weight: weight / jnp.maximum(self.delta, self._jumps(p)))
        self.apply = jax.jit(self._apply)
        self.factor = jax.jit(self._factor)
        self.solve = jax.jit(self._solve)
        self.innovation_solve = jax.jit(self._innovation_solve)
        self.project_gauge = jax.jit(lambda p: p - self.G @ (self.G.T @ p))

    def _residual(self, p):
        return p - self.U @ (self.U.T @ p)

    def _jumps(self, p):
        return jnp.linalg.norm(jnp.diff(self._residual(p).reshape(self.n, 2), axis=0), axis=1)

    def _value(self, p, weight):
        t = self._jumps(p)
        return weight * jnp.sum(
            jnp.where(t <= self.delta, t**2 / (2 * self.delta), t - self.delta / 2)
        )

    def _laplacian(self, p, w):
        shape = p.shape
        p = p.reshape((self.n, 2, *shape[1:]))
        dif = jnp.diff(p, axis=0) * w.reshape((-1,) + (1,) * (p.ndim - 1))
        pad = [(0, 0)] * p.ndim
        pad[0] = (0, 1)
        left = jnp.pad(dif, pad)
        pad[0] = (1, 0)
        return (jnp.pad(dif, pad) - left).reshape(shape)

    def _apply(self, p, w):
        return self._residual(self._laplacian(self._residual(p), w))

    def _factor(self, blocks, w):
        tri = _tridiagonal(blocks, w)
        ku = self._laplacian(self.U, w)
        b = jnp.concatenate([self.U, ku], axis=1)
        k = self.U.shape[1]
        zero, eye = jnp.zeros((k, k), jnp.float64), jnp.eye(k, dtype=jnp.float64)
        small = jnp.block([[zero, -eye], [-eye, -self.U.T @ ku]])
        ab = _pcr_solve(tri, b.reshape(self.n, 2, -1)).reshape(2 * self.n, -1)
        small = small + b.T @ ab
        fac = lu_factor((small + small.T) * 0.5)
        ag = _pcr_solve(tri, self.G.reshape(self.n, 2, -1)).reshape(2 * self.n, -1)
        hg = ag - ab @ lu_solve(fac, b.T @ ag)
        gauge = self.G.T @ hg
        return tri, b, ab, fac, hg, (gauge + gauge.T) * 0.5

    def _solve(self, factor, rhs):
        tri, b, ab, fac, hg, gauge = factor
        ar = _pcr_solve(tri, rhs.reshape(self.n, 2)).reshape(-1)
        result = ar - ab @ lu_solve(fac, b.T @ ar)
        return result - hg @ jnp.linalg.solve(gauge, self.G.T @ result)

    def _innovation_solve(self, blocks, w, rhs):
        """Solve F + K restricted to updates orthogonal to the spline and the gauge.

        For such updates d^T Q K Q d = d^T K d, so the plain tridiagonal system
        with the constraints C = [S, G] gives the exact constrained solution.
        """
        tri = _tridiagonal(blocks, w)
        ac = _pcr_solve(tri, self.C.reshape(self.n, 2, -1)).reshape(2 * self.n, -1)
        small = self.C.T @ ac
        ar = _pcr_solve(tri, rhs.reshape(self.n, 2)).reshape(-1)
        chol = jnp.linalg.cholesky(0.5 * (small + small.T))
        return ar - ac @ cho_solve((chol, True), self.C.T @ ar)
