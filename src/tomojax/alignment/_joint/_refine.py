"""Edge-based refinement of the per-view poses after the joint fit.

The volume is fixed. The data term is the robust loss of H(A(p) x - y), where
H removes each detector row's weighted mean and then a normalized Gaussian
local mean (8 native pixels): it keeps edges, which locate a view, and drops
smooth photometric changes, which do not. Only the part of the correction
orthogonal to the prior's spline space and the common gauge may change, so the
broad trajectory of the joint fit is kept and every view's residual motion,
steps included, stays free. A continuous per-view anchor trusts the joint fit
where a view's non-geometric residual energy is ordinary for its neighbours,
and lets an inconsistent view move.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ._operators import column_gram, column_score, robust_batch, robust_value

_SMOOTH_NATIVE_PX = 8.0
_WIDTH_SIGMA = 2.5
_ITERATIONS = 5
_NEIGHBOURS = 5


def _kernel(sigma):
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / float(sigma)) ** 2)
    return jnp.asarray(k / k.sum(), jnp.float32)


class EdgeFilter:
    """Masked linear H = (local-mean removal) after (row-mean removal), and H^T.

    Leading dimensions are free; precision broadcasts to them. Normalizing by
    the smoothed precision avoids artificial edges at masks and borders.
    """

    def __init__(self, ratios) -> None:
        self.ku = _kernel(_SMOOTH_NATIVE_PX * ratios[0])
        self.kv = _kernel(_SMOOTH_NATIVE_PX * ratios[1])

    def smooth(self, a):
        """Zero-extended separable Gaussian; symmetric, so its own transpose."""
        shape = a.shape
        x = a.reshape((-1, 1, *shape[-2:]))
        for k, axis in ((self.ku, -1), (self.kv, -2)):
            filt = k.reshape((1, 1, 1, -1) if axis == -1 else (1, 1, -1, 1))
            x = jax.lax.conv_general_dilated(
                x,
                filt,
                (1, 1),
                "SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
                precision=jax.lax.Precision.HIGHEST,
            )
        return x.reshape(shape)

    def forward(self, a, precision):
        """H a."""
        p = jnp.asarray(precision, jnp.float32)
        m = p > 0
        a = jnp.where(m, a, 0).astype(jnp.float32)
        rows = jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
        a = jnp.where(m, a - jnp.sum(p * a, axis=-1, keepdims=True) / rows, 0)
        den = jnp.maximum(self.smooth(p), jnp.finfo(jnp.float32).tiny)
        return jnp.where(m, a - self.smooth(p * a) / den, 0)

    def transpose(self, z, precision):
        """H^T z."""
        p = jnp.asarray(precision, jnp.float32)
        m = p > 0
        z = jnp.where(m, z, 0).astype(jnp.float32)
        den = jnp.maximum(self.smooth(p), jnp.finfo(jnp.float32).tiny)
        q = jnp.where(m, z - p * self.smooth(z / den), 0)
        rows = jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
        return jnp.where(m, q - p * jnp.sum(q, axis=-1, keepdims=True) / rows, 0)


def relative_precision(energy, sigma):
    """2 (B + s^2) / (B + E + 2 s^2), B the median energy of the ten neighbouring views."""
    n = len(energy)
    neighbours = []
    for offset in range(-_NEIGHBOURS, _NEIGHBOURS + 1):
        if offset:
            index = jnp.arange(n) + offset
            inside = (index >= 0) & (index < n)
            neighbours.append(jnp.where(inside, energy[jnp.clip(index, 0, n - 1)], jnp.nan))
    reference = jnp.maximum(jnp.nanmedian(jnp.stack(neighbours), axis=0), 0)
    variance = jnp.maximum(jnp.asarray(sigma) ** 2, 1e-20)
    return 2 * (reference + variance) / (reference + jnp.maximum(energy, 0) + 2 * variance)


class _EdgeObjective:
    """Batch-wise filtered residuals and columns; only per-view summaries survive."""

    def __init__(self, ops) -> None:
        self.ops = ops
        self.filter = EdgeFilter(ops.ratios)
        self.stats = jax.jit(self._stats)
        self.value = jax.jit(self._value)
        self.consistency = jax.jit(self._consistency)
        self.scale = jax.jit(self._scale)

    def _batches(self, q, y, precision):
        o = self.ops
        return (
            *o.views(q),
            o.batches(y),
            o.batches(precision),
        )

    def _residual(self, z, x):
        nominal, qb, valid, scales, frames, yy, w = z
        w = self.ops.batch_precision(w, valid)
        return self.filter.forward(
            self.ops.project(self.ops.coefficients(nominal, qb, valid, scales, frames), x) - yy,
            w,
        )

    def _stats(self, q, x, y, precision, delta):
        o, h = self.ops, self.filter

        def step(z):
            nominal, qb, valid, scales, frames, yy, w = z
            w = o.batch_precision(w, valid)
            raw = o.columns(nominal, qb, valid, x, scales=scales, frames=frames)
            r = h.forward(raw[:, 0] - yy, w)
            res, _, jw = robust_batch(r, h.forward(raw[:, 1:], w[:, None]), w, delta)
            return column_score(jw, res), column_gram(jw), robust_value(r, w, delta)

        g, gram, energy = jax.lax.map(step, self._batches(q, y, precision))
        n, k = o.n, g.shape[-1]
        return (
            o.parameters.reduce_score(g.reshape(o.padded_n, k)[:n]),
            gram.reshape(o.padded_n, k, k)[:n],
            jnp.sum(energy),
        )

    def _value(self, q, x, y, precision, delta):
        def step(total, z):
            w = self.ops.batch_precision(z[-1], z[2])
            return total + robust_value(self._residual(z, x), w, delta), None

        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), self._batches(q, y, precision))[0]

    def _consistency(self, q, x, y, precision):
        """Per-view mean energy of the smooth residual left after the best pose step."""
        o, h = self.ops, self.filter

        def step(z):
            nominal, qb, valid, scales, frames, yy, w = z
            w = o.batch_precision(w, valid)
            raw = o.columns(nominal, qb, valid, x, scales=scales, frames=frames)
            r, j = raw[:, 0] - yy, raw[:, 1:]
            gram = jnp.einsum("bcvu,bdvu,bvu->bcd", j, j, w, precision=jax.lax.Precision.HIGHEST)
            score = jnp.einsum("bcvu,bvu,bvu->bc", j, r, w, precision=jax.lax.Precision.HIGHEST)
            floor = jnp.maximum(jnp.trace(gram, axis1=1, axis2=2) * 1e-6, 1e-12)
            eye = jnp.eye(gram.shape[-1])
            d = jnp.linalg.solve(gram + floor[:, None, None] * eye, score[..., None])[..., 0]
            orthogonal = r - jnp.einsum("bcvu,bc->bvu", j, d, precision=jax.lax.Precision.HIGHEST)
            photometric = orthogonal - h.forward(orthogonal, w)
            mass = jnp.maximum(jnp.sum(w, axis=(-1, -2)), 1.0)
            return jnp.sum(w * photometric**2, axis=(-1, -2)) / mass

        return jax.lax.map(step, self._batches(q, y, precision)).reshape(-1)[: o.n]

    def _scale(self, q, x, y, precision):
        """Median over views of each view's MAD of the filtered residual."""

        def step(z):
            w = self.ops.batch_precision(z[-1], z[2])
            r = jnp.where(w > 0, self._residual(z, x), jnp.nan)
            med = jnp.nanmedian(r, axis=(-1, -2), keepdims=True)
            return 1.482602218505602 * jnp.nanmedian(jnp.abs(r - med), axis=(-1, -2))

        return jnp.nanmedian(
            jax.lax.map(step, self._batches(q, y, precision)).reshape(-1)[: self.ops.n]
        )


class _Refinement:
    """The fixed-volume edge objective, its anchored prior and one damped step."""

    def __init__(self, x, y, precision, prior, ops, sigma, c0) -> None:
        self.x, self.y, self.precision = x, y, precision
        self.prior, self.ops, self.c0 = prior, ops, c0
        self.objective = _EdgeObjective(ops)
        q = ops.pad_poses(c0)
        model_sigma = float(self.objective.scale(q, x, y, precision))
        if not np.isfinite(model_sigma):
            raise ValueError("no finite residual scale for the pose refinement")
        self.delta = max(_WIDTH_SIGMA * float(sigma), _WIDTH_SIGMA * model_sigma, 1e-8)
        self.first = self.objective.stats(q, x, y, precision, self.delta)
        gram = self.first[1]
        fisher = jnp.median(jnp.trace(gram, axis1=1, axis2=2))
        self.weight = float(sigma * jnp.sqrt(fisher) * np.sqrt(2 * np.log(ops.n)))
        energy = self.objective.consistency(q, x, y, precision)
        self.anchor = relative_precision(energy, float(sigma))[:, None, None] * gram

    def penalty(self, c):
        """Motion prior plus the anchor towards the joint fit's poses."""
        d = self.prior.expand(c - self.c0)
        return self.prior.value(c, self.weight) + 0.5 * jnp.einsum("vi,vij,vj->", d, self.anchor, d)

    def value(self, c):
        """The objective at pose parameters ``c``."""
        q = self.ops.pad_poses(c)
        data = self.objective.value(q, self.x, self.y, self.precision, self.delta)
        return float(data + self.penalty(c))

    def step(self, c, stats, f, lm):
        """Constrained Gauss-Newton with Armijo backtracking; damping x10 up to four times.

        Returns (parameters, objective, step length, step, damping), or a status
        string when no descent step exists or none is accepted.
        """
        prior, anchor = self.prior, self.anchor
        g, gram, _ = stats
        wpose = prior.weights(c, self.weight)
        change = prior.reduce_score(jnp.einsum("vij,vj->vi", anchor, prior.expand(c - self.c0)))
        gp = g + prior.apply(c, wpose) + change
        pd = jnp.diagonal(gram, axis1=1, axis2=2)
        pd = jnp.maximum(pd, jnp.maximum(jnp.max(gram), 1e-8) * 0.01)
        for _ in range(4):
            damped = gram + anchor + jax.vmap(jnp.diag)(lm * pd + 1e-10)
            dc = -prior.innovation_solve(damped, wpose, gp)
            slope = float(jnp.vdot(gp, dc))
            if not np.isfinite(slope):
                raise FloatingPointError("non-finite pose refinement slope")
            if slope >= 0:
                return "stationary"
            for ls in range(8):
                alpha = 0.5**ls
                fn = self.value(c + alpha * dc)
                if fn <= f + 1e-4 * alpha * slope:
                    return c + alpha * dc, fn, alpha, dc, lm
            lm *= 10
        return "line_search_stalled"


def refine(x, p, y, precision, prior, *, ops, sigma):
    """Refine local innovations; the jointly fitted global values stay fixed."""
    if prior.nlocal == 0:
        return p, {"status": "no_local_parameters", "steps": []}
    c = prior.project_gauge(jnp.asarray(p, jnp.float64).reshape(-1))
    y = ops.pad_images(y)
    precision = ops.pad_precision(precision)
    problem = _Refinement(x, y, precision, prior, ops, sigma, c)
    stats = problem.first
    f, lm, status, steps = float(stats[2] + problem.penalty(c)), 0.01, "budget_exhausted", []
    for it in range(_ITERATIONS):
        if it:
            stats = problem.objective.stats(ops.pad_poses(c), x, y, precision, problem.delta)
        result = problem.step(c, stats, f, lm)
        if isinstance(result, str):
            status = result
            break
        c, f, alpha, dc, lm = result
        step = float(jnp.sqrt(jnp.mean(jnp.sum((alpha * prior.expand(dc)) ** 2, axis=1))))
        steps.append({"objective": f, "pose_step_rms": step, "line_step": alpha, "lm": lm})
        lm = max(1e-5, lm * 0.5)
        if step < 1e-5:
            status = "converged"
            break
    return c, {"status": status, "steps": steps, "delta": problem.delta}
