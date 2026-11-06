"""Finite-budget joint Gauss-Newton on the volume and the detector translations.

Each outer step linearizes the robust data term at the current volume and
poses, freezes the IRLS weights and the lagged TV conductance, and eliminates
the poses exactly: the volume step solves the Schur system by preconditioned CG
(Jacobi: data diagonal + damping + TV diagonal), and the pose step follows from
it through the global prior solve. A backtracking line search on the true
objective chooses the volume step length and the fraction of the pose step.
Only scalar decisions leave the device.
"""

from dataclasses import dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import _tv as tv
from ._operators import translated


@dataclass(frozen=True)
class Budget:
    """Outer Gauss-Newton steps, inner CG iterations and initial FISTA steps."""

    outer: int
    inner: int
    init_iterations: int = 5
    lm_initial: float = 0.05


_CG_RTOL = 0.03
_LM_MIN = 1e-4
_POSE_TOL_PX = 0.003
_OBJECTIVE_TOL = 1e-5
_MIN_OUTER = 8


def _pcg(matvec, rhs, invdiag, maxiter):
    """Preconditioned CG from zero; FP32 vectors, FP64 reductions.

    Returns the iterate, the iteration count and the relative recursive residual.
    """
    b2 = tv.dot(rhs, rhs)
    p = rhs * invdiag

    def step(state):
        k, _, x, r, p, rz = state
        ap = matvec(p)
        pap = tv.dot(p, ap)
        safe = jnp.isfinite(pap) & (pap > 0)
        alpha = jnp.where(safe, rz / pap, 0).astype(jnp.float32)
        x, r = x + alpha * p, r - alpha * ap
        r2 = tv.dot(r, r)
        new_rz = tv.dot(r, r * invdiag)
        valid = jnp.isfinite(new_rz) & (new_rz > 0)
        beta = jnp.where(valid & safe, new_rz / rz, 0).astype(jnp.float32)
        go = safe & valid & (r2 > _CG_RTOL**2 * b2)
        return k + 1, go, x, r, r * invdiag + beta * p, new_rz

    initial = (jnp.int32(0), b2 > 0, jnp.zeros_like(rhs), rhs, p, tv.dot(rhs, p))
    k, _, x, r, _, _ = jax.lax.while_loop(lambda s: (s[0] < maxiter) & s[1], step, initial)
    return x, k, jnp.sqrt(tv.dot(r, r) / jnp.maximum(b2, 1e-300))


def _line_search(trial, f, gx_dx, gp_dc):
    """Armijo backtracking on the volume step, with the best of three pose fractions.

    ``trial(a, beta)`` returns the objective after volume step ``a`` and pose
    step ``a * beta``. A parabola through beta = 0, 1/2, 1 may add a fourth try.
    Returns (a, beta, objective, extra) for the accepted step, or None.
    """
    for ls in range(10):
        a = 0.5**ls
        tries = {beta: trial(a, beta) for beta in (1.0, 0.0, 0.5)}
        f1, f0, fh = tries[1.0][0], tries[0.0][0], tries[0.5][0]
        curvature, slope = 2 * (f1 + f0 - 2 * fh), 4 * fh - f1 - 3 * f0
        if curvature > 0:
            vertex = float(np.clip(-slope / (2 * curvature), 0, 1))
            if min(abs(vertex - t) for t in tries) > 0.01:
                tries[vertex] = trial(a, vertex)
        beta = min(tries, key=lambda t: tries[t][0])
        derivative = gx_dx + beta * gp_dc
        if derivative < 0 and tries[beta][0] <= f + 1e-4 * a * derivative:
            return a, beta, tries[beta]
    return None


class _Level:
    """One level's data, compiled operators and damped Gauss-Newton step."""

    def __init__(self, ops, prior, y, precision, cf0, *, lam, eps, scales, delta, inner) -> None:
        self.ops, self.prior, self.cf0 = ops, prior, cf0
        self.y, self.precision, self.lam, self.delta = y, precision, lam, delta
        self.tv_value = jax.jit(lambda x: tv.energy(x, eps, scales))
        self.conduct = jax.jit(lambda x: tv.conductance(x, eps, scales))
        self.diffuse = jax.jit(lambda x, d: tv.diffusion(x, d, scales))
        self.gradient = jax.jit(lambda gx, x, d: gx + jnp.float32(lam) * tv.diffusion(x, d, scales))
        self.inverse = jax.jit(
            lambda d, damp, dd, lm: tv.inverse_diagonal(d, damp, dd, lam, lm, scales)
        )
        self.rhs = jax.jit(
            lambda cf, sw, j, sp, gx, free: (ops.pose_back(cf, sw, j, sp) - gx) * free
        )
        self.norm = jax.jit(tv.dot)
        # Batch by batch: a whole-sinogram FP64 product would be a stack-sized temporary.
        self.energy = jax.jit(
            lambda q: jnp.sum(
                jax.lax.map(lambda b: tv.dot(b, b), ops.batches(q)), dtype=jnp.float64
            )
        )

        @jax.jit
        def schur(cf, sw, j, factor, free, d, damp, lm, rhs, invdiag):
            def matvec(v):
                v = v * free
                q, jq = ops.weighted_forward(cf, v, sw, j)
                out = ops.weighted_back(cf, q, sw, j, prior.solve(factor, jq))
                out = (
                    out
                    + jnp.float32(lam) * tv.diffusion(v, d, scales)
                    + (jnp.float32(lm) * damp) * v
                )
                return out * free

            return _pcg(matvec, rhs, invdiag, inner)

        self.schur = schur

    def initial_volume(self, cf, lipschitz, iterations):
        """FISTA from zero on the Huber data term, without TV."""
        ops, y, precision, delta = self.ops, self.y, self.precision, self.delta

        @jax.jit
        def run(x, cf, lipschitz):
            def fista(_, state):
                old, z, t = state
                new = jnp.maximum(
                    z - ops.initial_gradient(cf, z, y, precision, delta) / lipschitz, 0
                )
                tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
                return new, new + ((t - 1) / tn).astype(jnp.float32) * (new - old), tn

            return jax.lax.fori_loop(0, iterations, fista, (x, x, jnp.asarray(1.0, jnp.float64)))[0]

        return run(jnp.zeros(self.ops.shape, jnp.float32), cf, lipschitz)

    def trial(self, x, c, dx, dc, pose_weight, a, beta):
        """Objective after volume step a and pose step a * beta, with the new state."""
        xt = x + jnp.float32(a) * dx
        cc = c + a * beta * dc
        ccf = translated(self.cf0, cc, self.ops.ratios)
        total = float(self.ops.objective(ccf, xt, self.y, self.precision, self.delta))
        total += float(self.lam * self.tv_value(xt))
        return total + float(self.prior.value(cc, pose_weight)), xt, cc, ccf

    def step(self, x, c, cf, linear, damp, lm, pose_weight, f):
        """One damped step: Schur volume solve, exact pose back-substitution, line search.

        Raises the damping tenfold, up to five times, until the line search accepts.
        Returns the accepted step with its predicted reduction, or None.
        """
        ops, prior = self.ops, self.prior
        gx0, dd, sw, j, gr, gram = linear
        d = self.conduct(x)
        gx = self.gradient(gx0, x, d)
        wpose = prior.weights(c, pose_weight)
        gp = gr + prior.apply(c, wpose)
        free = (x > 0) | (gx < 0)
        pd = jnp.diagonal(gram, axis1=1, axis2=2)
        pd = jnp.maximum(pd, jnp.maximum(pd.max(), 1e-8) * 0.01)
        for _ in range(5):
            factor = prior.factor(gram + jax.vmap(jnp.diag)(lm * pd + 1e-10), wpose)
            rhs = self.rhs(cf, sw, j, prior.solve(factor, gp), gx, free)
            invdiag = self.inverse(d, damp, dd, lm)
            dx, cg_steps, cg_residual = self.schur(
                cf, sw, j, factor, free, d, damp, lm, rhs, invdiag
            )
            dx = jnp.maximum(jnp.where(free, x + dx, x), 0) - x
            q, jadx = ops.weighted_forward(cf, dx, sw, j)
            dc = -prior.solve(factor, gp + jadx)
            gx_dx, gp_dc = float(self.norm(gx, dx)), float(jnp.vdot(gp, dc))
            accepted = None
            if gx_dx + gp_dc < 0:
                trial = partial(self.trial, x, c, dx, dc, pose_weight)
                accepted = _line_search(trial, f, gx_dx, gp_dc)
            if accepted:
                a, beta, new = accepted
                dp = dc.reshape(ops.n, 2)
                curvature = float(self.energy(q)) + self.lam * float(
                    self.norm(dx, self.diffuse(dx, d))
                )
                curvature += beta * 2 * float(jnp.vdot(dc, jadx))
                pose = jnp.einsum("ni,nij,nj->", dp, gram, dp) + jnp.vdot(
                    dc, prior.apply(dc, wpose)
                )
                curvature += beta**2 * float(pose)
                predicted = -a * (gx_dx + beta * gp_dc) - 0.5 * a * a * curvature
                change = float(jnp.sqrt(jnp.mean(jnp.sum((a * beta * dp) ** 2, axis=1))))
                return a, beta, new, predicted, change, lm, int(cg_steps), float(cg_residual)
            lm *= 10
        return None


def solve(y, cf0, precision, prior, weight, eps, *, ops, budget, x0, p0, pose_sigma, delta, scales):
    """Fit the volume and per-view translations (native pixels) at one level.

    Returns the volume, the (views, 2) translations and a record of the steps.
    """
    start, n = time.perf_counter(), ops.n
    c = (
        jnp.zeros(2 * n, jnp.float64)
        if p0 is None
        else prior.project_gauge(jnp.asarray(p0, jnp.float64).reshape(-1))
    )
    cf0 = ops.pad_coefficients(cf0)
    cf = translated(cf0, c, ops.ratios)
    y = ops.pad_images(y)
    precision = ops.pad_images(
        np.ones((n, ops.detector.nv, ops.detector.nu), np.float32)
        if precision is None
        else precision
    )
    level = _Level(
        ops,
        prior,
        y,
        precision,
        cf0,
        lam=weight,
        eps=eps,
        scales=scales,
        delta=delta,
        inner=budget.inner,
    )
    damp, lipschitz = ops.sensitivity(cf)
    if x0 is None:
        x = level.initial_volume(cf, lipschitz, budget.init_iterations)
    else:
        x = jnp.maximum(jnp.asarray(x0, jnp.float32), 0)
    gx0, dd, sw, j, gr, gram, data, obstats = ops.linearize(cf, x, y, precision, delta)
    fisher = jnp.trace(gram, axis1=1, axis2=2)
    pose_weight = float(pose_sigma * jnp.sqrt(jnp.median(fisher)) * np.sqrt(2 * np.log(n)))
    f = float(data + weight * level.tv_value(x) + prior.value(c, pose_weight))
    history, lm, stable, status = [], budget.lm_initial, 0, "budget_exhausted"
    for it in range(budget.outer):
        if it:
            # Release the previous weights and columns first: two stack-sized sets
            # would not fit beside each other on a small device.
            del gx0, dd, sw, j, gr, gram
            gx0, dd, sw, j, gr, gram, data, obstats = ops.linearize(cf, x, y, precision, delta)
        result = level.step(x, c, cf, (gx0, dd, sw, j, gr, gram), damp, lm, pose_weight, f)
        if result is None:
            status = "line_search_stalled"
            break
        a, beta, (fn, x, c, cf), predicted, change, lm, cg_steps, cg_residual = result
        rho = (f - fn) / max(predicted, 1e-30)
        reduction = (f - fn) / max(abs(f), 1e-30)
        valid, down, _ = map(float, np.asarray(obstats))
        history.append(
            {
                "objective": fn,
                "pose_step_rms_px": change,
                "line_step": a,
                "pose_fraction": beta,
                "lm": lm,
                "gain_ratio": rho,
                "cg_iterations": cg_steps,
                "cg_relative_residual": cg_residual,
                "fraction_downweighted": down / max(valid, 1),
            }
        )
        f = fn
        if rho > 0.75 and a == 1:
            lm = max(_LM_MIN, lm / 2)
        elif rho < 0.25 or a < 0.5:
            lm = min(lm * 3, 1e6)
        stable = stable + 1 if change < _POSE_TOL_PX and reduction < _OBJECTIVE_TOL else 0
        if stable >= 2 and it + 1 >= _MIN_OUTER:
            status = "converged"
            break
    record = {
        "status": status,
        "seconds": time.perf_counter() - start,
        "steps": history,
        "tv_weight": float(weight),
    }
    return x, c.reshape(n, 2).astype(jnp.float32), record
