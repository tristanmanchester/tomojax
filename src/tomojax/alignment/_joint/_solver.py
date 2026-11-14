"""Finite-budget joint Gauss-Newton on the volume and the per-view poses.

Each outer step linearizes the robust data term at the current volume and
poses, freezes the IRLS weights and the lagged TV conductance, and eliminates
the poses exactly: the volume step solves the Schur system by preconditioned CG
(Jacobi: data diagonal + damping + TV diagonal), and the pose step follows from
it through the global prior solve. The views' shared offset and ramp are held
(``_common``): they add two residual rows that couple every view, solved by
Woodbury in the pose block. A backtracking line search on the true objective
chooses the volume step length and the fraction of the pose step. Only scalar
decisions leave the device.
"""

from dataclasses import asdict, dataclass
from functools import partial
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _common
from . import _tv as tv
from ._krylov import solve as krylov_solve
from ._operators import profiled_residual, robust_value


@dataclass(frozen=True)
class Budget:
    """Internal work limits for one level; convergence can stop sooner."""

    outer: int
    inner: int
    init_iterations: int = 5
    lm_initial: float = 0.001


_CG_RTOL = 0.03
_LM_MIN = 1e-8
_POSE_TOL_PX = 0.01
_MIN_OUTER = 4


def _gain_ratio(actual, predicted):
    """A non-descending model gives no evidence for decreasing LM damping."""
    return actual / predicted if predicted > 0 else 0.0


def _line_search(trial, f, gx_dx, gp_dc):
    """Compare pose fractions at each image step before accepting joint descent.

    ``trial(a, beta)`` returns the objective after volume step ``a`` and pose
    step ``a * beta``. Always compare beta = 0, 1/2, 1 and an optional parabolic
    interpolant: image improvement must not conceal a harmful pose update.
    Backtrack until the best trial satisfies Armijo. Returns
    (a, beta, trial_result) for the accepted step, or None.
    """
    for ls in range(10):
        a = 0.5**ls
        joint = trial(a, 1.0)
        tries = {1.0: joint, 0.0: trial(a, 0.0), 0.5: trial(a, 0.5)}
        f1, f0, fh = tries[1.0][0], tries[0.0][0], tries[0.5][0]
        curvature, slope = 2 * (f1 + f0 - 2 * fh), 4 * fh - f1 - 3 * f0
        if curvature > 0:
            vertex = float(np.clip(-slope / (2 * curvature), 0, 1))
            if min(abs(vertex - t) for t in tries) > 0.01:
                tries[vertex] = trial(a, vertex)
        beta = min(tries, key=lambda t: (tries[t][0], t))
        derivative = gx_dx + beta * gp_dc
        if derivative < 0 and tries[beta][0] <= f + 1e-4 * a * derivative:
            return a, beta, tries[beta]
    return None


class _StepContext(NamedTuple):
    """What every step from one linearisation shares, whatever its damping."""

    q: Any
    dd: Any
    sw: Any
    j: Any
    gram: Any
    residual: Any
    common: Any
    damp: Any
    d: Any
    gx: Any
    wpose: Any
    gp: Any
    free: Any
    pd: Any


class _Proposal(NamedTuple):
    """A clipped volume step, its pose back-substitution and what the line search needs."""

    dx: Any
    dc: Any
    r: Any
    jadx: Any
    bdx: Any
    linear_step: Any


class _Level:
    """One level's data, compiled operators and damped Gauss-Newton step."""

    def __init__(
        self,
        ops,
        prior,
        y,
        precision,
        *,
        lam,
        eps,
        scales,
        delta,
        inner,
    ) -> None:
        self.ops, self.prior = ops, prior
        self.y, self.precision, self.lam, self.delta = y, precision, lam, delta
        self.tv_value = jax.jit(lambda x: tv.energy(x, eps, scales))
        self.conduct = jax.jit(lambda x: tv.conductance(x, eps, scales))
        self.diffuse = jax.jit(lambda x, d: tv.diffusion(x, d, scales))
        self.gradient = jax.jit(lambda gx, x, d: gx + jnp.float32(lam) * tv.diffusion(x, d, scales))
        self.inverse = jax.jit(
            lambda d, damp, dd, lm: tv.inverse_diagonal(d, damp, dd, lam, lm, scales)
            if lam
            else 1 / jnp.maximum(dd + jnp.float32(lm) * damp, jnp.float32(1e-8))
        )

        weights = ops.batch_precision(precision, ops.valid)
        self.gram = _common.metric(weights)
        self.inverse_gram = jnp.linalg.pinv(self.gram)  # zero if no view is profiled
        self.data_moments = _common.moments(jnp.where(weights > 0, y, 0), weights)

        # At unchanged poses the prediction is linear in the image: the residual
        # plus a times the step's projection (returned scaled by sw, zero only
        # where unobserved) gives the profiled term exactly, without projecting;
        # the shared moments move by a times the step's B dx.
        def held(residual, step, sw, w, a, moments):
            w = ops.batch_precision(w, ops.valid)
            moved = residual + a * jnp.where(sw > 0, step / jnp.where(sw > 0, sw, 1), 0)
            return robust_value(profiled_residual(moved, w), w, delta) + 0.5 * jnp.vdot(
                moments, self.inverse_gram @ moments
            )

        self.held = jax.jit(held)

        @jax.jit
        def rhs(q, sw, j, sp, gx, free, precision, common):
            # Fixed voxels are discarded, so the gather may skip them.
            active = ops.projection.active(free)
            shared = common.image(common.inverse @ (common.poses @ sp))
            return (ops.pose_back(q, sw, j, sp, active, precision) + shared - gx) * free

        self.rhs = rhs
        self.norm = jax.jit(tv.dot)
        # Batch by batch: a whole-sinogram FP64 product would be a stack-sized temporary.
        self.energy = jax.jit(
            lambda r: jnp.sum(
                jax.lax.map(lambda b: tv.dot(b, b), ops.batches(r)), dtype=jnp.float64
            )
        )

        @jax.jit
        def schur(q, sw, j, factor, free, d, damp, lm, rhs, invdiag, precision, common, poses):
            # Fixed voxels' transpose is discarded: the gather may skip them.
            active = ops.projection.active(free)

            def matvec(v):
                v = v * free
                r, jr = ops.weighted_forward(q, v, sw, j, precision)
                bv = common.of_image(v)
                p = poses(prior.solve, factor, common, jr + common.poses.T @ (common.inverse @ bv))
                out = ops.weighted_back(q, r, sw, j, p, active, precision)
                out = out + common.image(common.inverse @ (bv - common.poses @ p))
                if lam:
                    out = out + jnp.float32(lam) * tv.diffusion(v, d, scales)
                out = out + (jnp.float32(lm) * damp) * v
                return out * free

            return krylov_solve(matvec, rhs, invdiag, inner, rtol=_CG_RTOL)

        self.schur = schur

    def image_penalty(self, x):
        """Image-prior objective, with no TV call or temporary when its weight is zero."""
        return float(self.lam * self.tv_value(x)) if self.lam else 0.0

    def initial_volume(self, q, lipschitz, iterations):
        """FISTA from zero on the Huber initialization data term with the shared term, without TV."""
        ops, y, precision, delta = self.ops, self.y, self.precision, self.delta
        images = ops.common_images(q, precision)
        inverse, data_moments = self.inverse_gram, self.data_moments
        # The shared term's Hessian B^T G^-1 B has rank two; its largest eigenvalue is G^-1 B B^T's.
        outer = jnp.einsum("i...,j...->ij", images.astype(jnp.float64), images.astype(jnp.float64))
        lipschitz = lipschitz + float(jnp.max(jnp.real(jnp.linalg.eigvals(inverse @ outer))))

        @jax.jit
        def run(x, q, y, precision, lipschitz):
            def fista(_, state):
                old, z, t = state
                shared = inverse @ (jnp.einsum("i...,...->i", images, z).astype(jnp.float64) - data_moments)
                gradient = ops.initial_gradient(q, z, y, precision, delta) + (
                    shared[0] * images[0] + shared[1] * images[1]
                ).astype(jnp.float32)
                new = jnp.maximum(
                    z - gradient / lipschitz, 0
                )
                tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
                return new, new + ((t - 1) / tn).astype(jnp.float32) * (new - old), tn

            return jax.lax.fori_loop(0, iterations, fista, (x, x, jnp.asarray(1.0, jnp.float64)))[0]

        return run(jnp.zeros(self.ops.shape, jnp.float32), ops.prepare(q), y, precision, lipschitz)

    def linearize(self, c, x):
        """The robust linearization at compact poses ``c`` and volume ``x``, with the shared term.

        Returns the gradient, diagonal, weights, columns, score, Gram blocks, data
        value, observation counts and residual, each including the views' shared
        offset and ramp (``_common.Common``, returned last).
        """
        q = self.ops.pad_poses(c)
        gx0, dd, sw, j, gr, gram, data, obstats, residual, moved = self.ops.linearize(
            q, x, self.y, self.precision, self.delta
        )
        common = _common.Common(
            self.gram,
            self.inverse_gram,
            self.ops.common_images(q, self.precision),
            moved,
            _common.moments(residual, self.ops.batch_precision(self.precision, self.ops.valid)),
        )
        gx0 = gx0 + common.image_gradient()
        dd = dd + common.diagonal()
        gr = gr + common.pose_score()
        return gx0, dd, sw, j, gr, gram, data + common.value(), obstats, residual, common

    def trial(self, x, c, dx, dc, pose_weight, held, a, beta):
        """Objective after volume step a and pose step a * beta, with the new state.

        ``held`` (residual, weighted projection of dx, weights, shared moments and
        their change B dx) gives beta = 0.
        """
        xt = x + jnp.float32(a) * dx
        cc = c + a * beta * dc
        if beta == 0:
            residual, step, sw, moments, change = held
            data = self.held(residual, step, sw, self.precision, jnp.float32(a), moments + a * change)
        else:
            q = self.ops.pad_poses(cc)
            value, moments = self.ops.objective(q, xt, self.y, self.precision, self.delta)
            data = value + 0.5 * jnp.vdot(moments, self.inverse_gram @ moments)
        total = float(data) + self.image_penalty(xt)
        return total + float(self.prior.value(cc, pose_weight)), xt, cc

    def _context(self, x, c, linear, pose_weight):
        """What every step from this linearisation shares, whatever its damping."""
        ops, prior = self.ops, self.prior
        gx0, dd, sw, j, gr, gram, residual, common = linear
        d = self.conduct(x) if self.lam else None
        gx = self.gradient(gx0, x, d) if self.lam else gx0
        wpose = prior.weights(c, pose_weight)
        pd = jnp.diagonal(gram, axis1=1, axis2=2)
        return _StepContext(
            q=ops.prepare(ops.pad_poses(c)),
            dd=dd,
            sw=sw,
            j=j,
            gram=gram,
            residual=residual,
            common=common,
            damp=jnp.maximum(dd, jnp.maximum(dd.max(), 1e-8) * 1e-4),
            d=d,
            gx=gx,
            wpose=wpose,
            gp=gr + prior.apply(c, wpose),
            free=(x > 0) | (gx < 0),
            pd=jnp.maximum(pd, jnp.maximum(pd.max(), 1e-8) * 0.01),
        )

    def _propose(self, x, k: _StepContext, lm) -> "_Proposal | None":
        """The Schur volume step at damping lm, clipped to x >= 0, and its pose back-substitution.

        Returns None when the volume solve does not give a finite residual.
        """
        ops, prior, common = self.ops, self.prior, k.common
        factor = prior.factor(k.gram + jax.vmap(jnp.diag)(lm * k.pd + 1e-10), k.wpose)
        poses = _common.PoseSolve.build(prior.solve, factor, common)
        sp = poses(prior.solve, factor, common, k.gp)
        rhs = self.rhs(k.q, k.sw, k.j, sp, k.gx, k.free, self.precision, common)
        invdiag = self.inverse(k.d, k.damp, k.dd, lm)
        linear_step = self.schur(
            k.q, k.sw, k.j, factor, k.free, k.d, k.damp, lm, rhs, invdiag, self.precision, common, poses
        )
        if not np.isfinite(float(linear_step.relative_residual)):
            return None
        dx = jnp.maximum(jnp.where(k.free, x + linear_step.solution, x), 0) - x
        r, jadx = ops.weighted_forward(k.q, dx, k.sw, k.j, self.precision)
        bdx = common.of_image(dx)
        jadx = jadx + common.poses.T @ (common.inverse @ bdx)
        dc = -poses(prior.solve, factor, common, k.gp + jadx)
        return _Proposal(dx, dc, r, jadx, bdx, linear_step)

    def proposal(self, x, c, linear, lm, pose_weight):
        """The RMS pose change one step from this linearisation proposes at damping lm, or None."""
        step = self._propose(x, self._context(x, c, linear, pose_weight), lm)
        if step is None:
            return None
        dp = self.prior.expand(step.dc)
        return float(jnp.sqrt(jnp.mean(jnp.sum(dp**2, axis=1))))

    def step(self, x, c, linear, lm, pose_weight, f):
        """One damped step: Schur volume solve, exact pose back-substitution, line search.

        Raises the damping tenfold, up to five times, until the line search accepts.
        Returns the accepted step with its predicted reduction, or None.
        """
        prior = self.prior
        k = self._context(x, c, linear, pose_weight)
        common = k.common
        for _ in range(5):
            proposed = self._propose(x, k, lm)
            if proposed is None:
                lm *= 10
                continue
            dx, dc, r, jadx, bdx = proposed.dx, proposed.dc, proposed.r, proposed.jadx, proposed.bdx
            linear_step = proposed.linear_step
            cg_steps, cg_residual = linear_step.iterations, linear_step.relative_residual
            gx_dx, gp_dc = float(self.norm(k.gx, dx)), float(jnp.vdot(k.gp, dc))
            accepted = None
            if gx_dx + gp_dc < 0:
                held = (k.residual, r, k.sw, common.moments, bdx)
                trial = partial(self.trial, x, c, dx, dc, pose_weight, held)
                accepted = _line_search(trial, f, gx_dx, gp_dc)
            if accepted:
                a, beta, new = accepted
                dp = prior.expand(dc)
                curvature = float(self.energy(r))
                if self.lam:
                    curvature += self.lam * float(self.norm(dx, self.diffuse(dx, k.d)))
                curvature += beta * 2 * float(jnp.vdot(dc, jadx))
                pose = jnp.einsum("ni,nij,nj->", dp, k.gram, dp) + jnp.vdot(
                    dc, prior.apply(dc, k.wpose)
                )
                curvature += beta**2 * float(pose)
                # The shared rows; their cross term is already in jadx.
                curvature += 2 * float(common.value(bdx) + beta**2 * common.value(common.poses @ dc))
                predicted = -a * (gx_dx + beta * gp_dc) - 0.5 * a * a * curvature
                change = float(jnp.sqrt(jnp.mean(jnp.sum((a * beta * dp) ** 2, axis=1))))
                explicit = bool(linear_step.residual_is_explicit)
                linear_record = {
                    "proposed_pose_step_rms": float(jnp.sqrt(jnp.mean(jnp.sum(dp**2, axis=1)))),
                    "cg_residual_kind": "explicit" if explicit else "recursive",
                    "cg_status": "converged"
                    if explicit and float(cg_residual) <= _CG_RTOL
                    else "budget_exhausted",
                    "cg_matvecs": int(linear_step.matvecs),
                }
                return (
                    a,
                    beta,
                    new,
                    predicted,
                    change,
                    lm,
                    int(cg_steps),
                    float(cg_residual),
                    linear_record,
                )
            lm *= 10
        return None

def _undamped_proposal(level, step, x, c, linear, budget, pose_weight):
    """The step's pose proposal at the initial damping when its damped one is small, else None.

    Alignment has converged when its proposed geometry updates are small, even
    if more image iterations could still reduce the residual. A proposal damped
    harder than the initial damping is no such evidence: damping can suppress
    any step. So a small damped proposal is checked again at the initial
    damping, from the same linearisation, and that proposal decides. The
    damping itself does not: clipping the image step at x >= 0 can keep it high
    long after the poses have settled.
    """
    if step["proposed_pose_step_rms"] >= _POSE_TOL_PX:
        return None
    if step["lm"] <= budget.lm_initial:
        return step["proposed_pose_step_rms"]
    return level.proposal(x, c, linear, budget.lm_initial, pose_weight)


def solve(y, precision, prior, weight, eps, *, ops, budget, x0, p0, pose_sigma, delta, scales):
    """Fit the volume and the per-view pose parameters at one level.

    Returns the volume, compact free parameters, and a record of the steps.
    """
    start, n = time.perf_counter(), ops.n
    width = prior.size
    c = (
        jnp.zeros(width, jnp.float64)
        if p0 is None
        else prior.project_gauge(jnp.asarray(p0, jnp.float64).reshape(-1))
    )
    y = ops.pad_images(y)
    precision = ops.pad_precision(precision)
    level = _Level(
        ops,
        prior,
        y,
        precision,
        lam=weight,
        eps=eps,
        scales=scales,
        delta=delta,
        inner=budget.inner,
    )
    lipschitz, observed = ops.sensitivity(ops.pad_poses(c))
    if x0 is None:
        x = level.initial_volume(ops.pad_poses(c), lipschitz, budget.init_iterations)
    else:
        x = jnp.maximum(jnp.asarray(x0, jnp.float32), 0)
    # Prolongation can populate voxels invisible to every ray at this level.
    # Start those undetermined values at zero, but let later pose/image steps
    # expose or populate them; this is not a permanent active-set constraint.
    x = jnp.where(observed, x, 0)
    del observed
    gx0, dd, sw, j, gr, gram, data, obstats, residual, common = level.linearize(c, x)
    fisher = jnp.trace(gram, axis1=1, axis2=2)
    pose_weight = float(pose_sigma * jnp.sqrt(jnp.median(fisher)) * np.sqrt(2 * np.log(n)))
    f = float(data + level.image_penalty(x) + prior.value(c, pose_weight))
    initial_objective = f
    history, lm, stable, status = [], budget.lm_initial, 0, "step_budget_reached"
    for it in range(budget.outer):
        if it:
            # Release the previous weights and columns first: two stack-sized sets
            # would not fit beside each other on a small device.
            del gx0, dd, sw, j, gr, gram, residual, common
            gx0, dd, sw, j, gr, gram, data, obstats, residual, common = level.linearize(c, x)
        linear = (gx0, dd, sw, j, gr, gram, residual, common)
        result = level.step(x, c, linear, lm, pose_weight, f)
        if result is None:
            status = "line_search_stalled"
            break
        x0, c0 = x, c
        a, beta, (fn, x, c), predicted, change, lm, cg_steps, cg_residual, linear_record = result
        rho = _gain_ratio(f - fn, predicted)
        valid, down = map(float, np.asarray(obstats))
        history.append(
            {
                "objective": fn,
                "pose_step_rms": change,
                "line_step": a,
                "pose_fraction": beta,
                "lm": lm,
                "gain_ratio": rho,
                "predicted_reduction": predicted,
                "cg_iterations": cg_steps,
                "cg_relative_residual": cg_residual,
                **linear_record,
                "fraction_downweighted": down / max(valid, 1),
            }
        )
        f = fn
        if rho > 0.75 and a == 1:
            lm = max(_LM_MIN, lm / 2)
        elif rho < 0.25 or a < 0.5:
            lm = min(lm * 3, 1e6)
        undamped = _undamped_proposal(level, history[-1], x0, c0, linear, budget, pose_weight)
        history[-1]["undamped_proposed_pose_step_rms"] = undamped
        stable = stable + 1 if undamped is not None and undamped < _POSE_TOL_PX else 0
        if stable >= 2 and it + 1 >= _MIN_OUTER:
            status = "pose_steps_below_tolerance"
            break
    final_prior = float(prior.value(c, pose_weight))
    final_image_prior = level.image_penalty(x)
    record = {
        "status": status,
        "seconds": time.perf_counter() - start,
        "steps": history,
        "tv_weight": float(weight),
        "noise_sigma": float(pose_sigma),
        "pose_prior_weight": float(pose_weight),
        "initial_objective": initial_objective,
        "final_objective": f,
        "final_data_objective": f - final_prior - final_image_prior,
        "final_pose_prior": final_prior,
        "final_image_prior": final_image_prior,
        "budget": asdict(budget),
    }
    return x, c, level.y[:n], record
