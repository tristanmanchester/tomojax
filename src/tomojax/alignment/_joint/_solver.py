"""Finite-budget joint Gauss-Newton on the volume and the per-view poses.

Each outer step linearizes the robust data term at the current volume and
poses, freezes the IRLS weights and the lagged TV conductance, and eliminates
the poses exactly: the volume step solves the Schur system by preconditioned CG
(Jacobi: data diagonal + damping + TV diagonal), and the pose step follows from
it through the global prior solve. A backtracking line search on the true
objective chooses the volume step length and the fraction of the pose step.
Only scalar decisions leave the device.
"""

from dataclasses import asdict, dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import _tv as tv
from ._krylov import solve as krylov_solve


@dataclass(frozen=True)
class Budget:
    """Internal work limits for one level; convergence can stop sooner."""

    outer: int
    inner: int
    init_iterations: int = 5
    lm_initial: float = 0.001


_CG_RTOL = 0.03
_LM_MIN = 1e-8
_POSE_TOL_PX = 0.003
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
        self.rhs = jax.jit(lambda q, sw, j, sp, gx, free: (ops.pose_back(q, sw, j, sp) - gx) * free)
        self.norm = jax.jit(tv.dot)
        # Batch by batch: a whole-sinogram FP64 product would be a stack-sized temporary.
        self.energy = jax.jit(
            lambda r: jnp.sum(
                jax.lax.map(lambda b: tv.dot(b, b), ops.batches(r)), dtype=jnp.float64
            )
        )

        @jax.jit
        def schur(q, sw, j, factor, free, d, damp, lm, rhs, invdiag):
            def matvec(v):
                v = v * free
                r, jr = ops.weighted_forward(q, v, sw, j)
                out = ops.weighted_back(q, r, sw, j, prior.solve(factor, jr))
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
        """FISTA from zero on the Huber initialization data term, without TV."""
        ops, y, precision, delta = self.ops, self.y, self.precision, self.delta

        @jax.jit
        def run(x, q, y, precision, lipschitz):
            def fista(_, state):
                old, z, t = state
                new = jnp.maximum(
                    z - ops.initial_gradient(q, z, y, precision, delta) / lipschitz, 0
                )
                tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
                return new, new + ((t - 1) / tn).astype(jnp.float32) * (new - old), tn

            return jax.lax.fori_loop(0, iterations, fista, (x, x, jnp.asarray(1.0, jnp.float64)))[0]

        return run(jnp.zeros(self.ops.shape, jnp.float32), ops.prepare(q), y, precision, lipschitz)

    def trial(self, x, c, dx, dc, pose_weight, a, beta):
        """Objective after volume step a and pose step a * beta, with the new state."""
        xt = x + jnp.float32(a) * dx
        cc = c + a * beta * dc
        total = float(
            self.ops.objective(self.ops.pad_poses(cc), xt, self.y, self.precision, self.delta)
        )
        total += self.image_penalty(xt)
        return total + float(self.prior.value(cc, pose_weight)), xt, cc

    def step(self, x, c, linear, lm, pose_weight, f):
        """One damped step: Schur volume solve, exact pose back-substitution, line search.

        Raises the damping tenfold, up to five times, until the line search accepts.
        Returns the accepted step with its predicted reduction, or None.
        """
        ops, prior = self.ops, self.prior
        q = ops.prepare(ops.pad_poses(c))
        gx0, dd, sw, j, gr, gram = linear
        damp = jnp.maximum(dd, jnp.maximum(dd.max(), 1e-8) * 1e-4)
        d = self.conduct(x) if self.lam else None
        gx = self.gradient(gx0, x, d) if self.lam else gx0
        wpose = prior.weights(c, pose_weight)
        gp = gr + prior.apply(c, wpose)
        free = (x > 0) | (gx < 0)
        pd = jnp.diagonal(gram, axis1=1, axis2=2)
        pd = jnp.maximum(pd, jnp.maximum(pd.max(), 1e-8) * 0.01)
        for _ in range(5):
            factor = prior.factor(gram + jax.vmap(jnp.diag)(lm * pd + 1e-10), wpose)
            rhs = self.rhs(q, sw, j, prior.solve(factor, gp), gx, free)
            invdiag = self.inverse(d, damp, dd, lm)
            linear_step = self.schur(q, sw, j, factor, free, d, damp, lm, rhs, invdiag)
            dx, cg_steps, cg_residual = (
                linear_step.solution,
                linear_step.iterations,
                linear_step.relative_residual,
            )
            if not np.isfinite(float(cg_residual)):
                lm *= 10
                continue
            dx = jnp.maximum(jnp.where(free, x + dx, x), 0) - x
            r, jadx = ops.weighted_forward(q, dx, sw, j)
            dc = -prior.solve(factor, gp + jadx)
            gx_dx, gp_dc = float(self.norm(gx, dx)), float(jnp.vdot(gp, dc))
            accepted = None
            if gx_dx + gp_dc < 0:
                accepted = _line_search(
                    partial(self.trial, x, c, dx, dc, pose_weight), f, gx_dx, gp_dc
                )
            if accepted:
                a, beta, new = accepted
                dp = prior.expand(dc)
                curvature = float(self.energy(r))
                if self.lam:
                    curvature += self.lam * float(self.norm(dx, self.diffuse(dx, d)))
                curvature += beta * 2 * float(jnp.vdot(dc, jadx))
                pose = jnp.einsum("ni,nij,nj->", dp, gram, dp) + jnp.vdot(
                    dc, prior.apply(dc, wpose)
                )
                curvature += beta**2 * float(pose)
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
    gx0, dd, sw, j, gr, gram, data, obstats = ops.linearize(
        ops.pad_poses(c), x, y, precision, delta
    )
    fisher = jnp.trace(gram, axis1=1, axis2=2)
    pose_weight = float(pose_sigma * jnp.sqrt(jnp.median(fisher)) * np.sqrt(2 * np.log(n)))
    f = float(data + level.image_penalty(x) + prior.value(c, pose_weight))
    initial_objective = f
    history, lm, stable, status = [], budget.lm_initial, 0, "budget_exhausted"
    for it in range(budget.outer):
        if it:
            # Release the previous weights and columns first: two stack-sized sets
            # would not fit beside each other on a small device.
            del gx0, dd, sw, j, gr, gram
            gx0, dd, sw, j, gr, gram, data, obstats = ops.linearize(
                ops.pad_poses(c), x, y, precision, delta
            )
        result = level.step(x, c, (gx0, dd, sw, j, gr, gram), lm, pose_weight, f)
        if result is None:
            status = "line_search_stalled"
            break
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
        # Alignment has converged when its proposed geometry updates are small,
        # even if more image iterations could still reduce the residual. Neither
        # clipping a large proposal nor raising damping to suppress it provides
        # evidence of convergence.
        proposed_change = linear_record["proposed_pose_step_rms"]
        small = proposed_change < _POSE_TOL_PX and history[-1]["lm"] <= budget.lm_initial
        stable = stable + 1 if small else 0
        if stable >= 2 and it + 1 >= _MIN_OUTER:
            status = "converged"
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
    return x, c, record
