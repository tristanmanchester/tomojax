"""Edge-based refinement of the per-view poses after the joint fit.

The volume is fixed. The data term is the robust loss of H(A(p) x - y), where
H removes each detector row's weighted mean and then a normalized Gaussian
local mean (8 native pixels): it keeps edges, which locate a view, and drops
smooth photometric changes, which do not. Only the part of the correction
orthogonal to the prior's spline space and the common gauge may change, so the
broad trajectory of the joint fit is kept and every view's residual motion,
steps included, stays free. A continuous per-view anchor trusts the joint fit
where a view's non-geometric residual energy is ordinary for its neighbors,
and lets an inconsistent view move.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.alignment._config import AlignConfig
from tomojax.core import checkpoint
from tomojax.core.robust import MAD_TO_SD

from ._operators import column_gram, column_score, robust_batch, robust_value
from ._stacks import Stack, level_stacks, temporaries, transient_reserve, unplaced

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._operators import AffineGeometry, IndexedViewBatch, Operators
    from ._pose import MotionPrior
    from ._solver import LevelRecord
    from ._stacks import ArrayStack, Place, StorageRecord

type _EdgeInputs = tuple[
    jax.Array | AffineGeometry,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array,
    jax.Array | None,
]
type _Stats = tuple[jax.Array, jax.Array, jax.Array]
type _RefinementStep = tuple[jax.Array, float, float, jax.Array, float]
type _StepStatus = Literal["stationary", "pose_steps_below_tolerance", "line_search_stalled"]


class _RefinementRecord(TypedDict):
    status: str
    steps: list[dict[str, float]]


class _RefinementSettings(TypedDict):
    iterations: int
    neighbors: int
    loss_width: NotRequired[float]


class RefinementRecord(_RefinementRecord, total=False):
    delta: float
    seconds: float
    stacks: StorageRecord


# A smaller proposed step is converged: it cannot change the pose error, and
# its decrease is below what the FP32 objective resolves, so backtracking on it
# only spends objective evaluations.
_FP32_STEP_PX = 1e-4
_LM_INITIAL = 0.01


def _kernel(sigma: float) -> list[float]:
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / float(sigma)) ** 2)
    return np.asarray(k / k.sum(), dtype=np.float32).tolist()


class EdgeFilter:
    """Masked linear H = (local-mean removal) after (row-mean removal), and H^T.

    Leading dimensions are free; precision broadcasts to them. Normalizing by
    the smoothed precision avoids artificial edges at masks and borders.
    """

    def __init__(self, ratios: tuple[float, float], smoothing: float = 8.0) -> None:
        """``smoothing`` is the local mean's width in native pixels; ``ratios`` scale it."""
        self.ku = _kernel(smoothing * ratios[0])
        self.kv = _kernel(smoothing * ratios[1])

    def smooth(self, a: jax.Array) -> jax.Array:
        """Zero-extended separable Gaussian; symmetric, so its own transpose."""
        for kernel, axis in ((self.ku, -1), (self.kv, -2)):
            radius = len(kernel) // 2
            width = a.shape[axis]
            reach = min(radius, width - 1)
            padding = [(0, 0)] * a.ndim
            padding[axis] = (reach, reach)
            padded = jnp.pad(a, padding)
            a = jax.lax.full_like(a, 0)
            for start, weight in enumerate(kernel[radius - reach : radius + reach + 1]):
                neighbor = jax.lax.slice_in_dim(padded, start, start + width, axis=axis)
                a = a + neighbor * weight
        return a

    def forward(self, a: jax.Array, precision: jax.Array | np.ndarray) -> jax.Array:
        """H a."""
        p = jnp.asarray(precision, jnp.float32)
        m = p > 0
        a = jnp.where(m, a, 0).astype(jnp.float32)
        rows = jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
        a = jnp.where(m, a - jnp.sum(p * a, axis=-1, keepdims=True) / rows, 0)
        den = jnp.maximum(self.smooth(p), jnp.finfo(jnp.float32).tiny)
        return jnp.where(m, a - self.smooth(p * a) / den, 0)

    def transpose(self, z: jax.Array, precision: jax.Array | np.ndarray) -> jax.Array:
        """H^T z."""
        p = jnp.asarray(precision, jnp.float32)
        m = p > 0
        z = jnp.where(m, z, 0).astype(jnp.float32)
        den = jnp.maximum(self.smooth(p), jnp.finfo(jnp.float32).tiny)
        q = jnp.where(m, z - p * self.smooth(z / den), 0)
        rows = jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
        return jnp.where(m, q - p * jnp.sum(q, axis=-1, keepdims=True) / rows, 0)


def relative_precision(
    energy: jax.Array, sigma: float | jax.Array, neighbors: int = 5
) -> jax.Array:
    """2 (B + s^2) / (B + E + 2 s^2), B the median energy of the neighboring views.

    ``neighbors`` views on each side make B.
    """
    n = len(energy)
    shifted = []
    for offset in range(-neighbors, neighbors + 1):
        if offset:
            index = jnp.arange(n) + offset
            inside = (index >= 0) & (index < n)
            shifted.append(jnp.where(inside, energy[jnp.clip(index, 0, n - 1)], jnp.nan))
    reference = jnp.maximum(jnp.nanmedian(jnp.stack(shifted), axis=0), 0)
    variance = jnp.maximum(jnp.asarray(sigma) ** 2, 1e-20)
    return 2 * (reference + variance) / (reference + jnp.maximum(energy, 0) + 2 * variance)


class _EdgeObjective:
    """Batch-wise filtered residuals and columns; only per-view summaries survive."""

    def __init__(self, ops: Operators, smoothing: float) -> None:
        self.ops = ops
        self.filter = EdgeFilter(ops.ratios, smoothing)
        self.stats = jax.jit(self._stats)
        self.value = jax.jit(self._value)
        self.consistency = jax.jit(self._consistency)
        self.scale = jax.jit(self._scale)

    def _batches(self, q: jax.Array) -> IndexedViewBatch:
        return (jnp.arange(self.ops.nb), *self.ops.views(q))

    def _z(self, z: IndexedViewBatch, y: ArrayStack, precision: ArrayStack | None) -> _EdgeInputs:
        """One batch's geometry, data and precision, read from the level's stacks."""
        i = z[0]
        return (*z[1:], y.read(i), self.ops.batch_of(precision, i))

    def _residual(self, z: _EdgeInputs, x: jax.Array) -> jax.Array:
        nominal, qb, valid, scales, frames, yy, w = z
        w = self.ops.batch_precision(w, valid)
        return self.filter.forward(
            self.ops.project(self.ops.coefficients(nominal, qb, valid, scales, frames), x) - yy,
            w,
        )

    def _stats(
        self,
        q: jax.Array,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
    ) -> _Stats:
        o, h = self.ops, self.filter

        def step(z: IndexedViewBatch) -> _Stats:
            nominal, qb, valid, scales, frames, yy, w = self._z(z, y, precision)
            w = o.batch_precision(w, valid)
            raw = o.columns(nominal, qb, valid, x, scales=scales, frames=frames)
            r = h.forward(raw[:, 0] - yy, w)
            res, _, jw = robust_batch(r, h.forward(raw[:, 1:], w[:, None]), w, delta)
            return column_score(jw, res), column_gram(jw), robust_value(r, w, delta)

        g, gram, energy = jax.lax.map(step, self._batches(q))
        n, k = o.n, g.shape[-1]
        return (
            o.parameters.reduce_score(g.reshape(o.padded_n, k)[:n]),
            gram.reshape(o.padded_n, k, k)[:n],
            jnp.sum(energy),
        )

    def _value(
        self,
        q: jax.Array,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        delta: float | jax.Array,
    ) -> jax.Array:
        def step(total: jax.Array, z: IndexedViewBatch) -> tuple[jax.Array, None]:
            inputs = self._z(z, y, precision)
            w = self.ops.batch_precision(inputs[-1], inputs[2])
            return total + robust_value(self._residual(inputs, x), w, delta), None

        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), self._batches(q))[0]

    def _consistency(
        self, q: jax.Array, x: jax.Array, y: ArrayStack, precision: ArrayStack | None
    ) -> jax.Array:
        """Per-view mean energy of the smooth residual left after the best pose step."""
        o, h = self.ops, self.filter

        def step(z: IndexedViewBatch) -> jax.Array:
            nominal, qb, valid, scales, frames, yy, w = self._z(z, y, precision)
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

        return jax.lax.map(step, self._batches(q)).reshape(-1)[: o.n]

    def _scale(
        self, q: jax.Array, x: jax.Array, y: ArrayStack, precision: ArrayStack | None
    ) -> jax.Array:
        """Median over views of each view's MAD of the filtered residual."""

        def step(z: IndexedViewBatch) -> jax.Array:
            inputs = self._z(z, y, precision)
            w = self.ops.batch_precision(inputs[-1], inputs[2])
            r = jnp.where(w > 0, self._residual(inputs, x), jnp.nan)
            med = jnp.nanmedian(r, axis=(-1, -2), keepdims=True)
            return MAD_TO_SD * jnp.nanmedian(jnp.abs(r - med), axis=(-1, -2))

        return jnp.nanmedian(jax.lax.map(step, self._batches(q)).reshape(-1)[: self.ops.n])


class _Refinement:
    """The fixed-volume edge objective, its anchored prior and one damped step."""

    def __init__(
        self,
        x: jax.Array,
        y: ArrayStack,
        precision: ArrayStack | None,
        prior: MotionPrior,
        objective: _EdgeObjective,
        ops: Operators,
        sigma: float,
        c0: jax.Array,
        *,
        loss_width: float,
        neighbors: int,
    ) -> None:
        self.x, self.y, self.precision = x, y, precision
        self.prior, self.ops, self.c0 = prior, ops, c0
        self.objective = objective
        q = ops.pad_poses(c0)
        model_sigma = float(self.objective.scale(q, x, y, precision))
        if not np.isfinite(model_sigma):
            raise ValueError("no finite residual scale for the pose refinement")
        self.delta = max(loss_width * float(sigma), loss_width * model_sigma, 1e-8)
        self.first = self.objective.stats(q, x, y, precision, self.delta)
        gram = self.first[1]
        fisher = jnp.median(jnp.trace(gram, axis1=1, axis2=2))
        self.photon_weight = float(sigma * jnp.sqrt(fisher) * np.sqrt(2 * np.log(ops.n)))
        self.weight: float | tuple[float, float] = self.photon_weight
        energy = self.objective.consistency(q, x, y, precision)
        self.anchor = relative_precision(energy, float(sigma), neighbors)[:, None, None] * gram

    def penalty(self, c: jax.Array) -> jax.Array:
        """Motion prior plus the anchor towards the joint fit's poses."""
        d = self.prior.expand(c - self.c0)
        return self.prior.value(c, self.weight) + 0.5 * jnp.einsum("vi,vij,vj->", d, self.anchor, d)

    def value(self, c: jax.Array) -> float:
        """The objective at pose parameters ``c``."""
        q = self.ops.pad_poses(c)
        data = self.objective.value(q, self.x, self.y, self.precision, self.delta)
        return float(data + self.penalty(c))

    def step(
        self, c: jax.Array, stats: _Stats, f: float, lm: float
    ) -> _RefinementStep | _StepStatus:
        """Constrained Gauss-Newton with Armijo backtracking; damping x10 up to four times.

        A proposal no trial along which lowers the objective, and whose predicted decrease is
        below the objective's FP32 rounding, has converged: the step says so rather than
        raising the damping.

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
        for attempt in range(4):
            damped = gram + anchor + jax.vmap(jnp.diag)(lm * pd + 1e-10)
            dc = -prior.innovation_solve(damped, wpose, gp)
            slope = float(jnp.vdot(gp, dc))
            if not np.isfinite(slope):
                raise FloatingPointError("non-finite pose refinement slope")
            if slope >= 0:
                return "stationary"
            # Raised damping shrinks a proposal without evidence of convergence.
            if not attempt and lm <= _LM_INITIAL and _rms(prior.expand(dc)) < _FP32_STEP_PX:
                return "pose_steps_below_tolerance"
            for ls in range(8):
                alpha = 0.5**ls
                fn = self.value(c + alpha * dc)
                if fn <= f + 1e-4 * alpha * slope:
                    return c + alpha * dc, fn, alpha, dc, lm
            # No point along the proposal lowers the objective, and the most it promises is below
            # the objective's FP32 rounding: the poses have converged, and more damping would only
            # promise less.
            if -slope <= np.finfo(np.float32).eps * abs(f):
                return "pose_steps_below_tolerance"
            lm *= 10
        return "line_search_stalled"


def _rms(steps: jax.Array) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.sum(steps**2, axis=1))))


def refine(
    x: jax.Array,
    p: jax.Array | np.ndarray,
    y: np.ndarray | jax.Array,
    precision: np.ndarray | jax.Array | None,
    prior: MotionPrior,
    *,
    ops: Operators,
    sigma: float,
    last: LevelRecord | None = None,
    placement: Mapping[str, Place] | None = None,
    config: AlignConfig | None = None,
) -> tuple[jax.Array | np.ndarray, RefinementRecord, tuple[np.ndarray, np.ndarray]]:
    """Refine local innovations; the jointly fitted global values stay fixed.

    ``config`` (an ``AlignConfig``; None: its defaults) gives the steps at most
    (``refine_iterations``, 0 skipping them), the edge filter's width
    (``refine_smoothing``), the views setting a view's expected residual
    (``refine_neighbors``) and the robust loss's width (``loss_width``).

    The translations' prior weight takes the ``last`` level's error scale (its
    ``translation_noise`` in place of the photon standard error, with its ``pose_fisher``):
    the same multiplier on this stage's photon weight, written so that sigma cancels.
    Rotation increments keep the photon weight. The stage is a level for the device:
    its data and precision are placed and allocated first (``placement`` as for a
    level) and released at its end. Returns the poses, the record, and each view's
    offset and ramp and residual energy at the result (``Operators.view_summary``).
    """
    start = time.perf_counter()
    config = AlignConfig() if config is None else config
    iterations = config.refine_iterations
    objective = _EdgeObjective(ops, config.refine_smoothing)
    transients = _transients(objective, x, prior, ops, precision)
    reserve = transient_reserve(transients)
    names = ("y", "precision")
    stacks = level_stacks(ops, y, precision, reserve=reserve, names=names, placement=placement)
    try:
        data, weights = stacks["y"], stacks.get("precision")
        record: RefinementRecord
        if prior.nlocal == 0 or iterations == 0:
            status = "no_local_parameters" if prior.nlocal == 0 else "off"
            c = p
            record = {"status": status, "steps": []}
        else:
            settings: _RefinementSettings = {
                "iterations": iterations,
                "neighbors": config.refine_neighbors,
            }
            settings["loss_width"] = config.loss_width
            c, record = _refine(
                x, p, data, weights, prior, objective, ops=ops, sigma=sigma, last=last, **settings
            )
        summary = ops.view_summary(ops.pad_poses(c), x, data, weights)
        offsets, energy = (np.asarray(a) for a in summary)
    except jax.errors.JaxRuntimeError as err:
        if progress.handler_failed(err):
            raise
        raise unplaced(err, stacks) from err
    finally:
        stacks.close()
    stacks_record: StorageRecord = {**stacks.record(), "transients": transients}
    record = {**record, "seconds": time.perf_counter() - start, "stacks": stacks_record}
    return c, record, (offsets, energy)


def _transients(
    objective: _EdgeObjective,
    x: jax.Array,
    prior: MotionPrior,
    ops: Operators,
    precision: np.ndarray | jax.Array | None,
) -> dict[str, int]:
    """The temporary bytes of the stage's compiled functions, from their compiles here.

    Its stacks are resident in the compile; the calls later reuse it.
    """
    specs = ops.stack_specs(precision)
    weights = None if precision is None else Stack.shaped(specs["precision"], ops.batch)
    q = jax.eval_shape(ops.pad_poses, jax.ShapeDtypeStruct((prior.size,), jnp.float64))
    data = (q, x, Stack.shaped(specs["y"], ops.batch), weights)
    out = {
        "stats": temporaries(objective.stats, *data, 1.0),
        "value": temporaries(objective.value, *data, 1.0),
        "consistency": temporaries(objective.consistency, *data),
        "scale": temporaries(objective.scale, *data),
        "summary": temporaries(ops.view_summary, *data),
    }
    if prior.nlocal:
        out["innovation_solve"] = prior.temporaries("innovation_solve")
    return out


class _Refined(NamedTuple):
    """What a checkpoint keeps of the refinement besides its poses."""

    objective: float
    lm: float
    steps: list[dict[str, float]]


def _refine(
    x: jax.Array,
    p: jax.Array | np.ndarray,
    y: ArrayStack,
    precision: ArrayStack | None,
    prior: MotionPrior,
    objective: _EdgeObjective,
    *,
    ops: Operators,
    sigma: float,
    last: LevelRecord | None,
    iterations: int,
    neighbors: int,
    loss_width: float,
) -> tuple[jax.Array, RefinementRecord]:
    c = prior.project_gauge(jnp.asarray(p, jnp.float64).reshape(-1))
    problem = _Refinement(
        x, y, precision, prior, objective, ops, sigma, c, loss_width=loss_width, neighbors=neighbors
    )
    if last is not None and last["translation_noise"] is not None:
        fisher = float(jnp.median(jnp.trace(problem.first[1], axis1=1, axis2=2)))
        scale = float(np.sqrt(fisher * last["pose_fisher"] * 2 * np.log(ops.n)))
        problem.weight = (last["translation_noise"] * scale, problem.photon_weight)
    stats, status = problem.first, "step_budget_reached"
    with checkpoint.loop("refine") as saves:
        if saves.saved is None:
            f, lm, steps = float(stats[2] + problem.penalty(c)), _LM_INITIAL, []
        else:
            c = jnp.asarray(saves.saved.arrays[0])
            f, lm, steps = saves.saved.scalars(_Refined)
        for it in range(len(steps), iterations):
            progress.check_canceled()
            if it:
                stats = problem.objective.stats(ops.pad_poses(c), x, y, precision, problem.delta)
            result = problem.step(c, stats, f, lm)
            if isinstance(result, str):
                status = result
                break
            c, f, alpha, dc, lm = result
            step = _rms(alpha * prior.expand(dc))
            steps.append({"objective": f, "pose_step_rms": step, "line_step": alpha, "lm": lm})
            if (task := progress.for_unit("updates")) is not None:
                jax.block_until_ready(c)
                task.advance(len(steps), {})
                progress.preview(lambda c=c: ops.preview(x, c))
            lm = max(1e-5, lm * 0.5)
            if saves.due():
                saves.save((c,), _Refined(f, lm, steps))
    if steps:
        progress.preview(lambda: ops.preview(x, c), final=True)
    return c, {"status": status, "steps": steps, "delta": problem.delta}
