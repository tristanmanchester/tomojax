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

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import time
from typing import TYPE_CHECKING, Concatenate, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.core.registration import planes
from tomojax.wire.metrics import Metric

from . import _common, _tv as tv
from ._krylov import solve as krylov_solve
from ._operators import profiled_residual, robust_value
from ._stacks import Stack, level_stacks, temporaries, transient_reserve, unplaced

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tomojax.alignment._config import AlignConfig

    from ._krylov import KrylovResult
    from ._operators import Coefficients, Operators
    from ._pose import FactorShape, MotionPrior, PoseFactor, PoseWeight
    from ._stacks import ArrayStack, LevelStacks, Place, StackSpec, StorageRecord, VolumeName

type _StepMetric = float | int | str | None


class LevelSummary(TypedDict):
    status: str
    steps: list[dict[str, _StepMetric]]


class _DescentRecord(LevelSummary):
    tv_weight: float
    noise_sigma: float
    pose_prior_weight: float | list[float]
    translation_noise: float | None
    pose_fisher: float
    initial_objective: float
    final_objective: float
    final_data_objective: float
    final_pose_prior: float
    final_image_prior: float
    iterations: int
    image_iterations: int
    initial_image_iterations: int


class LevelRecord(_DescentRecord):
    seconds: float
    stacks: StorageRecord


type _Linearization = tuple[
    jax.Array, jax.Array, ArrayStack, ArrayStack, jax.Array, jax.Array, ArrayStack, _common.Common
]
type _Trial = tuple[float, jax.Array, jax.Array]


class _AcceptedStep(NamedTuple):
    line_step: float
    pose_fraction: float
    trial: _Trial
    predicted: float
    pose_step_rms: float
    lm: float
    cg_steps: int
    cg_residual: float
    proposed_rms: float
    linear_record: dict[str, float | int | str]


type _FistaState = tuple[jax.Array, jax.Array, jax.Array]
type _StartOperands = tuple[
    Coefficients, ArrayStack, ArrayStack | None, float | jax.Array, jax.Array, jax.Array, jax.Array
]
type _SchurOperands = tuple[
    Coefficients,
    ArrayStack,
    ArrayStack,
    PoseFactor,
    jax.Array,
    jax.Array | None,
    tuple[jax.Array, jax.Array],
    float,
    jax.Array,
    jax.Array,
    ArrayStack | None,
    _common.Common,
    _common.PoseSolve,
]
type _Shape = jax.ShapeDtypeStruct
type _SchurShapes = tuple[
    tuple[_Shape, _Shape],
    Stack[_Shape],
    Stack[_Shape],
    FactorShape,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct | None,
    tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
    float,
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
    Stack[_Shape] | None,
    tuple[_Shape, _Shape, _Shape, _Shape, _Shape],
    tuple[_Shape, _Shape],
    Stack[_Shape],
]


@dataclass(frozen=True)
class Budget:
    """One level's work limits and stopping rule (``AlignConfig``'s words)."""

    iterations: int
    image_iterations: int
    initial_image_iterations: int = 5
    lm_initial: float = 0.001
    image_tolerance: float = 0.03
    tolerance: float = 0.01
    min_iterations: int = 4

    @classmethod
    def of(cls, config: AlignConfig, iterations: int) -> Budget:
        """A level of ``iterations`` Gauss-Newton iterations under ``AlignConfig`` ``config``."""
        return cls(
            iterations,
            config.image_iterations,
            config.initial_image_iterations,
            image_tolerance=config.image_tolerance,
            tolerance=config.tolerance,
            min_iterations=config.min_iterations,
        )


_LM_MIN = 1e-8

# The level writes each volume it holds into a buffer it owns (``level_volumes``), donated
# as the first argument; one the program does not read must still be kept to receive the
# output. A product is held behind a barrier where the uncompiled code made it on its own:
# the CPU compiler would fuse it into the sum that follows (one rounding in place of two).
_barrier = jax.lax.optimization_barrier


def _into[**P, R](fun: Callable[P, R]) -> Callable[P, R]:
    return jax.jit(fun, donate_argnums=(0,), keep_unused=True)


def _by_slice(
    update: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """An elementwise ``update(v, b, m)`` by the shared images b, into v a z-slice at a time.

    The same values as made whole, with a slice of FP64 terms at a time: whole, they
    were the largest temporaries of the level after the prior's (7 volumes).
    """

    def run(v: jax.Array, b: jax.Array, m: jax.Array) -> jax.Array:
        def one(k: jax.Array, v: jax.Array) -> jax.Array:
            vk, bk = (jax.lax.dynamic_slice_in_dim(a, k, 1, axis) for a, axis in ((v, 0), (b, 1)))
            return jax.lax.dynamic_update_slice_in_dim(v, update(vk, bk, m), k, 0)

        return jax.lax.fori_loop(0, v.shape[0], one, v)

    return _into(run)


class _Volumes:
    """Elementwise updates of a level's volumes, each compiled into the volume passed first."""

    @staticmethod
    @partial(jax.jit, donate_argnums=(0,), keep_unused=True)
    def start(x: jax.Array, x0: float | jax.Array | np.ndarray) -> jax.Array:
        """Zero, or ``x0`` clipped at zero: a coarser level's volume, upsampled first."""
        x0 = jnp.asarray(x0, jnp.float32)
        if x0.shape not in ((), x.shape):
            x0 = jax.image.resize(x0, x.shape, method="linear", antialias=False)
        return jnp.broadcast_to(jnp.maximum(x0, 0), x.shape)

    observed_only = _into(lambda x, observed: jnp.where(observed, x, 0))
    free = _into(lambda _out, x, gx: (x > 0) | (gx < 0))
    # The step to max(x + solution, 0) on the free voxels, in the solution's buffer.
    clipped = _into(lambda s, free, x: jnp.maximum(jnp.where(free, x + s, x), 0) - x)
    stepped = _into(lambda _out, x, a, dx: x + _barrier(a * dx))
    # The shared rows' image gradient B^T G^-1 b (m = G^-1 b) and their part of the image
    # Jacobi diagonal, diag(B^T G^-1 B), added to the linearisation's (``_common``).
    plus_image = _by_slice(
        lambda v, b, m: v
        + _barrier((_barrier(m[0] * b[0]) + _barrier(m[1] * b[1])).astype(jnp.float32))
    )
    plus_diagonal = _by_slice(
        lambda dd, b, g: dd
        + _barrier(
            (
                _barrier(g[0, 0] * b[0] ** 2)
                + _barrier(2 * g[0, 1] * b[0] * b[1])
                + _barrier(g[1, 1] * b[1] ** 2)
            ).astype(jnp.float32)
        )
    )


@jax.jit
def _shared_gram(images: jax.Array) -> jax.Array:
    """B B^T in FP64: compiled, the FP64 copy of the images is no array of its own."""
    return jnp.einsum("i...,j...->ij", images.astype(jnp.float64), images.astype(jnp.float64))


def level_volumes(
    ops: Operators, prior: MotionPrior, lam: float
) -> dict[str, jax.ShapeDtypeStruct | FactorShape]:
    """The arrays a level holds through its steps, as shapes: allocated once, at its setup.

    The volume ``x`` and ``other`` (the line search's trial; once a step is accepted, the
    previous volume), the linearisation's gradient ``gx0`` and diagonal ``dd``, the shared
    images (``_common``), the free-voxel mask, the Schur preconditioner and ``step``
    (right-hand side, then solution, then clipped step), with TV the
    stepped gradient ``gx`` and conductance ``d``, and the motion prior's ``factor``.
    """
    volume = jax.ShapeDtypeStruct(ops.shape, jnp.float32)
    names = ("x", "other", "gx0", "dd", "invdiag", "step")
    shapes: dict[str, jax.ShapeDtypeStruct | FactorShape] = dict.fromkeys(
        names + (("gx", "d") if lam else ()), volume
    )
    shapes["images"] = jax.ShapeDtypeStruct((2, *ops.shape), jnp.float32)
    shapes["free"] = jax.ShapeDtypeStruct(ops.shape, jnp.bool_)
    shapes["factor"] = prior.factor_shapes()
    return shapes


def _gain_ratio(actual: float, predicted: float) -> float:
    """A non-descending model gives no evidence for decreasing LM damping."""
    return actual / predicted if predicted > 0 else 0.0


def _line_search[Image, Pose](
    trial: Callable[[float, float], tuple[float, Image, Pose]], f: float, gx_dx: float, gp_dc: float
) -> tuple[float, float, tuple[float, Image, Pose]] | None:
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
    """What every step from one linearisation shares, whatever its damping.

    The LM damping diagonal is max(dd, damping_floor), recovered from dd at its two uses.
    """

    q: Coefficients
    dd: jax.Array
    sw: ArrayStack
    j: ArrayStack
    gram: jax.Array
    residual: ArrayStack
    common: _common.Common
    damping_floor: jax.Array
    d: jax.Array | None
    gx: jax.Array
    wpose: jax.Array
    gp: jax.Array
    free: jax.Array
    pd: jax.Array


class _Proposal(NamedTuple):
    """A clipped volume step, its pose back-substitution and what the line search needs."""

    dx: jax.Array
    dc: jax.Array
    r: ArrayStack
    jadx: jax.Array
    bdx: jax.Array
    linear_step: KrylovResult


class _Level:
    """One level's data, compiled operators and damped Gauss-Newton step."""

    def __init__(
        self,
        ops: Operators,
        prior: MotionPrior,
        *,
        lam: float,
        eps: float,
        scales: tuple[float, float, float],
        delta: float,
        inner: int,
        image_tolerance: float = 0.03,
    ) -> None:
        self.ops, self.prior, self.lam, self.delta = ops, prior, lam, delta
        self.image_tolerance = image_tolerance
        self._made = None
        self.tv_value = jax.jit(lambda x: tv.energy(x, eps, scales))
        self.conduct = _into(lambda _out, x: tv.conductance(x, eps, scales))
        self.diffuse = jax.jit(lambda x, d: tv.diffusion(x, d, scales))
        self.gradient = _into(
            lambda _out, gx, x, d: gx + jnp.float32(lam) * tv.diffusion(x, d, scales)
        )

        @_into
        def inverse(
            _out: jax.Array, d: jax.Array | None, dd: jax.Array, floor: jax.Array, lm: float
        ) -> jax.Array:
            value = dd + jnp.float32(lm) * jnp.maximum(dd, floor)
            if d is not None:
                value = value + jnp.float32(lam) * tv.diagonal(d, scales)
            return 1 / jnp.maximum(value, jnp.float32(1e-8))

        self.inverse = inverse

        # At unchanged poses the prediction is linear in the image: the residual
        # plus a times the step's projection (returned scaled by sw, zero only
        # where unobserved) gives the profiled term exactly, without projecting;
        # the shared moments move by a times the step's B dx.
        def held(
            residual: ArrayStack,
            step: ArrayStack,
            sw: ArrayStack,
            w: ArrayStack | None,
            a: jax.Array,
            moments: jax.Array,
        ) -> jax.Array:
            def batch(total: jax.Array, z: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
                # The robust value is a sum over views, one batch at a time.
                i, valid = z
                rb, sb, swb = residual.read(i), step.read(i), sw.read(i)
                wb = ops.batch_precision(None if w is None else w.read(i), valid)
                moved = rb + a * jnp.where(swb > 0, sb / jnp.where(swb > 0, swb, 1), 0)
                return total + robust_value(profiled_residual(moved, wb), wb, delta), None

            parts = (jnp.arange(ops.nb), ops.batches(ops.valid))
            value = jax.lax.scan(batch, jnp.asarray(0.0, jnp.float64), parts)[0]
            return value + 0.5 * jnp.vdot(moments, self.inverse_gram @ moments)

        self.held = jax.jit(held)

        @_into
        def rhs(
            _out: jax.Array,
            q: Coefficients,
            sw: ArrayStack,
            j: ArrayStack,
            sp: jax.Array,
            gx: jax.Array,
            free: jax.Array,
            precision: ArrayStack | None,
            common: _common.Common,
        ) -> jax.Array:
            # Fixed voxels are discarded, so the gather may skip them.
            active = ops.projection.active(free)
            shared = common.image(common.inverse @ (common.poses @ sp))
            return (ops.pose_back(q, sw, j, sp, active, precision) + shared - gx) * free

        self.rhs = rhs
        self.norm = jax.jit(tv.dot)
        # Batch by batch: a whole-sinogram FP64 product would be a stack-sized temporary.
        self.energy = jax.jit(
            lambda r: jnp.sum(
                jax.lax.map(lambda i: tv.dot(r.read(i), r.read(i)), jnp.arange(ops.nb)),
                dtype=jnp.float64,
            )
        )

        @partial(jax.jit, donate_argnums=(8, 13))
        def schur(
            q: Coefficients,
            sw: ArrayStack,
            j: ArrayStack,
            factor: PoseFactor,
            free: jax.Array,
            d: jax.Array | None,
            damping: tuple[jax.Array, jax.Array],
            lm: float,
            rhs: jax.Array,
            invdiag: jax.Array,
            precision: ArrayStack | None,
            common: _common.Common,
            poses: _common.PoseSolve,
            r: ArrayStack,
        ) -> tuple[KrylovResult, ArrayStack]:
            # Fixed voxels' transpose is discarded: the gather may skip them.
            active = ops.projection.active(free)

            def matvec(v: jax.Array, r: ArrayStack) -> tuple[jax.Array, ArrayStack]:
                v = v * free
                r, jr = ops.weighted_forward(q, v, sw, j, precision, r)
                bv = common.of_image(v)
                p = poses(prior.solve, factor, common, jr + common.poses.T @ (common.inverse @ bv))
                out = ops.weighted_back(q, r, sw, j, p, active, precision)
                out = out + common.image(common.inverse @ (bv - common.poses @ p))
                if d is not None:
                    out = out + jnp.float32(lam) * tv.diffusion(v, d, scales)
                diagonal, floor = damping
                out = out + (jnp.float32(lm) * jnp.maximum(diagonal, floor)) * v
                return out * free, r

            return krylov_solve(matvec, rhs, invdiag, inner, rtol=image_tolerance, work=r)

        self._schur = schur

    def place(
        self,
        y: np.ndarray,
        precision: np.ndarray | None,
        placement: Mapping[str, Place] | None = None,
        *,
        x0: jax.Array | np.ndarray | None = None,
    ) -> LevelStacks:
        """Allocate the level's volumes and place and allocate its stacks (``_stacks``).

        Beside them the plan keeps the largest transient free: the temporaries of the Schur
        solve, of the motion prior's factor, and of the start (with ``x0`` None, the Gram of
        the shared images FISTA starts from; otherwise ``x0`` brought to the level's shape),
        each from its compile here (the calls later reuse it); and the one chunk it lands
        in, which the allocator can hand out whole up to twice the request when that wastes
        under 128 MiB. Returns the level's storage, which the caller closes.
        """
        ops, prior = self.ops, self.prior
        specs = ops.stack_specs(precision)
        shapes = self._schur_shapes(specs, precision is not None)
        self.transients = {"schur": temporaries(self._schur, *shapes), "prior": prior.temporaries()}
        if x0 is None:
            start = temporaries(_shared_gram, jax.ShapeDtypeStruct((2, *ops.shape), jnp.float32))
        else:
            start = temporaries(_Volumes.start, jax.ShapeDtypeStruct(ops.shape, jnp.float32), x0)
        self.transients["start"] = start
        reserve = transient_reserve(self.transients)
        volumes = level_volumes(ops, prior, self.lam)
        self.stacks = level_stacks(
            ops, y, precision, reserve=reserve, volumes=volumes, placement=placement
        )
        self.y, self.precision = self.stacks["y"], self.stacks.get("precision")
        self.gram, self.data_moments = ops.shared_data(self.y, self.precision)
        self.inverse_gram = jnp.linalg.pinv(self.gram)  # zero if no view is profiled
        return self.stacks

    def _schur_shapes(
        self, specs: Mapping[str, StackSpec], weighted: bool | np.bool_
    ) -> _SchurShapes:
        """The Schur solve's arguments as shapes, every stack on the device."""
        ops, prior = self.ops, self.prior
        f64 = partial(jax.ShapeDtypeStruct, dtype=jnp.float64)
        volume = jax.ShapeDtypeStruct(ops.shape, jnp.float32)

        def stack(name: str) -> Stack[_Shape]:
            return Stack.shaped(specs[name], ops.batch)

        q = jax.eval_shape(lambda c: ops.prepare(ops.pad_poses(c)), f64((prior.size,)))
        factor = prior.factor_shapes()
        images = jax.ShapeDtypeStruct((2, *ops.shape), jnp.float32)
        parts = (f64((2, 2)), f64((2, 2)), images, f64((2, prior.size)), f64((2,)))
        common = jax.eval_shape(_common.Common, *parts)
        poses = jax.eval_shape(partial(_common.PoseSolve.build, prior.solve), factor, common)
        free = jax.ShapeDtypeStruct(ops.shape, jnp.bool_)
        d = volume if self.lam else None
        precision = stack("precision") if weighted else None
        damping = (volume, jax.ShapeDtypeStruct((), jnp.float32))
        head = (q, stack("sw"), stack("j"), factor, free, d, damping, 1.0, volume, volume)
        return (*head, precision, common, poses, stack("r"))

    def _write[**P](
        self,
        name: VolumeName,
        fun: Callable[Concatenate[jax.Array, P], jax.Array],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> jax.Array:
        """``fun`` run into the level's volume ``name``, its donated first argument; held."""
        out = fun(self.stacks[name], *args, **kwargs)
        self.stacks.update([(name, out)])
        return out

    def schur(self, *args: *_SchurOperands) -> KrylovResult:
        """Solve into the donated right-hand side and weighted-projection stack.

        The step volume holds the right-hand side on entry, the solution on return
        and the clipped step when the caller imposes nonnegativity.
        """
        result, r = self._schur(*args, self.stacks["r"])
        self.stacks.update([("r", r), ("step", result.solution)])
        return result

    def image_penalty(self, x: jax.Array) -> float:
        """Image-prior objective, with no TV call or temporary when its weight is zero."""
        return float(self.lam * self.tv_value(x)) if self.lam else 0.0

    def initial_volume(
        self, q: jax.Array, lipschitz: float | jax.Array, iterations: int
    ) -> jax.Array:
        """FISTA from zero on the Huber initialization data term and the shared term, no TV."""
        ops, y, precision, delta = self.ops, self.y, self.precision, self.delta
        images = ops.common_images(q, precision, self.stacks["images"])
        self.stacks.update([("images", images)])
        inverse, data_moments = self.inverse_gram, self.data_moments
        # The shared term's Hessian B^T G^-1 B has rank two; its largest eigenvalue is G^-1 B B^T's.
        outer = _shared_gram(images)
        lipschitz = lipschitz + float(jnp.max(jnp.real(jnp.linalg.eigvals(inverse @ outer))))

        # The shared images are an argument: captured, the program would hold them as a constant.
        def advance(
            state: _FistaState,
            count: int | jax.Array,
            q: Coefficients,
            y: ArrayStack,
            precision: ArrayStack | None,
            lipschitz: float | jax.Array,
            images: jax.Array,
            inverse: jax.Array,
            data_moments: jax.Array,
        ) -> _FistaState:
            def fista(_: jax.Array, state: _FistaState) -> _FistaState:
                old, z, t = state
                shared = inverse @ (
                    jnp.einsum("i...,...->i", images, z).astype(jnp.float64) - data_moments
                )
                gradient = ops.initial_gradient(q, z, y, precision, delta) + (
                    shared[0] * images[0] + shared[1] * images[1]
                ).astype(jnp.float32)
                new = jnp.maximum(z - gradient / lipschitz, 0)
                tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
                return new, new + ((t - 1) / tn).astype(jnp.float32) * (new - old), tn

            return jax.lax.fori_loop(0, count, fista, state)

        def first(x: jax.Array, count: int | jax.Array, *args: *_StartOperands) -> _FistaState:
            return advance((x, x, jnp.asarray(1.0, jnp.float64)), count, *args)

        self._write("x", _Volumes.start, 0.0)  # an argument: a constant zero could be folded
        args = (ops.prepare(q), y, precision, lipschitz, images, inverse, data_moments)
        task = progress.for_unit("iterations")
        if task is None or not iterations:

            @_into
            def run(x: jax.Array, *operands: *_StartOperands) -> jax.Array:
                return first(x, iterations, *operands)[0]

            return self._write("x", run, *args)
        # Only x is donated on the first dispatch: old and z both start from x.
        start_chunk, next_chunk = _into(first), _into(advance)
        state = None
        for low, high in progress.chunks(iterations):
            with progress.preparing("initial image update", max(0, low - 1)):
                state = (
                    start_chunk(self.stacks["x"], high, *args)
                    if state is None
                    else next_chunk(state, high - low, *args)
                )
                jax.block_until_ready(state)
            self.stacks.update([("x", state[0])])
            task.advance(high, {})
            progress.preview(
                lambda state=state: planes(state[0], self.ops.grid), final=high == iterations
            )
            progress.check_cancelled()
        return self.stacks["x"]

    def start(self, c: jax.Array, x0: jax.Array | np.ndarray | None, iterations: int) -> jax.Array:
        """The level's first volume at poses ``c``, from ``x0`` (None: FISTA from zero)."""
        q = self.ops.pad_poses(c)
        lipschitz, observed = self.ops.sensitivity(q, self.stacks["free"])
        self.stacks.update([("free", observed)])
        if x0 is None:
            with progress.task("image", "initial image", iterations, "iterations"):
                self.initial_volume(q, lipschitz, iterations)
        else:
            self._write("x", _Volumes.start, x0)
        # Prolongation can populate voxels invisible to every ray at this level.
        # Start those undetermined values at zero, but let later pose/image steps
        # expose or populate them; this is not a permanent active-set constraint.
        return self._write("x", _Volumes.observed_only, observed)

    def accept(self, x: jax.Array, previous: jax.Array) -> None:
        """Hold an accepted step's volume as the level's, and the previous one beside it."""
        self.stacks.update([("x", x), ("other", previous)])

    @progress.task("linearize", "Measure image and pose changes", None, "")
    def linearize(
        self, c: jax.Array, x: jax.Array
    ) -> tuple[
        jax.Array,
        jax.Array,
        ArrayStack,
        ArrayStack,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        ArrayStack,
        _common.Common,
    ]:
        """The robust linearization at compact poses ``c`` and volume ``x``, with the shared term.

        Returns the gradient, diagonal, weights, columns, score, Gram blocks, data
        value, observation counts and residual, each including the views' shared
        offset and ramp (``_common.Common``, returned last).
        """
        q = self.ops.pad_poses(c)
        stacks = self.stacks
        names = ("gx0", "dd", "sw", "j", "residual")
        gx0, dd, sw, j, gr, gram, data, obstats, moments, residual, moved = self.ops.linearize(
            q, x, self.y, self.precision, self.delta, tuple(stacks[n] for n in names)
        )
        images = self.ops.common_images(q, self.precision, stacks["images"])
        stacks.update(
            (
                ("gx0", gx0),
                ("dd", dd),
                ("sw", sw),
                ("j", j),
                ("residual", residual),
                ("images", images),
            )
        )
        common = _common.Common(self.gram, self.inverse_gram, images, moved, moments)
        gx0 = self._write("gx0", _Volumes.plus_image, images, common.inverse @ common.moments)
        dd = self._write("dd", _Volumes.plus_diagonal, images, common.inverse)
        gr = gr + common.pose_score()
        return gx0, dd, sw, j, gr, gram, data + common.value(), obstats, residual, common

    def trial(
        self,
        x: jax.Array,
        c: jax.Array,
        dx: jax.Array,
        dc: jax.Array,
        pose_weight: PoseWeight,
        held: tuple[ArrayStack, ArrayStack, ArrayStack, jax.Array, jax.Array],
        a: float,
        beta: float,
    ) -> _Trial:
        """Objective after volume step a and pose step a * beta, with the new state.

        ``held`` (residual, weighted projection of dx, weights, shared moments and
        their change B dx) gives beta = 0.
        """
        # The trial volume, made once for each line step a in the level's other volume.
        made = self._made
        if made is None or made[0] is not dx or made[1] != a:
            stepped = self._write("other", _Volumes.stepped, x, jnp.float32(a), dx)
            made = self._made = (dx, a, stepped)
        xt = made[2]
        cc = c + a * beta * dc
        if beta == 0:
            residual, step, sw, moments, change = held
            data = self.held(
                residual, step, sw, self.precision, jnp.float32(a), moments + a * change
            )
        else:
            q = self.ops.pad_poses(cc)
            value, moments = self.ops.objective(q, xt, self.y, self.precision, self.delta)
            data = value + 0.5 * jnp.vdot(moments, self.inverse_gram @ moments)
        total = float(data) + self.image_penalty(xt)
        return total + float(self.prior.value(cc, pose_weight)), xt, cc

    def _context(
        self, x: jax.Array, c: jax.Array, linear: _Linearization, pose_weight: PoseWeight
    ) -> _StepContext:
        """What every step from this linearisation shares, whatever its damping."""
        ops, prior = self.ops, self.prior
        gx0, dd, sw, j, gr, gram, residual, common = linear
        d = self._write("d", self.conduct, x) if self.lam else None
        gx = self._write("gx", self.gradient, gx0, x, d) if d is not None else gx0
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
            damping_floor=jnp.maximum(dd.max(), 1e-8) * 1e-4,
            d=d,
            gx=gx,
            wpose=wpose,
            gp=gr + prior.apply(c, wpose),
            free=self._write("free", _Volumes.free, x, gx),
            pd=jnp.maximum(pd, jnp.maximum(pd.max(), 1e-8) * 0.01),
        )

    def _propose(self, x: jax.Array, k: _StepContext, lm: float) -> _Proposal | None:
        """The Schur volume step at damping lm, clipped to x >= 0, and its pose back-substitution.

        Returns None when the volume solve does not give a finite residual.
        """
        ops, prior, common = self.ops, self.prior, k.common
        blocks = k.gram + jax.vmap(jnp.diag)(lm * k.pd + 1e-10)
        factor = prior.factor(blocks, k.wpose, self.stacks["factor"])
        self.stacks.update([("factor", factor)])
        poses = _common.PoseSolve.build(prior.solve, factor, common)
        sp = poses(prior.solve, factor, common, k.gp)
        rhs = self._write(
            "step", self.rhs, k.q, k.sw, k.j, sp, k.gx, k.free, self.precision, common
        )
        invdiag = self._write("invdiag", self.inverse, k.d, k.dd, k.damping_floor, lm)
        linear_step = self.schur(
            k.q,
            k.sw,
            k.j,
            factor,
            k.free,
            k.d,
            (k.dd, k.damping_floor),
            lm,
            rhs,
            invdiag,
            self.precision,
            common,
            poses,
        )
        if not np.isfinite(float(linear_step.relative_residual)):
            return None
        dx = self._write("step", _Volumes.clipped, k.free, x)
        # The step's projection and the matvec's never coexist: one stack serves both.
        r, jadx = ops.weighted_forward(k.q, dx, k.sw, k.j, self.precision, self.stacks["r"])
        self.stacks.update([("r", r)])
        bdx = common.of_image(dx)
        jadx = jadx + common.poses.T @ (common.inverse @ bdx)
        dc = -poses(prior.solve, factor, common, k.gp + jadx)
        return _Proposal(dx, dc, r, jadx, bdx, linear_step)

    @progress.task("solve", "Check convergence", None, "")
    def proposal(
        self, x: jax.Array, c: jax.Array, linear: _Linearization, lm: float, pose_weight: PoseWeight
    ) -> float | None:
        """The RMS pose change one step from this linearisation proposes at damping lm, or None."""
        step = self._propose(x, self._context(x, c, linear, pose_weight), lm)
        if step is None:
            return None
        dp = self.prior.expand(step.dc)
        return float(jnp.sqrt(jnp.mean(jnp.sum(dp**2, axis=1))))

    def step(
        self,
        x: jax.Array,
        c: jax.Array,
        linear: _Linearization,
        lm: float,
        pose_weight: PoseWeight,
        f: float,
    ) -> _AcceptedStep | None:
        """One damped step: Schur volume solve, exact pose back-substitution, line search.

        Raises the damping tenfold, up to five times, until the line search accepts.
        Returns the accepted step with its predicted reduction, or None.
        """
        k = self._context(x, c, linear, pose_weight)
        for attempt in range(5):
            # A refused proposal's step and trials end with the attempt, before the next
            # Schur solve: kept, they were the level's peak (13 volumes on K11 54013).
            with (
                progress.task("retry", f"Retry update {attempt}", None, "")
                if attempt
                else nullcontext()
            ):
                accepted = self._attempt(x, c, k, lm, pose_weight, f)
            if accepted is not None:
                return accepted
            lm *= 10
        return None

    def _attempt(
        self,
        x: jax.Array,
        c: jax.Array,
        k: _StepContext,
        lm: float,
        pose_weight: PoseWeight,
        f: float,
    ) -> _AcceptedStep | None:
        """One proposal at damping lm and its line search: the accepted step's record, or None."""
        prior, common = self.prior, k.common
        with progress.task("solve", "Solve image and pose update", None, ""):
            proposed = self._propose(x, k, lm)
        if proposed is None:
            return None
        dx, dc, r, jadx, bdx = proposed.dx, proposed.dc, proposed.r, proposed.jadx, proposed.bdx
        linear_step = proposed.linear_step
        cg_steps, cg_residual = linear_step.iterations, linear_step.relative_residual
        gx_dx, gp_dc = float(self.norm(k.gx, dx)), float(jnp.vdot(k.gp, dc))
        accepted = None
        if gx_dx + gp_dc < 0:
            held = (k.residual, r, k.sw, common.moments, bdx)
            trial = partial(self.trial, x, c, dx, dc, pose_weight, held)
            with progress.task("line_search", "Test update size", None, ""):
                accepted = _line_search(trial, f, gx_dx, gp_dc)
        if accepted:
            a, beta, new = accepted
            dp = prior.expand(dc)
            curvature = float(self.energy(r))
            if k.d is not None:
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
            proposed_rms = float(jnp.sqrt(jnp.mean(jnp.sum(dp**2, axis=1))))
            linear_record = {
                Metric.cg_residual_kind.name: "explicit" if explicit else "recursive",
                Metric.cg_status.name: "converged"
                if explicit and float(cg_residual) <= self.image_tolerance
                else "budget_exhausted",
                Metric.cg_matvecs.name: int(linear_step.matvecs),
            }
            return _AcceptedStep(
                a,
                beta,
                new,
                predicted,
                change,
                lm,
                int(cg_steps),
                float(cg_residual),
                proposed_rms,
                linear_record,
            )
        return None


def _undamped_proposal(
    level: _Level,
    step: _AcceptedStep,
    x: jax.Array,
    c: jax.Array,
    linear: _Linearization,
    budget: Budget,
    pose_weight: PoseWeight,
) -> float | None:
    """The step's pose proposal at the initial damping when its damped one is small, else None.

    Alignment has converged when its proposed geometry updates are small, even
    if more image iterations could still reduce the residual. A proposal damped
    harder than the initial damping is no such evidence: damping can suppress
    any step. So a small damped proposal is checked again at the initial
    damping, from the same linearisation, and that proposal decides. The
    damping itself does not: clipping the image step at x >= 0 can keep it high
    long after the poses have settled.
    """
    if step.proposed_rms >= budget.tolerance:
        return None
    if step.lm <= budget.lm_initial:
        return step.proposed_rms
    return level.proposal(x, c, linear, budget.lm_initial, pose_weight)


def _pose_weight(
    level: _Level, linear: _Linearization, sigma: float
) -> tuple[PoseWeight, float | None, float]:
    """The motion prior's weight from one linearisation, with the error scale it used.

    The weight is a threshold: the views' median Fisher information F, times the
    standard error of one view's step, times sqrt(2 log n). For the rotation
    increments that standard error is the photon model's, sigma / sqrt(F). For the
    translations it is taken from the data: each view's own step for each
    translation column, d = g / F_cc (the column's score over its diagonal Gram
    entry, the other columns held), and s = MAD / 0.6745 of the second differences
    of d along the views, over sqrt(6) (a second difference of independent values
    has six times their variance). Real motion is smooth or rare from one view to
    the next; the error a fit must not follow (noise, and model error that differs
    from view to view) is not. On clean data s comes out near the photon value;
    where every view carries its own model error it is larger, and so is the weight.

    Returns the weight ((translations', rotations') when both are fitted), s (None
    without translation columns) and F.
    """
    ops = level.ops
    _, _, sw, j, _, gram, residual, _ = linear
    n, local = ops.n, ops.parameters.local
    fisher = float(jnp.median(jnp.trace(gram, axis1=1, axis2=2)))
    photon = float(sigma * np.sqrt(fisher) * np.sqrt(2 * np.log(n)))
    translations = ops.parameters.translation_columns
    if not translations or n < 3:
        return photon, None, fisher
    g = np.asarray(ops.translation_scores(sw, j, residual, level.precision), np.float64)
    diagonal = np.diagonal(np.asarray(gram, np.float64), axis1=1, axis2=2)[:, translations]
    d = g / np.where(diagonal > 0, diagonal, np.inf)
    second = d[2:] - 2 * d[1:-1] + d[:-2]
    spread = np.median(np.abs(second - np.median(second, axis=0)), axis=0) / 0.6745 / np.sqrt(6)
    s = float(np.sqrt(np.mean(spread**2)))
    weight = float(fisher * s * np.sqrt(2 * np.log(n)))
    return (weight, photon) if len(translations) < len(local) else weight, s, fisher


def solve(
    y: np.ndarray,
    precision: np.ndarray | None,
    prior: MotionPrior,
    weight: float,
    eps: float,
    *,
    ops: Operators,
    budget: Budget,
    x0: jax.Array | np.ndarray | None,
    p0: jax.Array | np.ndarray | None,
    pose_sigma: float,
    delta: float,
    scales: tuple[float, float, float],
    placement: Mapping[str, Place] | None = None,
) -> tuple[jax.Array, jax.Array, np.ndarray, LevelRecord]:
    """Fit the volume and the per-view pose parameters at one level.

    ``y`` and ``precision`` are the level's host data (``precision`` None when
    uniform). The level's stacks are placed and allocated first (``_stacks``) and
    released when it ends, whatever the outcome; ``placement`` forces stacks to
    "device" or "host" by name. Returns the volume, compact free parameters, the
    data and a record of the steps, the stacks' placement and the level's time.
    """
    start = time.perf_counter()
    width = prior.size
    c = (
        jnp.zeros(width, jnp.float64)
        if p0 is None
        else prior.project_gauge(jnp.asarray(p0, jnp.float64).reshape(-1))
    )
    level = _Level(
        ops,
        prior,
        lam=weight,
        eps=eps,
        scales=scales,
        delta=delta,
        inner=budget.image_iterations,
        image_tolerance=budget.image_tolerance,
    )
    stacks = level.place(y, precision, placement, x0=x0)
    try:
        _, c, record = _descend(level, c, x0, budget, pose_sigma, weight)
        x = stacks.take("x")  # the caller's now: the level's other storage is released
    except jax.errors.JaxRuntimeError as err:
        if progress.handler_failed(err):
            raise
        raise unplaced(err, stacks) from err
    finally:
        stacks.close()
    stacks_record: StorageRecord = {**stacks.record(), "transients": dict(level.transients)}
    return x, c, y, {**record, "seconds": time.perf_counter() - start, "stacks": stacks_record}


def _descend(
    level: _Level,
    c: jax.Array,
    x0: jax.Array | np.ndarray | None,
    budget: Budget,
    pose_sigma: float,
    weight: float,
) -> tuple[jax.Array, jax.Array, _DescentRecord]:
    """The level's damped Gauss-Newton steps from poses ``c`` and volume ``x0`` (None: FISTA)."""
    prior = level.prior
    x = level.start(c, x0, budget.initial_image_iterations)
    history: list[dict[str, _StepMetric]]
    history, lm, stable, status = [], budget.lm_initial, 0, "step_budget_reached"
    f = 0.0
    previous_weight: PoseWeight | None = None
    level_task = progress.for_unit("steps")
    # A zero-step level still measures its initial objective and motion prior.
    for it in range(max(1, budget.iterations)):
        with (
            progress.task("step", f"Gauss-Newton step {it + 1}", None, "")
            if budget.iterations
            else nullcontext()
        ):
            gx0, dd, sw, j, gr, gram, data, obstats, residual, common = level.linearize(c, x)
            linear = (gx0, dd, sw, j, gr, gram, residual, common)
            pose_weight, noise, fisher = _pose_weight(level, linear, pose_sigma)
            if previous_weight is not None:
                f += float(prior.value(c, pose_weight)) - float(prior.value(c, previous_weight))
            else:
                f = float(data + level.image_penalty(x) + prior.value(c, pose_weight))
                initial_objective = f
                if not budget.iterations:
                    break
            previous_weight = pose_weight
            result = level.step(x, c, linear, lm, pose_weight, f)
            if result is None:
                status = "line_search_stalled"
                break
            x0, c0 = x, c
            fn, x, c = result.trial
            level.accept(x, x0)
            rho = _gain_ratio(f - fn, result.predicted)
            valid, down = map(float, np.asarray(obstats))
            history.append(
                {
                    Metric.objective.name: fn,
                    Metric.pose_step_rms.name: result.pose_step_rms,
                    Metric.line_step.name: result.line_step,
                    Metric.pose_fraction.name: result.pose_fraction,
                    Metric.lm.name: result.lm,
                    Metric.gain_ratio.name: rho,
                    Metric.predicted_reduction.name: result.predicted,
                    Metric.cg_iterations.name: result.cg_steps,
                    Metric.cg_relative_residual.name: result.cg_residual,
                    Metric.proposed_pose_step_rms.name: result.proposed_rms,
                    **result.linear_record,
                    Metric.fraction_downweighted.name: down / max(valid, 1),
                }
            )
            f, lm = fn, result.lm
            if rho > 0.75 and result.line_step == 1:
                lm = max(_LM_MIN, lm / 2)
            elif rho < 0.25 or result.line_step < 0.5:
                lm = min(lm * 3, 1e6)
            undamped = _undamped_proposal(level, result, x0, c0, linear, budget, pose_weight)
            history[-1][Metric.undamped_proposed_pose_step_rms.name] = undamped
            # Two sets of linearized view stacks would exceed the placement budget.
            del gx0, dd, sw, j, gr, gram, residual, common, linear
        if level_task is not None:
            jax.block_until_ready((x, c))
            level_task.advance(len(history), history[-1])
            progress.preview(lambda x=x, c=c: level.ops.preview(x, c))
        stable = stable + 1 if undamped is not None and undamped < budget.tolerance else 0
        if stable >= 2 and it + 1 >= budget.min_iterations:
            status = "pose_steps_below_tolerance"
            break
    if history:
        progress.preview(lambda: level.ops.preview(x, c), final=True)
    final_prior = float(prior.value(c, pose_weight))
    final_image_prior = level.image_penalty(x)
    record: _DescentRecord = {
        "status": status,
        "steps": history,
        "tv_weight": float(weight),
        "noise_sigma": float(pose_sigma),
        "pose_prior_weight": list(pose_weight) if isinstance(pose_weight, tuple) else pose_weight,
        "translation_noise": noise,
        "pose_fisher": fisher,
        "initial_objective": initial_objective,
        "final_objective": f,
        "final_data_objective": f - final_prior - final_image_prior,
        "final_pose_prior": final_prior,
        "final_image_prior": final_image_prior,
        "iterations": budget.iterations,
        "image_iterations": budget.image_iterations,
        "initial_image_iterations": budget.initial_image_iterations,
    }
    return x, c, record
