"""Exact bordered pose solves with translation drift/jumps and rotation size and continuity priors.

The local normal matrix is F + R + Q K Q: per-view data blocks F, physical
rotation precision R, and a difference Laplacian K, for translations outside
the drift spline U (Q = I - U U.T) and for rotations on their own increments.
The spline tree factors this system on contiguous view intervals. Systems with few knots
use cyclic reduction with a small Woodbury border. Gauge constraints and
shared calibration columns border either local factor.
"""

import math
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve
import numpy as np
from scipy import sparse

from . import _spline, _spline_solve
from ._stacks import temporaries

# Standard deviation of the per-view rotation size prior.
ROTATION_SIGMA_DEG = 0.5


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


def _pcr_factor(diag, upper, lower):
    """Parallel cyclic reduction on the remaining short chain."""
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


class _Tridiagonal(NamedTuple):
    levels: tuple
    coarse: tuple

    @classmethod
    def build(cls, blocks, w) -> "_Tridiagonal":
        """Eliminate alternate views until ordinary parallel reduction fits in 512 rows."""
        kdiag = jnp.pad(w, ((0, 1), (0, 0))) + jnp.pad(w, ((1, 0), (0, 0)))
        diag = blocks + jax.vmap(jnp.diag)(kdiag)
        upper = jax.vmap(jnp.diag)(-jnp.pad(w, ((0, 1), (0, 0))))
        levels = []
        while len(diag) > 512:
            odd = _inverse(diag[1::2])
            left, right = upper[: 2 * len(odd) : 2], upper[1::2]
            u, v = odd @ left.mT, odd @ right
            even = diag[::2]
            extra = len(even) - len(odd)
            diag = even - jnp.pad(left @ u, ((0, extra), (0, 0), (0, 0)))
            diag -= jnp.pad(right.mT @ v, ((1, 0), (0, 0), (0, 0)))[: len(even)]
            upper = jnp.pad(-left @ v, ((0, extra), (0, 0), (0, 0)))
            levels.append((odd, u, v))
        lower = _shift(upper.mT if levels else upper, -1)
        return cls(tuple(levels), _pcr_factor(diag, upper, lower))

    def solve(self, rhs):
        vector = rhs.ndim == 2
        rhs = rhs[..., None] if vector else rhs
        saved = []
        for inv, u, v in self.levels:
            even, odd = rhs[::2], rhs[1::2]
            extra = len(even) - len(odd)
            rhs = even - jnp.pad(u.mT @ odd, ((0, extra), (0, 0), (0, 0)))
            rhs -= jnp.pad(v.mT @ odd, ((1, 0), (0, 0), (0, 0)))[: len(even)]
            saved.append(inv @ odd)
        x = _pcr_solve(self.coarse, rhs)
        for (_, u, v), direct in zip(reversed(self.levels), reversed(saved), strict=True):
            m = len(direct)
            odd = direct - u @ x[:m] - v @ jnp.pad(x[1:], ((0, 1), (0, 0), (0, 0)))[:m]
            pair = jnp.stack([x[:m], odd], axis=1)
            x = jnp.concatenate([pair.reshape(2 * m, *x.shape[1:]), x[m:]])
        return x[..., 0] if vector else x


class _Woodbury(NamedTuple):
    tri: _Tridiagonal
    b: jax.Array
    solved: jax.Array
    lu: tuple | None

    @classmethod
    def build(cls, tri, u, ku) -> "_Woodbury":
        b = jnp.concatenate([u, ku], axis=1)
        k = u.shape[1]
        if k:
            zero, eye = jnp.zeros((k, k), jnp.float64), jnp.eye(k, dtype=jnp.float64)
            small = jnp.block([[zero, -eye], [-eye, -u.T @ ku]])
            local = tri.coarse[1].shape[-1]
            ab = tri.solve(b.reshape(-1, local, b.shape[-1])).reshape(u.shape[0], -1)
            small = small + b.T @ ab
            fac = lu_factor((small + small.T) * 0.5)
        else:
            ab, fac = b, None
        return cls(tri, b, ab, fac)

    def solve(self, rhs):
        k = self.tri.coarse[1].shape[-1]
        n = len(rhs) // k
        direct = self.tri.solve(rhs.reshape(n, k, *rhs.shape[1:])).reshape(rhs.shape)
        return (
            direct - self.solved @ lu_solve(self.lu, self.b.T @ direct)
            if self.lu is not None
            else direct
        )


class ViewOrder:
    """When each view was taken, as the motion prior and the report use it.

    ``positions`` place the views in acquisition time (a scan's
    ``view_positions``); None means consecutive. A view links to the next when
    its position increases; a decrease starts a new acquisition (a combined
    scan) with its own drift and no link across. Gaps are measured in units of
    the median gap, so a regular selection has unit steps.
    """

    def __init__(self, views, positions=None) -> None:
        self.n = int(views)
        if positions is None:
            gaps = np.ones(max(self.n - 1, 0))
        else:
            positions = np.asarray(positions, np.float64)
            if positions.shape != (self.n,) or not np.isfinite(positions).all():
                raise ValueError(f"view positions need one finite value for each of {self.n} views")
            gaps = np.diff(positions)
        self.linked = gaps > 0
        regular = np.median(gaps[self.linked]) if self.linked.any() else 1.0
        self.steps = np.where(self.linked, gaps / regular, 0.0)
        starts = np.r_[0, np.flatnonzero(~self.linked) + 1]
        self.segments = list(zip(starts, np.r_[starts[1:], self.n], strict=True))

    def spline_design(self, spacing):
        """Block-diagonal drift design: one spline per acquisition segment."""
        blocks = []
        for start, stop in self.segments:
            steps = np.r_[0.0, np.cumsum(self.steps[start : stop - 1])]
            blocks.append(_spline.sampled(steps, spacing))
        return blocks[0] if len(blocks) == 1 else sparse.block_diag(blocks, format="csr")

    def drift(self, spacing, translated):
        """The explicit basis for a narrow knot border or the interval decomposition."""
        views, nlocal = self.n, len(spacing)
        # A bounded knot border costs O(views), even for a long acquisition.
        # Beyond this width keep just the modes that cross interval ends.
        designs = {
            s: self.spline_design(s) for s in {s for i, s in enumerate(spacing) if translated[i]}
        }
        axes = [(i, designs[s]) for i, s in enumerate(spacing) if translated[i]]
        widths = [cast("tuple[int,int]", b.shape)[1] for _, b in axes]
        empty = np.empty((views * nlocal, 0))
        if not axes:
            return empty, None
        if sum(widths) <= 80 * nlocal:
            sampled = np.zeros((views * nlocal, sum(widths)))
            start = 0
            for (axis, b), width in zip(axes, widths, strict=True):
                sampled[axis::nlocal, start : start + width] = b.toarray()
                start += width
            u, r = np.linalg.qr(sampled, mode="reduced")
            if np.any(
                abs(r.diagonal())
                <= max(sampled.shape) * np.finfo(float).eps * np.linalg.norm(r, ord=np.inf)
            ):
                u = _orthonormal_columns(sampled)
            return u, None
        columns = [sparse.kron(b, np.eye(nlocal)[:, [i]], format="csr") for i, b in axes]
        return empty, _spline.Drift.build(sparse.hstack(columns, format="csr"), views, nlocal)

    @property
    def jump_scale(self):
        """Per link: 1/sqrt(steps), so a jump over n steps counts as n steps would; 0 unlinked."""
        return np.where(self.linked, 1 / np.sqrt(np.where(self.linked, self.steps, 1.0)), 0.0)

    @property
    def edge_scale(self):
        """Per link: 1/steps, the precision of an n-step difference; 0 unlinked."""
        return np.where(self.linked, 1 / np.where(self.linked, self.steps, 1.0), 0.0)


def _orthonormal_columns(matrix):
    """An orthonormal basis without null or repeated constraint columns."""
    matrix = np.asarray(matrix, np.float64)
    if not matrix.size:
        return np.zeros((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(singular.max(initial=0), 1)
    return u[:, singular > tolerance]


class PriorMatrices(NamedTuple):
    """The motion prior's fixed matrices, passed to its compiled functions as an argument."""

    chain: _spline.Drift | None
    U: jax.Array
    G: jax.Array
    C: jax.Array
    gauge_step: jax.Array
    rotation_precision: jax.Array


class MotionPrior:
    """Translation drift/jumps, a physical rotation prior, and an exact global border.

    Local translations have a free cubic drift spline and Student-t innovations,
    w delta log(1 + J^2 / delta^2) per link: the restoring force 2 w delta J /
    (delta^2 + J^2) is largest at J = delta, where it equals the weight w, so w
    is the force the prior can exert against a view's own error (the threshold
    the solver derives it as), and it falls for larger jumps. Rotations have a
    zero-mean Gaussian size prior with ``rotation_sigma_deg`` standard
    deviation, in physical radians rather than numerical pixel units, and
    Gaussian increments between neighbouring views of curvature
    ``weight / delta``, with no free drift: wobble is continuous in the angle.
    The weight may be a pair, (translations', rotation increments').
    Multiplication by ``noise_sigma**2`` puts that prior in the unnormalised
    data loss's units. Shared centre and angle parameters are not penalised.

    ``blocks`` contains every view's local/global Gram block. The local system
    is solved exactly on the orthogonal drift and innovation spaces, with
    rigid gauge constraints. A small Schur border then couples all views to the
    single copy of each global parameter. No view-wise global approximation is
    made. ``innovation_solve`` holds that border exactly fixed for refinement.

    ``gauge`` holds the tangents of the unobservable common coordinate changes
    and ``constraint`` the rows that fix them (``constraint.T @ p == 0``),
    by default the tangents themselves.
    """

    def __init__(
        self,
        views: int,
        gauge: np.ndarray,
        *,
        constraint: np.ndarray | None = None,
        knot_spacing: tuple[int, ...],
        delta: float,
        independent: bool = False,
        parameters=None,
        rotation_scales=None,
        noise_sigma=0.0,
        rotation_sigma_deg=ROTATION_SIGMA_DEG,
        order=None,
    ) -> None:
        self.parameters, self.n = parameters, views
        self.order = order = ViewOrder(views) if order is None else order
        self.nlocal = len(knot_spacing) if parameters is None else parameters.nlocal
        self.nglobal = 0 if parameters is None else parameters.nglobal
        self.k, self.size = self.nlocal + self.nglobal, views * self.nlocal + self.nglobal
        self.local_size, self.delta = views * self.nlocal, float(delta)
        self.independent = independent
        local = tuple(range(self.nlocal)) if parameters is None else parameters.local
        spacing = tuple(knot_spacing[i] for i in local)
        translated = (
            np.ones(self.nlocal)
            if parameters is None
            else np.asarray([i >= 3 for i in local], np.float64)
        )
        self.translation_mask = jnp.asarray(translated, jnp.float64)
        self.rotation_mask = jnp.asarray(1 - translated, jnp.float64)
        u, chain = order.drift(spacing, translated)
        gauge = np.asarray(gauge, np.float64)
        oblique = constraint is not None
        constraint = np.asarray(constraint, np.float64) if oblique else gauge
        if parameters is not None:
            gauge, constraint = (
                m.reshape(views, parameters.full_dofs, -1)[:, local, :].reshape(
                    self.local_size, m.shape[-1]
                )
                for m in (gauge, constraint)
            )
            # A constant parallel lab-x shift is indistinguishable from a global
            # detector-u centre shift. Fix its local mean and let centre own it.
            # Cone centre and object translation differ with depth, so imposing
            # the same constraint there would remove a measurable motion.
            if not parameters.cone and "center_u" in parameters.globals and 3 in local:
                constant = np.zeros((views, self.nlocal))
                constant[:, local.index(3)] = 1
                gauge = np.column_stack([gauge, constant.reshape(-1)])
                constraint = np.column_stack([constraint, constant.reshape(-1)])
        g = _orthonormal_columns(constraint)
        tangents = _orthonormal_columns(gauge) if oblique else g
        # Gauge projection moves along the tangents until the constraints hold;
        # that needs constraints transversal to the tangents, one for each.
        cross = g.T @ tangents
        if not oblique or not cross.size:
            gauge_step = jnp.asarray(tangents)  # fixed by themselves: orthogonal
        elif cross.shape[0] != cross.shape[1] or np.linalg.cond(cross) > 1e8:
            condition = np.linalg.cond(cross) if cross.shape[0] == cross.shape[1] else np.inf
            raise ValueError(
                f"gauge constraints ({g.shape[1]} independent) do not fix the "
                f"{tangents.shape[1]} gauge tangents of this geometry (condition {condition:.3g})"
            )
        else:
            gauge_step = jnp.asarray(tangents @ np.linalg.inv(cross))
        residual = (
            chain.residual(g.reshape(views, self.nlocal, -1), np).reshape(g.shape)
            if chain is not None and g.shape[1]
            else g - u @ (u.T @ g)
        )
        outside = _orthonormal_columns(residual)
        rotation_precision = np.zeros((views, self.nlocal))
        if parameters is not None and "rotations" in parameters.fit:
            scales = np.broadcast_to(
                np.asarray(rotation_scales, np.float64), (views, parameters.full_dofs)
            )[:, local]
            radians = np.deg2rad(rotation_sigma_deg)
            rotation_precision = (float(noise_sigma) * scales / radians) ** 2 * (1 - translated)
        self.rotation_sigma_deg = float(rotation_sigma_deg)
        # Passed to the compiled functions as an argument, never captured: a captured array is
        # a constant of each executable, once per function.
        self.matrices = PriorMatrices(
            chain=jax.tree.map(jnp.asarray, chain),
            U=jnp.asarray(u),
            G=jnp.asarray(g),
            C=jnp.asarray(np.column_stack([u, outside])),
            gauge_step=gauge_step,
            rotation_precision=jnp.asarray(rotation_precision.reshape(-1), jnp.float64),
        )
        self._compiled = {
            name: jax.jit(getattr(self, f"_{name}"))
            for name in ("value", "weights", "apply", "innovation_solve", "project_gauge")
        }
        # A level writes the factor into buffers it owns (``_stacks``), donated and unread.
        self._compiled["factor"] = jax.jit(self._factor, donate_argnums=(3,), keep_unused=True)
        # The solve takes everything it needs from the factor, so it can run inside the Schur
        # solve's compiled program without bringing a matrix into it as a constant.
        self.solve = jax.jit(self._solve)

    def _factor_arguments(self):
        """The factor's arguments but the matrices, and its result, as shapes."""
        m = self.matrices
        p = jax.ShapeDtypeStruct((self.size,), m.U.dtype)
        w = jax.eval_shape(self._weights, m, p, 1.0)
        blocks = jax.ShapeDtypeStruct((self.n, self.k, self.k), m.U.dtype)
        return blocks, w, jax.eval_shape(self._factor, m, blocks, w)

    def factor_shapes(self):
        """The factor's arrays as shapes (``factor``), for a buffer to hold it."""
        with jax.enable_x64(self.matrices.U.dtype == jnp.float64):
            return self._factor_arguments()[2]

    def temporaries(self, name="factor") -> int:
        """The temporary bytes of ``factor`` (into a buffer) or ``innovation_solve``, compiled.

        Their calls reuse the compile.
        """
        with jax.enable_x64(self.matrices.U.dtype == jnp.float64):
            blocks, w, out = self._factor_arguments()
            last = out if name == "factor" else jax.ShapeDtypeStruct((self.size,), blocks.dtype)
            return temporaries(self._compiled[name], self.matrices, blocks, w, last)

    @property
    def U(self):
        """The Woodbury solver's explicit orthonormal drift columns."""
        return self.matrices.U

    @property
    def G(self):
        """The gauge constraints' rows (local rows by gauge directions)."""
        return self.matrices.G

    @property
    def C(self):
        """The explicit innovation constraints. The tree fixes drift internally."""
        return self.matrices.C

    @property
    def rotation_precision(self):
        """The rotation increments' size precision, per local row."""
        return self.matrices.rotation_precision

    def value(self, p, weight):
        """The prior's value at compact poses ``p`` with motion weight ``weight``."""
        return self._compiled["value"](self.matrices, p, weight)

    def weights(self, p, weight):
        """Per link and local column, the IRLS weights at ``p``."""
        return self._compiled["weights"](self.matrices, p, weight)

    def apply(self, p, w):
        """The prior's quadratic form at weights ``w`` applied to ``p``."""
        return self._compiled["apply"](self.matrices, p, w)

    def factor(self, blocks, w, out=None):
        """The pose normal system with the prior at weights ``w``, factored for ``solve``.

        Raises
        ------
        MemoryError
            If the device cannot allocate the factor workspace.
        """
        try:
            return self._compiled["factor"](self.matrices, blocks, w, out)
        except jax.errors.JaxRuntimeError as err:
            if "RESOURCE_EXHAUSTED" not in str(err):
                raise
            gib = self.temporaries("factor") / 2**30
            raise MemoryError(
                f"tj.align's motion prior needs one {gib:.2f} GiB block of device memory to "
                f"factor its pose system at {self.n} views "
                "and the device has none. Align fewer views or use a GPU with more memory"
            ) from err

    def innovation_solve(self, blocks, w, rhs):
        """The local innovation step with drift, gauge and globals held."""
        return self._compiled["innovation_solve"](self.matrices, blocks, w, rhs)

    def project_gauge(self, state):
        """``state`` with its gauge coordinates fixed by the constraints."""
        return self._compiled["project_gauge"](self.matrices, state)

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

    def _project_gauge(self, m, state):
        local = state[: self.local_size]
        local = local - m.gauge_step @ (m.G.T @ local)
        return jnp.concatenate([local, state[self.local_size :]])

    def _residual(self, m, p):
        if m.chain is None:
            return p - m.U @ (m.U.T @ p)
        return m.chain.residual(p.reshape(self.n, self.nlocal, 1)).reshape(-1)

    def _squared_jumps(self, m, p):
        residual = self._residual(m, p[: self.local_size]).reshape(self.n, self.nlocal)
        jumps = jnp.diff(residual, axis=0) * self.translation_mask * self.order.jump_scale[:, None]
        return jnp.sum(jumps**2, axis=1)

    def _weights(self, m, p, weight):
        """Per link and local column: robust for translations, constant for rotations."""
        if self.independent:  # no term ties a view to its neighbours
            return jnp.zeros_like(jnp.asarray(self.order.edge_scale)[:, None] * self.rotation_mask)
        weight, turn_weight = weight if isinstance(weight, tuple) else (weight, weight)
        edge = jnp.asarray(self.order.edge_scale)[:, None]
        robust = (2 * weight * self.delta / (self.delta**2 + self._squared_jumps(m, p)))[:, None]
        rotation = turn_weight / self.delta * self.rotation_mask
        return edge * (robust * self.translation_mask + rotation)

    def _value(self, m, p, weight):
        if self.independent:  # the rotations' size prior alone
            return 0.5 * jnp.sum(m.rotation_precision * p[: self.local_size] ** 2)
        weight, turn_weight = weight if isinstance(weight, tuple) else (weight, weight)
        jumps = weight * self.delta * jnp.sum(jnp.log1p(self._squared_jumps(m, p) / self.delta**2))
        local = p[: self.local_size].reshape(self.n, self.nlocal)
        turns = jnp.diff(local, axis=0) * self.rotation_mask * self.order.jump_scale[:, None]
        size = m.rotation_precision * p[: self.local_size] ** 2
        return jumps + 0.5 * (turn_weight / self.delta * jnp.sum(turns**2) + jnp.sum(size))

    def _laplacian(self, p, w):
        shape = p.shape
        p = p.reshape((self.n, self.nlocal, *shape[1:]))
        weights = w.reshape((*w.shape, *((1,) * (p.ndim - 2))))
        dif = jnp.diff(p, axis=0) * weights
        pad = [(0, 0)] * p.ndim
        pad[0] = (0, 1)
        left = jnp.pad(dif, pad)
        pad[0] = (1, 0)
        return (jnp.pad(dif, pad) - left).reshape(shape)

    def _apply(self, m, p, w):
        local = p[: self.local_size]
        jump = self._residual(m, self._laplacian(self._residual(m, local), w))
        return jnp.concatenate([jump + m.rotation_precision * local, jnp.zeros(self.nglobal)])

    def _local_factor(self, m, blocks, w, *, innovation=False):
        blocks = blocks[:, : self.nlocal, : self.nlocal]
        blocks = blocks + jax.vmap(jnp.diag)(m.rotation_precision.reshape(self.n, self.nlocal))
        if m.chain is None:
            u = m.U[:, :0] if innovation else m.U
            tri = _Tridiagonal.build(blocks, w)
            base = _Woodbury.build(tri, u, self._laplacian(u, w))
        else:
            base = _spline_solve.factor(m.chain, blocks, w, innovation)
        g = m.C if innovation else m.G
        hg = base.solve(g) if g.shape[1] else g
        gram = g.T @ hg
        chol = jnp.linalg.cholesky((gram + gram.T) * 0.5) if g.shape[1] else None
        return base, hg, chol, g

    def _local_solve(self, factor, rhs):
        base, hg, chol, g = factor
        result = base.solve(rhs)
        return result - hg @ cho_solve((chol, True), g.T @ result) if g.shape[1] else result

    def _factor(self, m, blocks, w, _out=None):
        """The factor, written into ``_out`` (donated, unread) when given."""
        local = self._local_factor(m, blocks, w) if self.nlocal else None
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

    def _innovation_solve(self, m, blocks, w, rhs):
        """Exact local innovation step with drift, rigid gauge and globals held."""
        if not self.nlocal:
            return jnp.zeros_like(rhs)
        factor = self._local_factor(m, blocks, w, innovation=True)
        ar = self._local_solve(factor, rhs[: self.local_size])
        return jnp.concatenate([ar, jnp.zeros(self.nglobal)])
