"""Exact bordered pose solves with translation drift/jumps and rotation size and continuity priors.

The local normal matrix is F + R + Q K Q: per-view data blocks F, physical
rotation precision R, and a difference Laplacian K, for translations outside
the drift spline U (Q = I - U U.T) and for rotations on their own increments.
Block cyclic reduction plus a spline Woodbury correction solves it with the
rigid gauge fixed. Shared calibration columns
form a small Schur border around that complete local system.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, lu_factor, lu_solve
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

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


def _tridiagonal(blocks, w):
    """Factor F + K, K the first-difference Laplacian with (links, columns) weights ``w``."""
    kdiag = jnp.pad(w, ((0, 1), (0, 0))) + jnp.pad(w, ((1, 0), (0, 0)))
    diagonal = jax.vmap(jnp.diag)(kdiag)
    return _pcr_factor(blocks + diagonal, -jnp.pad(w, ((0, 1), (0, 0))))


def _spline_design(steps, spacing):
    """Cubic B-spline design matrix (views, knots) with a knot every ``spacing`` steps.

    ``steps`` are the views' distances from the first, in units of the regular
    gap between views; consecutive views are one apart.
    """
    steps = np.asarray(steps, np.float64)
    end = float(steps[-1])
    # Integer steps land exactly on linspace's points, as consecutive views always have.
    sites = np.linspace(-1, 1, math.ceil(end) + 1)
    t = np.interp(steps, np.arange(len(sites)), sites)
    knots = list(spacing * np.arange(math.floor(end / spacing) + 1))
    if knots[-1] != end:
        knots.append(end)
    k = min(3, len(knots) - 1)
    at = np.interp(np.asarray(knots, np.float64), np.arange(len(sites)), sites)
    spline = make_interp_spline(at, np.zeros(len(knots)), k=k)
    return BSpline.design_matrix(t, spline.t, k).toarray()


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
            blocks.append(_spline_design(steps, spacing))
        design = np.zeros((self.n, sum(b.shape[1] for b in blocks)))
        row = column = 0
        for block in blocks:
            design[row : row + len(block), column : column + block.shape[1]] = block
            row, column = row + len(block), column + block.shape[1]
        return design

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
    is solved by block cyclic reduction and a spline Woodbury correction, with
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
        local = tuple(range(self.nlocal)) if parameters is None else parameters.local
        spacing = tuple(knot_spacing[i] for i in local)
        translated = (
            np.ones(self.nlocal)
            if parameters is None
            else np.asarray([i >= 3 for i in local], np.float64)
        )
        self.translation_mask = jnp.asarray(translated, jnp.float64)
        self.rotation_mask = jnp.asarray(1 - translated, jnp.float64)
        columns = [
            np.kron(order.spline_design(step), np.eye(self.nlocal)[:, [i]])
            for i, step in enumerate(spacing)
            if translated[i]
        ]
        u = (
            np.linalg.qr(np.concatenate(columns, axis=1), mode="reduced")[0]
            if columns
            else np.zeros((self.local_size, 0))
        )
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
            if not parameters.cone and "centre_u" in parameters.globals and 3 in local:
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
        outside = _orthonormal_columns(g - u @ (u.T @ g))
        rotation_precision = np.zeros((views, self.nlocal))
        if parameters is not None and "rotations" in parameters.fit:
            scales = np.broadcast_to(
                np.asarray(rotation_scales, np.float64), (views, parameters.full_dofs)
            )[:, local]
            radians = np.deg2rad(rotation_sigma_deg)
            rotation_precision = (float(noise_sigma) * scales / radians) ** 2 * (1 - translated)
        self.rotation_sigma_deg = float(rotation_sigma_deg)
        # Passed to the compiled functions as an argument, never captured: a captured array is
        # a constant of each executable, once per function (U and C grow as views squared).
        self.matrices = PriorMatrices(
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

    def device_bytes(self) -> dict[str, int]:
        """Device bytes the prior needs during a level, from its shapes: held and transient.

        Held through a step: its matrices (U and C, local rows by about views / 10 knots
        each, so they grow as views squared; G, the gauge step, the rotation precision) and
        the factor of the pose system it keeps (b = [U, K U] and its tridiagonal solve, each
        local rows by twice the knots, and the small LU), sized exactly from the factor's
        output shapes. Transient: the factor's temporaries, measured from 300 to 4000 views
        as five copies of b (its cyclic reduction) and a fixed 5 MB (the small solves'
        workspace), within 1%; planned as six copies and 8 MiB. Item 64 makes all of it
        linear.
        """
        m = self.matrices
        item, rows, knots = m.U.dtype.itemsize, self.local_size, 2 * m.U.shape[1]
        factor = self.factor_shapes()
        # Each array rounded up to 256 bytes, as the allocator aligns its buffers.
        held = sum(
            -(-int(a.size) * a.dtype.itemsize // 256) * 256 for a in (*m, *jax.tree.leaves(factor))
        )
        return {"held": held, "transient": 6 * item * rows * knots + 8 * 2**20}

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

    def temporaries(self) -> int:
        """The factor's temporary bytes, from its compile, which its calls into a buffer reuse."""
        with jax.enable_x64(self.matrices.U.dtype == jnp.float64):
            blocks, w, out = self._factor_arguments()
            return temporaries(self._compiled["factor"], self.matrices, blocks, w, out)

    @property
    def U(self):
        """The drift splines' orthonormal basis (local rows by knots)."""
        return self.matrices.U

    @property
    def G(self):
        """The gauge constraints' rows (local rows by gauge directions)."""
        return self.matrices.G

    @property
    def C(self):
        """The drift basis with the gauge rows it does not span."""
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
        """The pose normal system with the prior at weights ``w``, factored for ``solve``."""
        return self._compiled["factor"](self.matrices, blocks, w, out)

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
        return p - m.U @ (m.U.T @ p)

    def _squared_jumps(self, m, p):
        residual = self._residual(m, p[: self.local_size]).reshape(self.n, self.nlocal)
        jumps = jnp.diff(residual, axis=0) * self.translation_mask * self.order.jump_scale[:, None]
        return jnp.sum(jumps**2, axis=1)

    def _weights(self, m, p, weight):
        """Per link and local column: robust for translations, constant for rotations."""
        weight, turn_weight = weight if isinstance(weight, tuple) else (weight, weight)
        edge = jnp.asarray(self.order.edge_scale)[:, None]
        robust = (2 * weight * self.delta / (self.delta**2 + self._squared_jumps(m, p)))[:, None]
        rotation = turn_weight / self.delta * self.rotation_mask
        return edge * (robust * self.translation_mask + rotation)

    def _value(self, m, p, weight):
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

    def _local_factor(self, m, blocks, w):
        blocks = blocks + jax.vmap(jnp.diag)(m.rotation_precision.reshape(self.n, self.nlocal))
        tri = _tridiagonal(blocks, w)
        ku = self._laplacian(m.U, w)
        b = jnp.concatenate([m.U, ku], axis=1)
        k = m.U.shape[1]
        if k:
            zero, eye = jnp.zeros((k, k), jnp.float64), jnp.eye(k, dtype=jnp.float64)
            small = jnp.block([[zero, -eye], [-eye, -m.U.T @ ku]])
            ab = _pcr_solve(tri, b.reshape(self.n, self.nlocal, -1)).reshape(self.local_size, -1)
            small = small + b.T @ ab
            fac = lu_factor((small + small.T) * 0.5)
        else:
            ab, fac = b, None
        ag = _pcr_solve(tri, m.G.reshape(self.n, self.nlocal, -1)).reshape(m.G.shape)
        hg = ag - ab @ lu_solve(fac, b.T @ ag) if k else ag
        gauge = m.G.T @ hg
        chol = jnp.linalg.cholesky((gauge + gauge.T) * 0.5) if m.G.shape[1] else None
        return tri, b, ab, fac, hg, chol, m.G

    def _local_solve(self, factor, rhs):
        tri, b, ab, fac, hg, chol, g = factor
        ar = _pcr_solve(tri, rhs.reshape((self.n, self.nlocal, *rhs.shape[1:]))).reshape(rhs.shape)
        result = ar - ab @ lu_solve(fac, b.T @ ar) if b.shape[1] else ar
        if g.shape[1]:
            result = result - hg @ cho_solve((chol, True), g.T @ result)
        return result

    def _factor(self, m, blocks, w, _out=None):
        """The factor, written into ``_out`` (donated, unread) when given."""
        local = (
            self._local_factor(m, blocks[:, : self.nlocal, : self.nlocal], w)
            if self.nlocal
            else None
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

    def _innovation_solve(self, m, blocks, w, rhs):
        """Exact local innovation step with drift, rigid gauge and globals held."""
        if not self.nlocal:
            return jnp.zeros_like(rhs)
        blocks = blocks[:, : self.nlocal, : self.nlocal]
        blocks = blocks + jax.vmap(jnp.diag)(m.rotation_precision.reshape(self.n, self.nlocal))
        tri = _tridiagonal(blocks, w)
        ar = _pcr_solve(tri, rhs[: self.local_size].reshape(self.n, self.nlocal)).reshape(-1)
        if m.C.shape[1]:
            ac = _pcr_solve(tri, m.C.reshape(self.n, self.nlocal, -1)).reshape(m.C.shape)
            small = m.C.T @ ac
            chol = jnp.linalg.cholesky(0.5 * (small + small.T))
            ar = ar - ac @ cho_solve((chol, True), m.C.T @ ar)
        return jnp.concatenate([ar, jnp.zeros(self.nglobal)])
