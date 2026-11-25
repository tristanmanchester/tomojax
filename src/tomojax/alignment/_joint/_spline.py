"""An orthogonal multilevel decomposition of the sampled drift space.

A subtree settles the directions spanned by splines that end inside it.
Only restrictions of splines crossing its ends can still be undecided. Their
number is at most four cubic splines at each cut per translation, regardless
of the interval length.
"""

from __future__ import annotations

import math
from operator import index
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import ArrayLike


def sampled(steps: np.ndarray, spacing: int) -> sparse.csr_array:
    """Not-a-knot collocation, omitting knots too far from any sample to contribute."""
    end = steps[-1]
    if end == 0:
        return sparse.csr_array([[1.0]])
    last = math.ceil(end / spacing)
    degree = min(3, last)
    # Keep both complete cubic stencils even next to the omitted end sites.
    centres = (steps // spacing).astype(int)
    runs = np.split(centres, np.flatnonzero(np.diff(centres) > 12) + 1)
    inner = spacing * np.concatenate(
        [np.arange(max(2, r[0] - 5), min(last - 1, r[-1] + 7)) for r in runs]
    )
    # Use the same affine sampling coordinates as the original interpolating
    # spline, without allocating a sample for every missing acquisition step.
    count = math.ceil(end)

    def coordinate(x: ArrayLike) -> np.ndarray:
        i = np.floor(x)
        lo = -1 + i * (2 / count)
        hi = np.where(i + 1 == count, 1, -1 + (i + 1) * (2 / count))
        return np.where(i == count, 1, lo + (x - i) * (hi - lo))

    knots = np.r_[
        np.full(degree + 1, -1.0), coordinate(inner), np.full(degree + 1, coordinate(end))
    ]
    design = BSpline.design_matrix(coordinate(steps), knots, degree)
    design.eliminate_zeros()
    return design


def design_shape(design: sparse.csr_array | sparse.csr_matrix) -> tuple[int, int]:
    """The views by modes shape of a SciPy sparse spline design."""
    # SciPy sets the final shape in its sparse constructors, after base initialisation.
    shape: object = getattr(design, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("expected a two-dimensional sparse spline design")
    rows, cols = shape
    return index(rows), index(cols)


class Modes(NamedTuple):
    drift: jax.Array | np.ndarray
    rough: jax.Array | np.ndarray
    carry: jax.Array | np.ndarray


def _split(
    matrix: np.ndarray, inside: np.ndarray, tol: float | np.floating
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not a.shape[1]:
            return a, np.eye(len(a))
        u, s, _ = np.linalg.svd(a, full_matrices=True)
        rank = np.count_nonzero(s > tol)
        return u[:, :rank], u[:, rank:]

    # A confined spline can be declared free here. In its complement the
    # crossing splines identify exactly what still needs neighbouring views.
    drift, rest = basis(matrix[:, inside])
    across = matrix[:, ~inside]
    pending, rough = basis(rest.T @ across)
    carry = rest @ pending
    return (drift, rest @ rough, carry), carry.T @ across


class Drift(NamedTuple):
    views: jax.Array | np.ndarray
    unpack: jax.Array | np.ndarray
    levels: tuple[Modes, ...]

    @classmethod
    def build(cls, design: sparse.csr_array | sparse.csr_matrix, n: int, k: int) -> Drift:
        # The leaves cover contiguous, nearly equal intervals. No empty subtree.
        leaves = 1 << max(0, (math.ceil(n / 8) - 1).bit_length())
        groups = np.array_split(np.arange(n), leaves)
        width = len(groups[0])
        views = np.full((leaves, width), n, np.int32)
        design = design.tocsr()
        shape = design_shape(design)
        coo = design.tocoo()
        first, last = np.full(shape[1], n), np.full(shape[1], -1)
        np.minimum.at(first, coo.col, coo.row // k)
        np.maximum.at(last, coo.col, coo.row // k)
        tol = (
            np.finfo(float).eps
            * max(shape)
            * max(1, np.sqrt(np.asarray(design.power(2).sum(axis=0)).max(initial=0)))
        )
        # Each item carries an interval, its active knot IDs, their sampled
        # columns in child coordinates, and the occupied rows of the batch.
        pending = []
        for i, idx in enumerate(groups):
            views[i, : len(idx)] = idx
            start, stop = design.indptr[[idx[0] * k, (idx[-1] + 1) * k]]
            entries = slice(start, stop)
            ids, cols = np.unique(coo.col[entries], return_inverse=True)
            b = np.zeros((len(idx) * k, len(ids)))
            b[coo.row[entries] - idx[0] * k, cols] = coo.data[entries]
            pending.append((idx[0], idx[-1], ids, b, np.arange(len(b))))
        levels = []
        rows = width * k
        while True:
            modes, remaining = [], []
            row_ids = []
            for lo, hi, ids, b, valid in pending:
                row_ids.append(valid)
                inside = (first[ids] >= lo) & (last[ids] <= hi)
                parts, y = _split(b, inside, tol)
                modes.append(parts)
                remaining.append((lo, hi, ids[~inside], y))
            packed = []
            for columns in zip(*modes, strict=True):
                count = max(a.shape[1] for a in columns)
                table = np.zeros((len(columns), rows, count))
                for i, a in enumerate(columns):
                    table[i, row_ids[i], : a.shape[1]] = a
                packed.append(table)
            levels.append(Modes(*packed))
            if len(pending) == 1:
                return cls(views, np.flatnonzero(views.ravel() < n).astype(np.int32), tuple(levels))
            rows = 2 * packed[-1].shape[-1]
            pending = []
            for left, right in zip(remaining[::2], remaining[1::2], strict=True):
                ids = np.union1d(left[2], right[2])
                b = np.zeros((rows, len(ids)))
                for offset, (_, _, cols, a) in zip((0, rows // 2), (left, right), strict=True):
                    b[offset : offset + len(a), np.searchsorted(ids, cols)] = a
                # Padding of child carry coordinates must not become free modes.
                valid = np.r_[
                    np.arange(rows // 2) < len(left[3]), np.arange(rows // 2) < len(right[3])
                ]
                pending.append((left[0], right[1], ids, b[valid], np.flatnonzero(valid)))

    def residual[ArrayT: (jax.Array, np.ndarray)](self, x: ArrayT, xp: ModuleType = jnp) -> ArrayT:
        """Apply the orthogonal complement by a pair of tree sweeps."""
        cols = x.shape[-1]
        local = xp.pad(x, ((0, 1), (0, 0), (0, 0)))[self.views].reshape(len(self.views), -1, cols)
        saved = []
        for modes in self.levels:
            saved.append(modes.rough @ (modes.rough.mT @ local))
            local = modes.carry.mT @ local
            if len(local) > 1:
                local = local.reshape(len(local) // 2, 2 * local.shape[1], cols)
        for modes, rough in zip(reversed(self.levels), reversed(saved), strict=True):
            local = rough + modes.carry @ local.reshape(len(rough), modes.carry.shape[-1], cols)
        return local.reshape(-1, x.shape[1], cols)[self.unpack]
