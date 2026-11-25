"""Exact Gaussian elimination on the spline tree.

For each interval minimize over resolved modes, conditional on the unresolved
drift and innovation coordinates and a force on each endpoint innovation.
Joining two intervals introduces their newly resolved coordinates and one
internal force t: maximizing t sqrt(w) (r_right-r_left) - t²/2 supplies
exactly the missing jump energy, including when w is zero.

At a node let child coordinates be A z + B x (parent state z, new variables
x), and H the Hessian on (x,z). Eliminating x leaves the Schur complement on z.
T = A - B Hxx^-1 Hxz and J = B Hxx^-1 B.T carry right-hand sides: send T.T b
up the tree, and return J b + T z down it. At a leaf these maps give poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np

    from ._spline import Drift


def _columns(a: jax.Array | np.ndarray, start: int, width: int) -> jax.Array:
    """Embed a coordinate map at its columns of the joint system."""
    return jnp.pad(a, ((0, 0), (0, 0), (start, width - start - a.shape[-1])))


def _eliminate(
    h: jax.Array, nx: int, coordinates: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    a, coupling = h[:, :nx, :nx], h[:, :nx, nx:]
    # Pose coefficients and dual link forces have reciprocal physical
    # scalings. Equilibrate the pivot before factoring it so changing the
    # units of the whole loss does not change its condition number.
    scale = 1 / jnp.sqrt(jnp.abs(jnp.diagonal(a, axis1=1, axis2=2)))
    units = scale[:, :, None] * scale[:, None, :]
    inv = units * jnp.linalg.inv(units * a) if nx else a
    cross = inv @ coupling
    b, direct = coordinates[:, :, :nx], coordinates[:, :, nx:]
    transfer = direct - b @ cross
    response = (b @ inv) @ b.mT
    schur = h[:, nx:, nx:] - coupling.mT @ cross
    return schur, (transfer, response)


def factor(tree: Drift, blocks: jax.Array, w: jax.Array, innovation: bool | np.bool_) -> Factor:
    """Factor the pose normal system, or restrict it to innovation alone."""
    k = blocks.shape[-1]
    edge = jnp.pad(w, ((0, 1), (0, 0)))
    factors = []
    schur = None
    for depth, m in enumerate(tree.levels):
        d, r, c = m
        if innovation:
            d = d[:, :, :0]
        groups, rows, nc = c.shape
        nd, nr = d.shape[-1], r.shape[-1]
        nx = nd + nr + (k if depth else 0)
        nz = 2 * nc + 2 * k
        # Rows of these two maps are the child drift and innovation, respectively.
        width = nx + nz
        drift = _columns(d, 0, width) + _columns(c, nx, width)
        rough = _columns(r, nd, width) + _columns(c, nx + nc, width)
        pad = jnp.concatenate([jnp.sum(d * d, axis=1) == 0, jnp.sum(r * r, axis=1) == 0], axis=1)
        if schur is None:
            coordinates = drift + rough
            views = tree.views
            poses = coordinates.reshape(groups, -1, k, nx + nz)
            residual = rough.reshape(poses.shape)
            f = jnp.pad(blocks, ((0, 1), (0, 0), (0, 0)))[views]
            h = coordinates.mT @ (f @ poses).reshape(coordinates.shape)
            steps = jnp.diff(residual, axis=1)
            weights = jnp.pad(edge, ((0, 1), (0, 0)))[views[:, :-1]]
            valid = views[:, 1:] < len(blocks)
            h += jnp.einsum("nvki,nvk,nvkj->nij", steps, weights * valid[..., None], steps)
            end = jnp.sum(views < len(blocks), axis=1) - 1
            endpoints = jnp.concatenate([residual[:, 0], residual[jnp.arange(groups), end]], axis=1)
            forces = jnp.pad(endpoints, ((0, 0), (nx + 2 * nc, 0), (0, 0)))
            h += forces + forces.mT
        else:
            # Endpoint forces shared by a child pair have opposite signs.
            forces = jnp.broadcast_to(jnp.eye(2 * k), (groups, 2 * k, 2 * k))
            forces = _columns(forces, nx + 2 * nc, width)
            split = tree.views.reshape(groups, 2, -1)[:, 1, 0] - 1
            shared = jnp.sqrt(edge[split])[:, :, None] * jnp.eye(k)
            shared = _columns(shared, nd + nr, width)
            left, right = [
                jnp.concatenate([drift[:, sl], rough[:, sl], lo, hi], axis=1)
                for sl, lo, hi in (
                    (slice(None, rows // 2), forces[:, :k], -shared),
                    (slice(rows // 2, None), shared, forces[:, k:]),
                )
            ]
            coordinates = jnp.concatenate([left, right], axis=1)
            h = left.mT @ schur[::2] @ left + right.mT @ schur[1::2] @ right
            pad = jnp.concatenate([pad, -jnp.ones((groups, k))], axis=1)
        h += jnp.pad(pad, ((0, 0), (0, nz)))[:, :, None] * jnp.eye(nx + nz)
        schur, maps = _eliminate(h, nx, coordinates)
        factors.append(maps)
    return Factor(tuple(factors), tree.views, tree.unpack)


class Factor(NamedTuple):
    maps: tuple[tuple[jax.Array, jax.Array], ...]
    views: jax.Array | np.ndarray
    unpack: jax.Array | np.ndarray

    def solve(self, rhs: jax.Array) -> jax.Array:
        """Apply the inverse using precomputed transfers and local responses."""
        maps, views, unpack = self
        cols = 1 if rhs.ndim == 1 else rhs.shape[1]
        k = maps[0][0].shape[1] // views.shape[1]
        x = jnp.pad(rhs.reshape(-1, k, cols), ((0, 1), (0, 0), (0, 0)))[views]
        saved = []
        for transfer, response in maps:
            x = x.reshape(len(transfer), transfer.shape[1], cols)
            saved.append(response @ x)
            x = transfer.mT @ x
        x = jnp.zeros_like(x)
        for (transfer, _), direct in zip(reversed(maps), reversed(saved), strict=True):
            x = direct + transfer @ x.reshape(len(transfer), transfer.shape[-1], cols)
        return x.reshape(-1, k, cols)[unpack].reshape(rhs.shape)
