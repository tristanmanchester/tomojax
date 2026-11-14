"""The scan's common offset and row ramp, which the per-view profile must not remove.

Each view's offset and row ramp are profiled out of the residual (``profile``),
but an image can produce exactly two of those directions: in parallel beam
every view of an image has the same mass, so the image can make a per-view
constant only if it is the same in every view, and likewise one shared row
ramp. Profiling them away left the image a free pedestal (the disc
c/sqrt(R^2 - r^2) along z), which a fit under mismatch filled and whose rim at
the detector's edge pulled every view's shift toward nominal. The level the
views share therefore belongs to the data, as before 45f9932d removed the
per-view gains.

The objective is the robust loss of each view's profiled residual plus
0.5 b^T G^-1 b, with b = [sum w r, sum w v r] over every measured pixel of every
view and G the Gram matrix of the whitened offset and ramp, [s, s v]. That term
is the squared norm of the whitened residual's part along the shared offset and
ramp, which the profile removes, so together they are |Pi' s r|^2 / 2 with
Pi' = Pi + P_C wherever no residual is in the log tail. b is linear in the image
and the poses, b = B x + D p - b_y, so Gauss-Newton carries it as two more
residual rows [B D]. They couple every view, which is why the pose block is
solved by Woodbury rather than by the prior's own factor alone.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def rows(nv):
    """The row coordinate the profile's ramp uses: -1 at the top row, +1 at the bottom."""
    return jnp.linspace(-1, 1, nv, dtype=jnp.float64)[:, None]


def view_grams(precision):
    """Each view's (g00, g01, g11), the Gram matrix of its whitened offset and ramp."""
    w, v = precision.astype(jnp.float64), rows(precision.shape[-2])
    return tuple(jnp.sum(w * v**k, axis=(-2, -1)) for k in (0, 1, 2))


def profiled_views(precision):
    """Whether the profile removes each view's offset and ramp: a view whose ramp is not
    distinct from its offset (one measured row, or none) is left whole."""
    g00, g01, g11 = view_grams(precision)
    return g00 * g11 - g01 * g01 > 1e-12 * jnp.maximum(g00 * g11, 1e-300)


def shared_weights(precision):
    """The precision of the views the profile removes a level from; the others carry
    their shared part already, unprofiled."""
    keep = profiled_views(precision)
    return precision * keep.reshape(keep.shape + (1, 1)).astype(precision.dtype)


def moments(residual, precision):
    """b = [sum w r, sum w v r] over the profiled views of residuals (view, v, u)."""
    w, r = shared_weights(precision).astype(jnp.float64), residual.astype(jnp.float64)
    v = rows(r.shape[-2])
    return jnp.stack([jnp.sum(w * r), jnp.sum(w * v * r)])


def metric(precision):
    """G, the Gram matrix of the whitened shared offset and ramp over the profiled views."""
    g00, g01, g11 = (jnp.sum(g) for g in view_grams(shared_weights(precision)))
    return jnp.array([[g00, g01], [g01, g11]])


class Common(NamedTuple):
    """The linearised common term at one image and pose state.

    ``images`` are B = [A^T w, A^T (w v)] summed over views (2, *volume),
    ``poses`` D, the moments' derivatives along the compact pose parameters
    (2, P), and ``moments`` b at the state.
    """

    gram: jax.Array
    inverse: jax.Array
    images: jax.Array
    poses: jax.Array
    moments: jax.Array

    def value(self, b=None):
        """0.5 b^T G^-1 b, at the state or at moments ``b``."""
        b = self.moments if b is None else b
        return 0.5 * jnp.vdot(b, self.inverse @ b)

    def image(self, m):
        """sum_i m_i B_i, a volume."""
        return (m[0] * self.images[0] + m[1] * self.images[1]).astype(jnp.float32)

    def of_image(self, v):
        """B v, the moments a volume step changes."""
        return jnp.stack([jnp.vdot(self.images[0], v), jnp.vdot(self.images[1], v)]).astype(
            jnp.float64
        )

    def image_gradient(self):
        return self.image(self.inverse @ self.moments)

    def pose_score(self):
        return self.poses.T @ (self.inverse @ self.moments)

    def diagonal(self):
        """The rows' contribution to the image Jacobi diagonal, diag(B^T G^-1 B)."""
        g, b = self.inverse, self.images
        return (g[0, 0] * b[0] ** 2 + 2 * g[0, 1] * b[0] * b[1] + g[1, 1] * b[1] ** 2).astype(
            jnp.float32
        )


class PoseSolve(NamedTuple):
    """(H + D^T G^-1 D)^-1 by Woodbury, given the prior's solve with H's factor.

    ``z`` holds H^-1 D^T as rows and ``m`` (G + D H^-1 D^T)^-1.
    """

    z: jax.Array
    m: jax.Array

    @classmethod
    def build(cls, solve, factor, common):
        z = jnp.stack([solve(factor, common.poses[0]), solve(factor, common.poses[1])])
        return cls(z, jnp.linalg.pinv(common.gram + common.poses @ z.T))

    def __call__(self, solve, factor, common, g):
        h = solve(factor, g)
        return h - self.z.T @ (self.m @ (common.poses @ h))
