from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import jax
import jax.numpy as jnp
import numpy as np

# check-public-imports: allow-private
from tomojax.align._lm_numerics import (
    finite_difference_jacobian,
)


def test_finite_difference_jacobian_matches_linear_operator() -> None:
    matrix = jnp.asarray(
        [[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]],
        dtype=jnp.float32,
    )

    def function(params: jax.Array) -> jax.Array:
        return matrix @ params

    jacobian = finite_difference_jacobian(
        function,
        jnp.asarray([0.2, -0.3, 0.4], dtype=jnp.float32),
        step_size=1.0e-3,
    )

    np.testing.assert_allclose(np.asarray(jacobian), np.asarray(matrix), atol=1.0e-4)
