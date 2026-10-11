import jax
import jax.numpy as jnp
import numpy as np

from tomojax.forward.api import pseudo_huber_loss


def test_pseudo_huber_keeps_small_residuals_and_its_gradient():
    r = jnp.asarray([0.0, 1e-4, -0.5, 3.0, -1e3], jnp.float32)
    reference = 2.0**2 * (np.sqrt(1 + (np.asarray(r, np.float64) / 2.0) ** 2) - 1)
    # The direct formula gives 0 in float32 at r = 1e-4; this one does not.
    np.testing.assert_allclose(pseudo_huber_loss(r, delta=2.0), reference, rtol=1e-6)
    slope = jax.vmap(jax.grad(lambda x: pseudo_huber_loss(x, delta=2.0)))(r)
    np.testing.assert_allclose(slope, r / np.sqrt(1 + (r / 2.0) ** 2), rtol=1e-6)
