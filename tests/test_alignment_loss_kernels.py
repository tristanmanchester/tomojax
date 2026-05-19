from __future__ import annotations

import jax.numpy as jnp
import pytest

# check-public-imports: allow-private
from tomojax.align._objectives.loss_kernels import _loss_barron

# check-public-imports: allow-private
from tomojax.align._objectives.loss_state import LossState


def test_barron_loss_uses_general_alpha_coefficient() -> None:
    pred = jnp.full((2, 2), 2.0, dtype=jnp.float32)
    target = jnp.ones((2, 2), dtype=jnp.float32)
    alpha = 1.5
    beta = abs(alpha - 2.0)
    expected_per_element = (beta / alpha) * ((1.0 + 1.0 / beta) ** (alpha / 2.0) - 1.0)

    actual = _loss_barron(pred, target, LossState(kind="barron", params={"alpha": alpha, "c": 1.0}))

    assert float(actual) == pytest.approx(expected_per_element * 4.0)
