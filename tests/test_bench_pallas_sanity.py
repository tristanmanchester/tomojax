from __future__ import annotations

import jax.numpy as jnp

from tomojax.bench import pallas_sanity


def test_pallas_sanity_relative_l2_handles_nonzero_reference() -> None:
    base = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    changed = base * jnp.float32(1.01)

    rel = pallas_sanity._rel_l2(changed, base)

    assert 0.009 < rel < 0.011


def test_pallas_sanity_volume_is_asymmetric() -> None:
    volume = pallas_sanity._make_volume(24)

    assert volume.shape == (24, 24, 24)
    assert float(volume.sum()) > 0.0
    assert float(volume.max()) > float(volume.min())
