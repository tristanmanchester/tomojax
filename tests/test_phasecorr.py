import jax.numpy as jnp

from tomojax.utils.phasecorr import _wrap_shift, phase_corr_shift


def test_wrap_shift_maps_nyquist_boundary_to_negative_half_extent() -> None:
    assert _wrap_shift(4, 8) == -4
    assert _wrap_shift(8, 16) == -8
    assert _wrap_shift(16, 32) == -16


def test_phase_corr_shift_uses_half_open_nyquist_convention() -> None:
    ref = jnp.zeros((8, 8), dtype=jnp.float32)
    ref = ref.at[2, 1].set(1.0)
    tgt = jnp.roll(ref, shift=4, axis=1)

    du, dv = phase_corr_shift(ref, tgt)

    assert float(du) == -4.0
    assert float(dv) == 0.0
