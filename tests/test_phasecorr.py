import jax
import jax.numpy as jnp

from tomojax.utils.phasecorr import _wrap_shift, phase_corr_shift


def test_wrap_shift_maps_nyquist_boundary_to_negative_half_extent() -> None:
    assert int(_wrap_shift(4, 8)) == -4
    assert int(_wrap_shift(8, 16)) == -8
    assert int(_wrap_shift(16, 32)) == -16


def test_wrap_shift_preserves_middle_index_for_odd_lengths() -> None:
    assert int(_wrap_shift(2, 5)) == 2
    assert int(_wrap_shift(3, 7)) == 3


def test_phase_corr_shift_uses_half_open_nyquist_convention() -> None:
    ref = jnp.zeros((8, 8), dtype=jnp.float32)
    ref = ref.at[2, 1].set(1.0)
    tgt = jnp.roll(ref, shift=4, axis=1)

    du, dv = phase_corr_shift(ref, tgt)

    assert float(du) == -4.0
    assert float(dv) == 0.0


def test_phase_corr_shift_returns_scalar_float32_shifts() -> None:
    ref = jnp.zeros((8, 8), dtype=jnp.float32).at[2, 1].set(1.0)
    tgt = jnp.roll(ref, shift=1, axis=1)

    du, dv = phase_corr_shift(ref, tgt)

    assert du.shape == ()
    assert dv.shape == ()
    assert du.dtype == jnp.float32
    assert dv.dtype == jnp.float32


def test_phase_corr_shift_supports_vmap() -> None:
    ref = jnp.zeros((8, 8), dtype=jnp.float32).at[2, 1].set(1.0)
    tgt = jnp.roll(ref, shift=1, axis=1)
    refs = jnp.stack([ref, ref, ref], axis=0)
    tgts = jnp.stack([tgt, tgt, tgt], axis=0)

    du, dv = jax.vmap(phase_corr_shift)(refs, tgts)

    assert du.shape == (3,)
    assert dv.shape == (3,)
    assert du.dtype == jnp.float32
    assert dv.dtype == jnp.float32
    assert jnp.all(du == du[0])
    assert jnp.all(dv == dv[0])
