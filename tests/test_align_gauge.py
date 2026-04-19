from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.align.dofs import DOF_INDEX
from tomojax.align.gauge import (
    apply_alignment_gauge,
    normalize_gauge_fix,
    validate_alignment_gauge_feasible,
)


def _bounds(
    *,
    dx: tuple[float, float] = (-np.inf, np.inf),
    dz: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lower = np.full((5,), -np.inf, dtype=np.float32)
    upper = np.full((5,), np.inf, dtype=np.float32)
    lower[DOF_INDEX["dx"]], upper[DOF_INDEX["dx"]] = dx
    lower[DOF_INDEX["dz"]], upper[DOF_INDEX["dz"]] = dz
    return jnp.asarray(lower), jnp.asarray(upper)


def test_mean_translation_gauge_zeroes_active_translation_means():
    params5 = jnp.asarray(
        [
            [0.1, 0.2, 0.3, 1.0, -3.0],
            [0.0, 0.1, 0.2, 2.0, -1.0],
            [0.2, 0.0, 0.1, 4.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    lower, upper = _bounds()

    gauged, stats = apply_alignment_gauge(
        params5,
        mode="mean_translation",
        active_mask=(True, True, True, True, True),
        bounds_lower=lower,
        bounds_upper=upper,
    )

    np.testing.assert_allclose(np.asarray(gauged[:, :3]), np.asarray(params5[:, :3]))
    assert float(jnp.mean(gauged[:, DOF_INDEX["dx"]])) == pytest.approx(0.0, abs=1e-6)
    assert float(jnp.mean(gauged[:, DOF_INDEX["dz"]])) == pytest.approx(0.0, abs=1e-6)
    assert stats["mode"] == "mean_translation"
    assert stats["dofs"] == ["dx", "dz"]


def test_none_gauge_preserves_parameters():
    params5 = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    lower, upper = _bounds()

    gauged, stats = apply_alignment_gauge(
        params5,
        mode="none",
        active_mask=(True, True, True, True, True),
        bounds_lower=lower,
        bounds_upper=upper,
    )

    np.testing.assert_array_equal(np.asarray(gauged), np.asarray(params5))
    assert stats["mode"] == "none"
    assert stats["dofs"] == []


def test_mean_translation_gauge_preserves_inactive_translation_columns():
    params5 = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 4.0, -3.0],
            [0.0, 0.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 0.0, 6.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    lower, upper = _bounds()

    gauged, _ = apply_alignment_gauge(
        params5,
        mode="mean_translation",
        active_mask=(True, True, True, False, True),
        bounds_lower=lower,
        bounds_upper=upper,
    )

    np.testing.assert_array_equal(np.asarray(gauged[:, DOF_INDEX["dx"]]), [4.0, 5.0, 6.0])
    assert float(jnp.mean(gauged[:, DOF_INDEX["dz"]])) == pytest.approx(0.0, abs=1e-6)


def test_mean_translation_gauge_respects_feasible_bounds():
    params5 = jnp.zeros((3, 5), dtype=jnp.float32)
    params5 = params5.at[:, DOF_INDEX["dx"]].set(jnp.asarray([-4.0, 0.25, 5.0]))
    lower, upper = _bounds(dx=(-1.0, 2.0))

    gauged, _ = apply_alignment_gauge(
        params5,
        mode="mean_translation",
        active_mask=(True, True, True, True, False),
        bounds_lower=lower,
        bounds_upper=upper,
    )

    dx = np.asarray(gauged[:, DOF_INDEX["dx"]])
    assert np.min(dx) >= -1.0 - 1e-6
    assert np.max(dx) <= 2.0 + 1e-6
    assert float(np.mean(dx)) == pytest.approx(0.0, abs=1e-6)


def test_mean_translation_gauge_rejects_infeasible_active_bounds():
    lower, upper = _bounds(dx=(1.0, 2.0))

    with pytest.raises(ValueError, match="requires active dx bounds to include 0"):
        validate_alignment_gauge_feasible(
            mode="mean_translation",
            active_mask=(True, True, True, True, True),
            bounds_lower=lower,
            bounds_upper=upper,
        )


def test_normalize_gauge_fix_aliases_and_rejects_unknown_modes():
    assert normalize_gauge_fix("mean-translation") == "mean_translation"
    assert normalize_gauge_fix("off") == "none"
    with pytest.raises(ValueError, match="gauge_fix must be one of"):
        normalize_gauge_fix("anchor")
