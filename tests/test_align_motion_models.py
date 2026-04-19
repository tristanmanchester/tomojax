from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from tomojax.align.motion_models import (
    build_pose_motion_model,
    expand_motion_coefficients,
    fit_motion_coefficients,
    scan_coordinate_from_geometry,
)


class _GeometryWithAngles:
    thetas_deg = np.linspace(0.0, 180.0, 10, endpoint=False)


def _params(n_views: int) -> jnp.ndarray:
    values = np.zeros((n_views, 5), dtype=np.float32)
    values[:, 0] = np.linspace(-0.2, 0.2, n_views, dtype=np.float32)
    values[:, 1] = np.linspace(0.3, -0.1, n_views, dtype=np.float32)
    values[:, 2] = np.linspace(-0.5, 0.5, n_views, dtype=np.float32)
    values[:, 3] = np.sin(np.linspace(0.0, np.pi, n_views)).astype(np.float32)
    values[:, 4] = np.cos(np.linspace(0.0, np.pi, n_views)).astype(np.float32)
    return jnp.asarray(values)


def test_polynomial_motion_model_expands_to_per_view_params_shape():
    n_views = 8
    init = _params(n_views)
    model = build_pose_motion_model(
        pose_model="polynomial",
        n_views=n_views,
        active_dofs=("alpha", "beta", "phi", "dx", "dz"),
        frozen_params5=jnp.zeros_like(init),
        degree=2,
    )

    coeffs = fit_motion_coefficients(model, init)
    expanded = expand_motion_coefficients(model, coeffs)

    assert coeffs.shape == (3, 5)
    assert expanded.shape == (n_views, 5)


def test_spline_motion_model_expands_to_per_view_params_shape():
    n_views = 10
    init = _params(n_views)
    scan = scan_coordinate_from_geometry(_GeometryWithAngles(), n_views)
    model = build_pose_motion_model(
        pose_model="spline",
        n_views=n_views,
        active_dofs=("alpha", "beta", "phi", "dx", "dz"),
        frozen_params5=jnp.zeros_like(init),
        scan_coordinate=scan,
        knot_spacing=3,
        degree=3,
    )

    coeffs = fit_motion_coefficients(model, init)
    expanded = expand_motion_coefficients(model, coeffs)

    assert coeffs.shape == (4, 5)
    assert expanded.shape == (n_views, 5)


def test_smooth_motion_model_uses_fewer_variables_than_per_view():
    n_views = 10
    init = jnp.zeros((n_views, 5), dtype=jnp.float32)
    per_view = build_pose_motion_model(
        pose_model="per_view",
        n_views=n_views,
        active_dofs=("dx", "dz"),
        frozen_params5=init,
    )
    spline = build_pose_motion_model(
        pose_model="spline",
        n_views=n_views,
        active_dofs=("dx", "dz"),
        frozen_params5=init,
        knot_spacing=5,
        degree=3,
    )

    assert spline.variable_count < per_view.variable_count
    assert spline.variable_count == 6
    assert per_view.variable_count == 20


def test_smooth_motion_model_preserves_frozen_dof_columns():
    n_views = 9
    frozen = _params(n_views)
    target = frozen.at[:, 3].set(jnp.linspace(-1.0, 1.0, n_views))
    model = build_pose_motion_model(
        pose_model="polynomial",
        n_views=n_views,
        active_dofs=("dx",),
        frozen_params5=frozen,
        degree=2,
    )

    expanded = expand_motion_coefficients(model, fit_motion_coefficients(model, target))

    np.testing.assert_array_equal(np.asarray(expanded[:, :3]), np.asarray(frozen[:, :3]))
    np.testing.assert_array_equal(np.asarray(expanded[:, 4]), np.asarray(frozen[:, 4]))
    assert model.variable_count == 3
    assert model.per_view_variable_count == n_views
