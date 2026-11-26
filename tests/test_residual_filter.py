"""The alignment compares model and data through K = [1, 2, 1] / 4 along u and v."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint._projection import filtered_precision, residual_filter
from tomojax.geometry import stack_view_poses


def test_the_filter_is_its_own_transpose():
    # The objective's gradient applies K^T; with the edge replicated K is symmetric.
    rng = np.random.default_rng(0)
    a, b = (jnp.asarray(rng.standard_normal((3, 7, 9))) for _ in range(2))
    with jax.enable_x64(True):
        a, b = a.astype(jnp.float64), b.astype(jnp.float64)
        left = float(jnp.vdot(residual_filter(a), b))
        right = float(jnp.vdot(a, residual_filter(b)))
    assert left == pytest.approx(right, rel=1e-12)


def test_the_filter_gives_the_same_on_the_host_as_on_the_device():
    # Each level's data are filtered on the host (a native scan does not fit on the device
    # whole); the fit's projections are filtered on the device. Both must be the same K.
    images = np.random.default_rng(1).standard_normal((4, 5, 6)).astype(np.float32)
    host = residual_filter(images)
    assert isinstance(host, np.ndarray) and host.dtype == np.float32
    np.testing.assert_allclose(host, np.asarray(residual_filter(jnp.asarray(images))), rtol=1e-6)


def test_a_projection_alternating_pixel_to_pixel_costs_nothing_and_a_smooth_one_is_kept():
    # At the Nyquist frequency a sampled model cannot be right (the alias is as large as the
    # signal), so the comparison ignores it; at zero frequency, and for any linear trend away
    # from the edge, it is the plain comparison.
    v, u = np.mgrid[:8, :10].astype(np.float64)
    alternating = (-1.0) ** (v + u)
    with jax.enable_x64(True):
        filtered = np.asarray(residual_filter(jnp.asarray(alternating[None])))[0]
        np.testing.assert_allclose(filtered[1:-1, 1:-1], 0.0, atol=1e-15)
        along_u = np.asarray(residual_filter(jnp.asarray(((-1.0) ** u)[None])))[0]
        np.testing.assert_allclose(along_u[:, 1:-1], 0.0, atol=1e-15)
        ramp = 0.3 * v - 0.7 * u + 2.0
        kept = np.asarray(residual_filter(jnp.asarray(ramp[None])))[0]
        np.testing.assert_allclose(kept[1:-1, 1:-1], ramp[1:-1, 1:-1], rtol=1e-14, atol=1e-14)
        level = np.asarray(residual_filter(jnp.full((1, 8, 10), 1.5)))[0]
        np.testing.assert_allclose(level, 1.5, rtol=1e-15)


def test_filtered_data_are_unknown_wherever_a_tap_was_not_measured():
    # A filtered sample mixes its 3 x 3 neighborhood, so one unmeasured pixel removes
    # the nine samples around it; partial precision elsewhere is kept as it was.
    precision = np.full((2, 6, 7), 0.75, np.float32)
    precision[0, 2, 3] = 0.0
    expected = precision.copy()
    expected[0, 1:4, 2:5] = 0.0
    np.testing.assert_array_equal(filtered_precision(precision), expected)
    edge = np.ones((1, 4, 4), np.float32)
    edge[0, 0, 0] = 0.0
    out = filtered_precision(edge)
    assert out[0, :2, :2].sum() == 0 and out[0, 2:, :].min() == 1 and out[0, :, 2:].min() == 1
    assert filtered_precision(None) is None


@pytest.mark.numerical
@pytest.mark.slow
def test_thin_plates_on_a_half_turn_recover_their_motion():
    # Three plates thinner than a working voxel (0.48-0.80 voxels), with holes and teeth, on a
    # grid four times as fine, projected and binned: a sharp, flat object the working model
    # can only alias. On a half turn, moving in u and v with waves, a trend and a slip, the
    # plain comparison fitted this at 0.45-0.56 px of landmark error at the default budget;
    # through K it is 0.10-0.16 px.
    n, views, f = 32, 120, 128
    rng = np.random.default_rng(3)
    z, y, x = (np.mgrid[:f, :f, :f] - (f - 1) / 2) / f
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    fine = np.zeros((f, f, f), np.float32)
    for z0, thick in ((-0.15, 0.02), (0.0, 0.015), (0.15, 0.025)):
        plate = (np.abs(z - z0) < thick / 2) & (r < 0.46)
        plate &= ~((r > 0.42) & (np.cos(24 * theta) < 0))
        for _ in range(6):
            c, hole = rng.uniform(-0.3, 0.3, 2), rng.uniform(0.02, 0.07)
            plate &= np.hypot(x - c[0], y - c[1]) > hole
        fine += rng.uniform(0.5, 1.0) * plate
    del z, y, x, r, theta
    angles = np.linspace(0, 180, views, endpoint=False)
    t, ramp = np.deg2rad(angles), np.linspace(0, 1, views)
    slip = (np.arange(views) >= views // 3).astype(float)
    table = np.zeros((views, 6))
    for column, phase, sign in ((3, 0.3, 1), (4, 1.9, -1)):
        waves = 0.6 * np.cos(2 * t + phase) + 0.4 * np.cos(3 * t + 2 * phase)
        table[:, column] = waves + 0.8 * ramp + 0.7 * sign * slip
    h = n / f
    true = tj.Scan(
        np.zeros((views, f, f), np.float32),
        tj.ParallelGeometry(tj.Grid(f, f, f, h, h, h), tj.Detector(f, f, h, h), angles),
    ).with_poses(table)
    data = np.asarray(tj.project(true.geometry, fine))
    data = data.reshape(views, n, 4, n, 4).mean((2, 4)).astype(np.float32)
    declared = tj.ParallelGeometry(tj.Grid(n, n, n, 1, 1, 1), tj.Detector(n, n, 1, 1), angles)
    result = tj.align(tj.Scan(data, declared), fit="translations")
    estimated = np.asarray(stack_view_poses(result.scan.geometry, views))
    truth = np.asarray(stack_view_poses(true.geometry, views))
    assert _landmark_error(estimated, truth, n) < 0.25


def _landmark_error(estimated, truth, n):
    """RMS detector-pixel error of eight landmarks after one common rigid motion."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    marks = np.stack(np.meshgrid(*[[-0.3, 0.3]] * 3, indexing="ij"), -1).reshape(-1, 3) * n

    def pixels(m, p):
        return (np.einsum("nij,kj->nki", m[:, :3, :3], p) + m[:, None, :3, 3])[..., [0, 2]]

    target = pixels(truth, marks)

    def residual(par):
        moved = marks @ Rotation.from_rotvec(par[:3]).as_matrix().T + par[3:]
        return (pixels(estimated, moved) - target).ravel()

    fit = least_squares(residual, np.zeros(6))
    return float(np.sqrt(np.mean(residual(fit.x).reshape(-1, 2) ** 2) * 2))
