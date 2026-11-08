"""Independent coefficient tangents and image/pose adjoints of shared calibration."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment._joint._operators import Operators
from tomojax.alignment._joint._parameters import Parameters  # check-public-imports: allow-private
from tomojax.alignment._joint._projection import PackedMask  # check-public-imports: allow-private
from tomojax.core.cone import beam_frame, frame_coefficients
from tomojax.core.joseph import plane_coefficients
from tomojax.geometry import ConeBeam, Detector, Grid, LaminographyGeometry, stack_view_poses

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize("fit", ["centre", "translations", ("centre", "translations")])
def test_affine_parallel_cache_matches_physical_coefficients_and_active_tangents(fit):
    grid = Grid(16, 15, 12, 0.7, 0.8, 1.1)
    detector = Detector(48, 48, 0.7, 0.9, center=(0.3, -0.2))
    nominal = np.asarray(
        stack_view_poses(
            LaminographyGeometry(
                grid, detector, np.linspace(0.3, 360.3, 60, endpoint=False), tilt_deg=25
            ),
            60,
        )
    )
    parameters = Parameters(60, fit, lever=16)
    rng = np.random.default_rng(2153)
    mask = rng.uniform(size=(60, 48, 48)) > 0.05
    ops = Operators(
        grid,
        detector,
        nominal,
        lever=16,
        native=(0.35, 0.45),
        parameters=parameters,
        detector_roll_deg=11,
        detector_pivot=(0.65, 0.25),
        ray_binning=(2, 2),
        pixel_mask=PackedMask(jnp.packbits(mask, axis=-1, bitorder="little"), (48, 48)),
    )
    assert ops.affine is not None
    state = rng.normal(0, 0.3, parameters.size)
    q = ops.pad_poses(state)
    cached = ops.prepare(q)
    direct = ops.coefficients(ops.nominal, q, ops.valid, ops.scales, pixel_mask=ops.pixel_mask)
    np.testing.assert_allclose(cached.rays, direct.rays, rtol=1e-6, atol=5e-6)
    derivative = jax.jit(ops._tangent)(ops.nominal, q, ops.valid, ops.scales)
    np.testing.assert_allclose(ops.affine.tangent, derivative, rtol=1e-6, atol=1e-6)
    cached_derivative = jax.jit(ops._tangent)(ops.affine, q, ops.valid, ops.scales)
    np.testing.assert_allclose(cached_derivative, derivative, rtol=1e-6, atol=1e-6)
    assert cached.pixel_mask is ops.pixel_mask


@pytest.mark.parametrize("cone", [False, True])
def test_global_coefficient_tangents_match_calibration_of_the_original_detector(cone):
    """Cropping must preserve the original roll pivot, including in derivative columns."""
    n = 60
    grid = Grid(24, 23, 22, 0.7, 0.9, 1.1, vol_center=(0.2, -0.1, 0.3))
    original = Detector(49, 49, 0.7, 1.1, center=(0.3, -0.2))
    detector = replace(
        original,
        nu=48,
        nv=48,
        center=(original.center[0] - original.du / 2, original.center[1] - original.dv / 2),
    )
    nominal = np.asarray(
        stack_view_poses(
            LaminographyGeometry(
                grid, original, np.linspace(0.37, 360.37, n, endpoint=False), tilt_deg=25
            ),
            n,
        )
    )
    parameters = Parameters(n, ("centre", "tilt", "roll"), cone=cone, lever=17)
    beam = ConeBeam(40, 65, detector_roll_deg=4, detector_pitch_deg=-3, detector_yaw_deg=2)
    original_frame = beam_frame(beam, original).astype(np.float64)
    cropped = original_frame.copy()
    cropped[0] -= 0.5 * (original.du * cropped[1] + original.dv * cropped[2])
    frames = np.broadcast_to(cropped, (n, 4, 3))
    ops = Operators(
        grid,
        detector,
        nominal,
        lever=17,
        native=(original.du, original.dv),
        parameters=parameters,
        frames=frames if cone else None,
        detector_roll_deg=3 if not cone else 0,
        detector_pivot=original.center,
    )
    values = {"centre_u": 0.6, "centre_v": -0.4, "tilt": 0.2, "roll": 0.25}
    state = np.array([values[name] for name in parameters.globals])

    def reference(p):
        # Calibrate the physical, original detector first and only then take
        # its first 48 x 48 pixels. No solver pose/frame transform is used.
        g = dict(zip(parameters.globals, p, strict=True))
        tilt, roll = g["tilt"] / 17, g["roll"] / 17
        poses = nominal.copy()
        poses[:, :3, :3] = Rotation.from_rotvec([-tilt, 0, 0]).as_matrix() @ nominal[:, :3, :3]
        if cone:
            frame = original_frame.copy()
            frame[1] = np.cos(roll) * original_frame[1] + np.sin(roll) * original_frame[2]
            frame[2] = -np.sin(roll) * original_frame[1] + np.cos(roll) * original_frame[2]
            frame[0] += [g["centre_u"] * original.du, 0, g["centre_v"] * original.dv]
            frame[0] -= 0.5 * (original.du * frame[1] + original.dv * frame[2])
            return np.asarray(frame_coefficients(jnp.asarray(poses), frame, grid, detector))[:, :12]
        roll += np.deg2rad(3)
        u, v = np.meshgrid(
            (np.arange(48) - 24) * original.du,
            (np.arange(48) - 24) * original.dv,
        )
        x = np.cos(roll) * u - np.sin(roll) * v + original.center[0] + g["centre_u"] * original.du
        z = np.sin(roll) * u + np.cos(roll) * v + original.center[1]
        return np.asarray(plane_coefficients(jnp.asarray(poses), grid, detector, (x, z)))

    rows = ops.pad_poses(state)[:n]
    valid = jnp.ones(n, jnp.bool_)
    actual = np.asarray(
        ops.coefficients(
            ops.nominal[:n], rows, valid, ops.scales[:n], ops.frames[:n] if cone else None
        ).rays
    )
    tangent = np.asarray(
        jax.jit(ops._tangent)(
            ops.nominal[:n], rows, valid, ops.scales[:n], ops.frames[:n] if cone else None
        )
    )
    if cone:
        actual, tangent = actual[:, :12], tangent[..., :12]
    np.testing.assert_allclose(actual, reference(state), rtol=3e-6, atol=1e-5)
    for coordinate in range(parameters.nglobal):
        step = np.zeros(parameters.size)
        step[coordinate] = 0.02
        expected = (reference(state + step) - reference(state - step)) / 0.04
        relative = np.linalg.norm(tangent[:, coordinate] - expected) / np.linalg.norm(expected)
        assert relative < 0.004


def test_shared_global_image_pose_cross_block_has_its_exact_transpose():
    """The global score sums views while the matching backprojection broadcasts it."""
    n = 60
    grid = Grid(16, 16, 12, 1, 1, 1)
    detector = Detector(48, 48, 0.5, 0.7, center=(0.2, -0.3))
    nominal = np.asarray(
        stack_view_poses(
            LaminographyGeometry(
                grid, detector, np.linspace(0.31, 360.31, n, endpoint=False), tilt_deg=23
            ),
            n,
        )
    )
    parameters = Parameters(n, ("translations", "centre", "roll"), lever=16)
    rng = np.random.default_rng(1129)
    finite = rng.uniform(size=(n, 48, 48)) > 0.05
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=16,
            native=(0.5, 0.7),
            parameters=parameters,
            pixel_mask=PackedMask(jnp.packbits(finite, axis=-1, bitorder="little"), (48, 48)),
        )
        state = jnp.asarray(rng.normal(0, 0.05, parameters.size))
        packet = ops.prepare(ops.pad_poses(state))
        x = jnp.asarray(rng.normal(size=ops.shape), jnp.float32)
        sqrtweights = ops.pad_images(rng.uniform(0.1, 1, size=(n, 48, 48)))
        columns = jnp.asarray(rng.normal(size=(ops.padded_n, parameters.k, 48, 48)), jnp.float32)
        step = jnp.asarray(rng.normal(size=parameters.size))
        _, score = ops.weighted_forward(packet, x, sqrtweights, columns)
        transpose = ops.pose_back(packet, sqrtweights, columns, step)
        left, right = (
            jnp.vdot(step, score),
            jnp.vdot(x.astype(jnp.float64), transpose.astype(jnp.float64)),
        )
        np.testing.assert_allclose(left, right, rtol=5e-6, atol=3e-5)
