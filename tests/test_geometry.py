import sys
import numpy as np
import pytest

from tomojax.core.geometry import (
    Grid,
    Detector,
    ParallelGeometry,
    LaminographyGeometry,
)
from tomojax.core.geometry.lamino import laminography_tilt_matrix
from tomojax.core.geometry.transforms import rotz, rot_axis_angle, exp_se3, invert, compose
import numpy as np


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def test_parallel_pose_identity_and_quarter_turn():
    grid = Grid(nx=32, ny=32, nz=32, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 90.0])

    T0 = np.array(geom.pose_for_view(0))
    assert np.allclose(T0, np.eye(4))

    T1 = np.array(geom.pose_for_view(1))
    Rz = rotz(np.pi / 2.0)
    assert np.allclose(T1, Rz, atol=1e-7)


def test_parallel_rays_direction_and_pixel_centers():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=2.0, dv=2.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0])
    origin_fn, dir_fn = geom.rays_for_view(0)

    # Center pixel (u=1, v=1) in 0-index coordinates for 4x4: centers at -3,-1,1,3
    o = origin_fn(1, 1)
    d = dir_fn(1, 1)
    assert np.allclose(d, (0.0, 1.0, 0.0))
    assert np.isclose(o[0], -1.0) and np.isclose(o[2], -1.0)


def test_lamino_axis_tilt_and_pose():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=[0.0, 10.0], tilt_deg=30.0, tilt_about="x")

    ax = geom._axis_unit()
    exp_ax = np.array([0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))])
    assert np.allclose(ax, exp_ax, atol=1e-7)

    T0 = np.array(geom.pose_for_view(0))
    # At theta=0 the pose equals the fixed alignment S that maps e_z to axis
    ez = np.array([0.0, 0.0, 1.0, 1.0])
    t0_ez = T0 @ ez
    assert np.allclose(t0_ez[:3], ax, atol=1e-6)

    T1 = np.array(geom.pose_for_view(1))
    # Sample-frame parametrization: R = S @ R_z(theta), S aligns e_z -> axis
    ez = np.array([0.0, 0.0, 1.0])
    theta = np.deg2rad(10.0)
    # Build S by aligning e_z to axis via cross-product formula
    u = ez / np.linalg.norm(ez)
    v = ax / np.linalg.norm(ax)
    c = np.dot(u, v)
    if c > 1.0 - 1e-12:
        S = np.eye(3)
    elif c < -1.0 + 1e-12:
        S = np.diag([1.0, -1.0, -1.0])
    else:
        k = np.cross(u, v); s = np.linalg.norm(k); k = k / s
        K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],float)
        # Proper Rodrigues formula for aligning u->v
        S = np.eye(3) + s * K + (1 - c) * (K @ K)
    Rz = rot_axis_angle(ez, theta)[:3, :3]
    R_ref = S @ Rz
    assert np.allclose(T1[:3, :3], R_ref, atol=1e-7)
    # Rotation angle should be 10 deg around some axis; check orthonormality
    R = T1[:3, :3]
    assert np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_se3_exp_compose_invert_consistency():
    # Small random twists
    rng = np.random.default_rng(0)
    xi_a = rng.normal(scale=0.01, size=(6,))
    xi_b = rng.normal(scale=0.01, size=(6,))
    T_a = exp_se3(xi_a)
    T_b = exp_se3(xi_b)
    T_ab = compose(T_a, T_b)
    T_ab_inv = invert(T_ab)
    I = compose(T_ab, T_ab_inv)
    assert np.allclose(I, np.eye(4), atol=1e-9)
