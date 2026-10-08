"""Cone-beam geometry, projector, solvers, FDK and dataset round trips."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax.core.cone import cone_backproject, cone_coefficients, cone_project, use_cuda_cone
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, grid_volume_origin
from tomojax.io import build_geometry_from_dataset_metadata, load_dataset, save_dataset
from tomojax.recon import (
    CGLSConfig,
    ConeAxisConfig,
    FDKConfig,
    FDKHostConfig,
    calibrate_cone_axis,
    cgls,
    fbp,
    fbp_host,
    fdk,
    fdk_host,
)

cuda = pytest.param("cuda", marks=pytest.mark.gpu)


def _ellipsoids(extent: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
    shapes = []
    for amp, centre, radii, angle in [
        (1.0, (0.0, 0.0, 0.0), (0.28, 0.26, 0.30), 0.0),
        (-0.55, (-0.10, 0.04, 0.03), (0.09, 0.12, 0.10), 23.0),
        (0.75, (0.12, -0.08, -0.09), (0.065, 0.055, 0.08), -17.0),
    ]:
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        metric = rotation @ np.diag(1 / (extent * np.asarray(radii)) ** 2) @ rotation.T
        shapes.append((amp, extent * np.asarray(centre), metric))
    return shapes


def _voxelise(grid: Grid, shapes: list, supersample: int = 3) -> np.ndarray:
    origin = np.asarray(grid_volume_origin(grid))
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    axes = [
        origin[i] + (np.arange(n)[:, None] + offsets[None, :]).ravel() * spacing[i]
        for i, n in enumerate((grid.nx, grid.ny, grid.nz))
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    inside = sum(
        amp * (np.einsum("...i,ij,...j->...", points - c, m, points - c) <= 1)
        for amp, c, m in shapes
    )
    shape = (grid.nx, supersample, grid.ny, supersample, grid.nz, supersample)
    return np.asarray(inside, np.float64).reshape(shape).mean(axis=(1, 3, 5)).astype(np.float32)


def _analytic(shapes: list, geometry: ConeGeometry) -> np.ndarray:
    """Exact line integrals from the source through each pixel centre."""
    beam, det = geometry.beam, geometry.detector
    centre, u_dir, v_dir = beam.detector_frame(det)
    u = (np.arange(det.nu) - (det.nu - 1) / 2) * det.du
    v = (np.arange(det.nv) - (det.nv - 1) / 2) * det.dv
    pixels = centre + u[None, :, None] * u_dir + v[:, None, None] * v_dir
    out = []
    for pose in geometry.poses():
        rot, trans = pose[:3, :3], pose[:3, 3]
        source = rot.T @ (beam.source() - trans)
        direction = (pixels - trans) @ rot - source
        values = np.zeros(direction.shape[:-1])
        for amp, c, m in shapes:
            delta = source - c
            a = np.einsum("...i,ij,...j->...", direction, m, direction)
            b = np.einsum("...i,ij,j->...", direction, m, delta)
            cc = delta @ m @ delta - 1
            length = 2 * np.sqrt(np.maximum(b * b - a * cc, 0)) / a
            values += amp * length * np.linalg.norm(direction, axis=-1)
        out.append(values)
    return np.asarray(out, np.float32)


def _scan(
    kind: str, n: int = 20, views: int = 24
) -> tuple[ConeGeometry, Grid, Detector, np.ndarray]:
    grid = Grid(n, n - 3, n - 5, 1.0, 1.1, 0.9, vol_center=(0.4, -0.3, 0.2))
    detector = Detector(int(1.6 * n) + 3, int(1.5 * n) - 1, 1.1, 0.95, (0.7, -0.4))
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    if kind == "tilted":
        beam = ConeBeam(2.5 * n, 4.0 * n, 1.5, -2.0, 0.7)
        geometry = ConeGeometry(grid, detector, angles, beam, tilt_deg=20.0)
    elif kind == "steep":
        # Rays steeper than 45 degrees in z, sampled along z.
        geometry = ConeGeometry(grid, detector, angles, ConeBeam(3.0 * n, 4.5 * n), tilt_deg=60.0)
    else:
        geometry = ConeGeometry(grid, detector, angles, ConeBeam(3.0 * n, 4.5 * n))
    poses = geometry.poses()
    if kind == "perturbed":
        rng = np.random.default_rng(1)
        for pose in poses:
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix()
            pose[:3, 3] += rng.normal(0, 0.5, 3)
    return geometry, grid, detector, poses


@pytest.mark.parametrize("kind", ["turntable", "perturbed", "tilted", "steep"])
@pytest.mark.parametrize("backend", ["jax", cuda])
def test_cone_projector_has_a_matched_transpose(kind, backend):
    geometry, grid, detector, poses = _scan(kind)
    coeff = cone_coefficients(jnp.asarray(poses, jnp.float32), grid, detector, geometry.beam)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.random((grid.nx, grid.ny, grid.nz)), jnp.float32)
    y = jnp.asarray(rng.random((len(poses), detector.nv, detector.nu)), jnp.float32)
    ax = cone_project(x, coeff, grid, detector, backend=backend)
    aty = cone_backproject(y, coeff, grid, detector, backend=backend)
    lhs = float(np.vdot(np.asarray(ax, np.float64), np.asarray(y, np.float64)))
    rhs = float(np.vdot(np.asarray(x, np.float64), np.asarray(aty, np.float64)))
    assert abs(lhs - rhs) <= 2e-6 * abs(lhs)
    accumulated = cone_backproject(y, coeff, grid, detector, backend=backend, accumulate=x)
    np.testing.assert_allclose(accumulated, aty + x, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["turntable", "perturbed", "tilted", "steep"])
def test_cuda_cone_kernels_agree_with_the_jax_reference(kind):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    geometry, grid, detector, poses = _scan(kind, n=40, views=36)
    coeff = cone_coefficients(jnp.asarray(poses, jnp.float32), grid, detector, geometry.beam)
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.random((grid.nx, grid.ny, grid.nz)), jnp.float32)
    y = jnp.asarray(rng.random((len(poses), detector.nv, detector.nu)), jnp.float32)
    for function, value in ((cone_project, x), (cone_backproject, y)):
        expected = function(value, coeff, grid, detector, backend="jax")
        actual = function(value, coeff, grid, detector, backend="cuda")
        assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 3e-6


@pytest.mark.parametrize("backend", ["jax", cuda])
def test_cone_projection_matches_analytic_line_integrals(backend):
    n = 32
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(52, 50, 1.0, 1.0)
    geometry = ConeGeometry(
        grid, detector, np.linspace(0, 360, 12, endpoint=False), ConeBeam(80, 130)
    )
    shapes = _ellipsoids(float(n))
    coeff = cone_coefficients(
        jnp.asarray(geometry.poses(), jnp.float32), grid, detector, geometry.beam
    )
    projected = np.asarray(
        cone_project(_voxelise(grid, shapes), coeff, grid, detector, backend=backend)
    )
    truth = _analytic(shapes, geometry)
    # Partial-volume error of sharp edges on a 32-cubed grid; 2.3% at 64-cubed.
    assert np.linalg.norm(projected - truth) / np.linalg.norm(truth) < 0.06


def test_cone_projection_is_differentiable_in_the_poses():
    geometry, grid, detector, poses = _scan("perturbed", n=12, views=3)
    x = jnp.asarray(np.random.default_rng(3).random((grid.nx, grid.ny, grid.nz)), jnp.float32)

    def project(shift: jax.Array) -> jax.Array:
        moved = jnp.asarray(poses, jnp.float32).at[:, :3, 3].add(shift)
        coeff = cone_coefficients(moved, grid, detector, geometry.beam)
        return cone_project(x, coeff, grid, detector, backend="jax")

    direction = jnp.asarray([0.3, -0.5, 0.4], jnp.float32)
    _, tangent = jax.jvp(project, (jnp.zeros(3),), (direction,))
    step = 1e-2
    central = (project(step * direction) - project(-step * direction)) / (2 * step)
    assert float(jnp.linalg.norm(tangent - central) / jnp.linalg.norm(central)) < 0.05


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_cgls_solves_a_cone_beam_scan(backend):
    n = 24
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(40, 40, 1.0, 1.0)
    geometry = ConeGeometry(
        grid, detector, np.linspace(0, 360, 40, endpoint=False), ConeBeam(60, 100)
    )
    truth = _voxelise(grid, _ellipsoids(float(n)))
    coeff = cone_coefficients(
        jnp.asarray(geometry.poses(), jnp.float32), grid, detector, geometry.beam
    )
    data = cone_project(truth, coeff, grid, detector, backend="jax")
    config = CGLSConfig(iters=60, projector_backend=backend)
    volume, _ = cgls(geometry, grid, detector, data, config=config)
    assert np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth) < 0.05


@pytest.mark.parametrize("backend", ["jax", cuda])
def test_fdk_reconstructs_full_and_short_scans(backend):
    n = 32
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(48, 48, 1.0, 1.0)
    beam = ConeBeam(96, 144)
    shapes = _ellipsoids(float(n))
    truth = _voxelise(grid, shapes)
    fan = np.rad2deg(2 * np.arctan(24 / beam.source_to_detector))
    errors = []
    for arc, views in ((360.0, 120), (180.0 + fan + 2.0, 90)):
        angles = np.linspace(0.0, arc, views, endpoint=arc < 360.0)
        geometry = ConeGeometry(grid, detector, angles, beam)
        volume = fdk(
            geometry, grid, detector, _analytic(shapes, geometry), config=FDKConfig(backend=backend)
        )
        errors.append(np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth))
    assert errors[0] < 0.12 and errors[1] < 0.16
    short = ConeGeometry(grid, detector, np.linspace(0.0, 150.0, 40), beam)
    with pytest.raises(ValueError, match="short scans"):
        fdk(short, grid, detector, np.zeros((40, 48, 48), np.float32))


@pytest.mark.parametrize("backend", ["jax", cuda])
def test_fdk_reconstructs_a_full_turn_on_an_offset_detector(backend):
    # The axis projects 5.5 columns from one edge: the far side is measured once
    # per turn and the near side's filtered tail lies beyond the detector.
    n = 32
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    shapes = _ellipsoids(float(n))
    truth = _voxelise(grid, shapes)
    angles = np.linspace(0.0, 360.0, 120, endpoint=False)
    errors = []
    for detector in (Detector(48, 48, 1.0, 1.0), Detector(32, 48, 1.0, 1.0, (10.0, 0.0))):
        geometry = ConeGeometry(grid, detector, angles, ConeBeam(96, 144))
        config = FDKConfig(backend=backend)
        volume = fdk(geometry, grid, detector, _analytic(shapes, geometry), config=config)
        errors.append(np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth))
    assert errors[1] < 1.05 * errors[0] < 0.1


@pytest.mark.parametrize("backend", ["jax", cuda])
def test_fdk_host_reconstructs_in_slabs_into_a_memmap(backend, tmp_path: Path):
    n = 24
    grid = Grid(n, n - 2, n, 1.0, 1.0, 1.1, vol_center=(0.5, -0.3, 1.5))
    detector = Detector(40, 38, 1.0, 1.0, (0.4, -1.2))
    angles = np.linspace(0.0, 360.0, 48, endpoint=False)
    data = np.random.default_rng(0).random((48, detector.nv, detector.nu)).astype(np.float32)
    config = FDKConfig(backend=backend, views_per_batch=20)
    for beam in (ConeBeam(72, 108), ConeBeam(72, 108, detector_roll_deg=1.0)):
        geometry = ConeGeometry(grid, detector, angles, beam)
        expected = np.asarray(fdk(geometry, grid, detector, data, config=config))
        # A slab's detector window shifts the coordinates, and CUDA's texture unit
        # rounds interpolation weights to 1/256 (as ASTRA's FDK), so slabs then agree
        # only to that; random data is the worst case.
        scale = np.abs(expected).max() * (2e-5 if backend == "jax" else 2e-3)
        for depth in (5, n):
            out = np.lib.format.open_memmap(
                tmp_path / "volume.npy", mode="w+", dtype=np.float32, shape=(n, n - 2, n)
            )
            result = fdk_host(
                geometry, grid, detector, data,
                config=FDKHostConfig(slices_per_batch=depth, fdk=config), out=out,
            )  # fmt: skip
            assert result is out
            np.testing.assert_allclose(out, expected, atol=scale)


def test_fbp_reconstructs_cone_scans_with_fdk_and_parallel_only_paths_refuse_them():
    geometry, grid, detector, _ = _scan("turntable", n=16, views=30)
    data = np.random.default_rng(4).random((30, detector.nv, detector.nu)).astype(np.float32)
    np.testing.assert_allclose(
        fbp(geometry, grid, detector, data), fdk(geometry, grid, detector, data), atol=1e-6
    )
    with pytest.raises(ValueError, match="parallel rays"):
        fbp_host(geometry, grid, detector, data)


def test_cone_geometry_round_trips_through_saved_datasets(tmp_path: Path):
    from tomojax.io import ProjectionDataset

    geometry, grid, detector, _ = _scan("tilted", n=10, views=6)
    dataset = ProjectionDataset(
        projections=np.zeros((6, detector.nv, detector.nu), np.float32),
        angles_deg=np.asarray(geometry.thetas_deg, np.float32),
        detector=detector,
        grid=grid,
        geometry_type="cone",
        geometry_metadata=geometry.geometry_metadata(),
    )
    save_dataset(tmp_path / "cone.nxs", dataset)
    loaded = load_dataset(tmp_path / "cone.nxs")
    _, _, rebuilt = build_geometry_from_dataset_metadata(loaded.geometry_inputs())
    assert isinstance(rebuilt, ConeGeometry)
    assert rebuilt.beam == geometry.beam
    np.testing.assert_allclose(rebuilt.poses(), geometry.poses(), atol=1e-6)


def test_import_records_cone_beam_geometry(tmp_path: Path):
    from tomojax.cli.main import main

    from ._helpers import write_angle_csv, write_tiff_stack

    write_tiff_stack(tmp_path / "tiffs", [1.0, 2.0, 3.0, 4.0], shape=(4, 6))
    write_angle_csv(tmp_path / "angles.csv", [0.0, 90.0, 180.0, 270.0])
    args = ["import", str(tmp_path / "tiffs"), "-o", str(tmp_path / "scan.nxs"), "--force"]
    args += ["--angles"]
    args += [str(tmp_path / "angles.csv"), "--geometry", "cone", "--source-to-axis", "50"]
    assert main([*args, "--source-to-detector", "80", "--detector-roll", "0.5"]) == 0
    loaded = load_dataset(tmp_path / "scan.nxs")
    _, _, geometry = build_geometry_from_dataset_metadata(loaded.geometry_inputs())
    assert isinstance(geometry, ConeGeometry)
    assert geometry.beam == ConeBeam(50.0, 80.0, detector_roll_deg=0.5)
    assert main([*args, "--source-to-detector", "80", "--axis-offset", "1.5"]) == 0
    _, _, shifted = build_geometry_from_dataset_metadata(
        load_dataset(tmp_path / "scan.nxs").geometry_inputs()
    )
    assert isinstance(shifted, ConeGeometry) and shifted.beam.axis_offset == 1.5
    np.testing.assert_allclose(shifted.poses()[:, :3, 3], [[1.5, 0.0, 0.0]] * 4)
    with pytest.raises(SystemExit):
        main([*args, "--source-to-detector", "40"])


def test_cone_projection_is_continuous_as_a_view_turns_through_45_degrees():
    n = 16
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(26, 24, 1.0, 1.0)
    beam = ConeBeam(48, 72)
    x = jnp.asarray(_voxelise(grid, _ellipsoids(float(n)), supersample=2))

    def project(deg: float) -> jax.Array:
        pose = ConeGeometry(grid, detector, [deg], beam).poses()
        coeff = cone_coefficients(jnp.asarray(pose, jnp.float32), grid, detector, beam)
        return cone_project(x, coeff, grid, detector, backend="jax")[0]

    step = 1e-2
    across = float(jnp.linalg.norm(project(45 + step) - project(45 - step)))
    elsewhere = float(jnp.linalg.norm(project(40 + step) - project(40 - step)))
    assert across < 2 * elsewhere


@pytest.mark.parametrize("dof", ["alpha", "beta", "phi", "dx", "dz", "dy"])
def test_cone_pose_alignment_recovers_each_degree_of_freedom(dof):
    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2LossSpec, align, apply_pose_updates, least_motion_estimate

    n, views = 16, 12
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(26, 26, 1.0, 1.0)
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = ConeGeometry(grid, detector, angles, ConeBeam(48, 72))
    volume = jnp.asarray(_voxelise(grid, _ellipsoids(float(n)), supersample=2))
    k = ("alpha", "beta", "phi", "dx", "dz", "dy").index(dof)
    truth = np.zeros((views, 6), np.float32)
    rng = np.random.default_rng(k)
    truth[:, k] = rng.uniform(-1, 1, views) * (np.deg2rad(0.3) if k < 3 else 0.6)
    truth[:, k] -= truth[:, k].mean() if dof == "dy" else 0.0  # a common dy is the scale gauge
    poses = apply_pose_updates(
        jnp.asarray(geometry.poses(), jnp.float32), jnp.asarray(truth), translation_frame="detector"
    )
    data = cone_project(
        volume, cone_coefficients(poses, grid, detector, geometry.beam), grid, detector
    )
    config = AlignConfig(
        pose_translation_frame="detector",
        optimise_dofs=(dof,),
        outer_iters=4,
        recon_iters=1,
        recon_L=1e12,  # keeps the known volume fixed
        lambda_tv=0,
        loss=L2LossSpec(),
        early_stop=False,
        gather_dtype="fp32",
        gn_jacobian="central",
        ray_integrator="joseph",
        projector_backend="jax",
    )
    _, params, _ = align(geometry, grid, detector, data, config=config, init_x=volume)
    # Alignment reports the least-motion estimate: a common phi or dz is the
    # object's rotation or position, not motion.
    expected = least_motion_estimate(
        np.asarray(volume),
        truth,
        nominal=np.asarray(geometry.poses()),
        grid=grid,
        translation_frame="detector",
        active=(dof,),
        cone_beam=True,
    )[1]
    scale = np.rad2deg(1) if k < 3 else 1.0
    np.testing.assert_allclose(np.asarray(params)[:, k] * scale, expected[:, k] * scale, atol=2e-3)


def test_parallel_geometry_keeps_dy_inactive_and_cone_geometry_adds_it():
    from tomojax.alignment import AlignConfig

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_loop import _with_beam_translation
    from tomojax.geometry import ParallelGeometry

    grid, detector = Grid(4, 4, 4, 1.0, 1.0, 1.0), Detector(6, 6, 1.0, 1.0)
    cone = ConeGeometry(grid, detector, [0.0, 90.0], ConeBeam(20, 30))
    parallel = ParallelGeometry(grid, detector, [0.0, 90.0])
    five = (True, True, True, True, True, False)
    assert _with_beam_translation(five, parallel, AlignConfig()) == five
    assert _with_beam_translation(five, cone, AlignConfig())[-1]
    frozen = AlignConfig(freeze_dofs=("dy",))
    assert not _with_beam_translation(five, cone, frozen)[-1]


def _blob_scan(n: int, views: int, beam: ConeBeam) -> tuple[ConeGeometry, np.ndarray]:
    """A blob phantom's projections under ``beam``, and the nominal (centred) geometry."""
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(int(1.6 * n), int(1.6 * n), 1.0, 1.0)
    rng = np.random.default_rng(0)
    axes = np.meshgrid(*(np.arange(n) - (n - 1) / 2,) * 3, indexing="ij")
    volume = np.zeros((n, n, n), np.float32)
    for _ in range(40):
        centre, radius = rng.uniform(-0.35 * n, 0.35 * n, 3), rng.uniform(1.5, 0.06 * n)
        if np.hypot(centre[0], centre[1]) < 0.4 * n:
            r2 = sum((a - c) ** 2 for a, c in zip(axes, centre, strict=True))
            volume += rng.uniform(0.3, 1.0) * np.exp(-r2 / (2 * radius**2)).astype(np.float32)
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    truth = ConeGeometry(grid, detector, angles, beam)
    coeff = cone_coefficients(jnp.asarray(truth.poses(), jnp.float32), grid, detector, beam)
    data = np.array(cone_project(jnp.asarray(volume), coeff, grid, detector))
    data += rng.normal(0, 0.01 * data.std(), data.shape).astype(np.float32)
    nominal = ConeBeam(beam.source_to_axis, beam.source_to_detector)
    return ConeGeometry(grid, detector, angles, nominal), data


@pytest.mark.gpu
def test_calibrate_cone_axis_recovers_the_axis_offset_and_detector_roll():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    n = 96
    geometry, data = _blob_scan(n, 240, ConeBeam(3 * n, 4.5 * n, 0.7, axis_offset=-4.3))
    calibration = calibrate_cone_axis(geometry, geometry.grid, geometry.detector, data)
    assert abs(calibration.axis_offset + 4.3) < 0.1
    assert abs(calibration.detector_roll_deg - 0.7) < 0.1
    calibrated = calibration.apply(geometry)
    assert calibrated.beam.axis_offset == calibration.axis_offset
    np.testing.assert_allclose(calibrated.poses()[:, 0, 3], calibration.axis_offset)


def test_calibrate_cone_axis_finds_the_offset_with_the_jax_backend():
    n = 32
    geometry, data = _blob_scan(n, 60, ConeBeam(3 * n, 4.5 * n, axis_offset=2.4))
    config = ConeAxisConfig(
        estimate_roll=False,
        slices=4,
        fdk=FDKConfig(filter_name="hann", backend="jax", views_per_batch=60),
    )
    calibration = calibrate_cone_axis(
        geometry, geometry.grid, geometry.detector, data, config=config
    )
    assert abs(calibration.axis_offset - 2.4) < 0.2
    assert calibration.detector_roll_deg == 0.0 and len(calibration.heights) == 1


@pytest.mark.gpu
def test_align_cor_mode_writes_the_calibrated_cone_beam(tmp_path: Path):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    from tomojax.cli.main import main
    from tomojax.io import ProjectionDataset

    n = 64
    geometry, data = _blob_scan(n, 180, ConeBeam(3 * n, 4.5 * n, -0.5, axis_offset=3.1))
    dataset = ProjectionDataset(
        projections=data,
        angles_deg=np.asarray(geometry.thetas_deg, np.float32),
        detector=geometry.detector,
        grid=geometry.grid,
        geometry_type="cone",
        geometry_metadata=geometry.geometry_metadata(),
    )
    save_dataset(tmp_path / "scan.nxs", dataset)
    out = tmp_path / "aligned.nxs"
    assert main(["align", str(tmp_path / "scan.nxs"), "-o", str(out), "--mode", "cor"]) == 0
    _, _, calibrated = build_geometry_from_dataset_metadata(load_dataset(out).geometry_inputs())
    assert isinstance(calibrated, ConeGeometry)
    assert abs(calibrated.beam.axis_offset - 3.1) < 0.15
    assert abs(calibrated.beam.detector_roll_deg + 0.5) < 0.15


def test_import_reads_a_nikon_xtekct_scan(tmp_path: Path):
    import imageio.v3 as iio

    from tomojax.cli.main import main

    n, views = 24, 60
    grid = Grid(n, n, n, 0.05, 0.05, 0.05)
    detector = Detector(40, 36, 0.075, 0.075)
    beam = ConeBeam(12.0, 18.0)
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    geometry = ConeGeometry(grid, detector, angles, beam)
    shapes = _ellipsoids(n * 0.05)
    data = _analytic(shapes, geometry)
    white = 60000.0
    for i, image in enumerate(np.exp(-data) * white):
        # Detector images store their top row first.
        iio.imwrite(tmp_path / f"part_{i + 1:04d}.tif", np.round(image[::-1]).astype(np.uint16))
    rows = "\n".join(f"{i + 1}\t{angle:.4f}\t0" for i, angle in enumerate(angles))
    (tmp_path / "part_ctdata.txt").write_text(f"Projection\tAngle(deg)\tUnused\n\n{rows}\n")
    (tmp_path / "part.xtekct").write_text(
        "[XTekCT]\nName=part\nInputSeparator=_\nSrcToObject=12.0\nSrcToDetector=18.0\n"
        "DetectorPixelsX=40\nDetectorPixelsY=36\nDetectorPixelSizeX=0.075\n"
        "DetectorPixelSizeY=0.075\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
        f"VoxelsX={n}\nVoxelsY={n}\nVoxelsZ={n}\nVoxelSizeX=0.05\nVoxelSizeY=0.05\n"
        f"VoxelSizeZ=0.05\nWhiteLevel={white}\nProjections={views}\nInitialAngle=0\n"
        "AngularStep=99\nObjectOffsetX=0.01\n"
    )
    assert main(["import", str(tmp_path / "part.xtekct"), "-o", str(tmp_path / "part.nxs")]) == 0
    loaded = load_dataset(tmp_path / "part.nxs")
    _, _, ingested = build_geometry_from_dataset_metadata(loaded.geometry_inputs())
    assert isinstance(ingested, ConeGeometry) and ingested.beam == beam
    assert ingested.detector.du == 0.075 and loaded.grid == grid
    np.testing.assert_allclose(loaded.angles_deg, angles, atol=1e-4)  # from _ctdata.txt
    np.testing.assert_allclose(loaded.projections, data, atol=3e-4)
    volume = fdk(ingested, grid, ingested.detector, loaded.projections)
    truth = _voxelise(grid, shapes)
    assert np.linalg.norm(np.asarray(volume) - truth) / np.linalg.norm(truth) < 0.15


def test_pose_wrappers_stack_their_own_poses():
    # check-public-imports: allow-private
    from tomojax._data.geometry_meta import AugmentedGeometry

    # check-public-imports: allow-private
    from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry
    from tomojax.geometry import stack_view_poses

    geometry, _, _, _ = _scan("turntable", n=8, views=7)
    params = np.random.default_rng(3).normal(0, 0.05, (7, 6)).astype(np.float32)
    adjusted = PoseAdjustedGeometry(geometry, jnp.asarray(params), "detector")
    per_view = np.stack([np.asarray(adjusted.pose_for_view(i), np.float32) for i in range(7)])
    np.testing.assert_allclose(stack_view_poses(adjusted, 7), per_view, atol=1e-6)
    # A wrapper forwarding attributes to its base must not take the base's stacking.
    saved = AugmentedGeometry(geometry, params, "detector")
    per_view = np.stack([np.asarray(saved.pose_for_view(i), np.float32) for i in range(7)])
    np.testing.assert_allclose(stack_view_poses(saved, 7), per_view, atol=1e-6)


def test_fbp_reconstructs_volumes_larger_than_the_device_on_the_host(monkeypatch):
    import tomojax.backends
    from tomojax.recon.api import (
        ReconstructionAlgorithmOptions,
        ReconstructionAlgorithmRequest,
        run_reconstruction_algorithm,
    )

    geometry, grid, detector, _ = _scan("turntable", n=16, views=24)
    data = np.random.default_rng(4).random((24, detector.nv, detector.nu)).astype(np.float32)
    request = ReconstructionAlgorithmRequest(
        options=ReconstructionAlgorithmOptions(algorithm="fbp"),
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=data,
        detector_grid=None,
        volume_mask=None,
        views_per_batch=8,
        views_per_batch_mode="auto",
        gather_dtype="fp32",
    )
    on_device = np.asarray(run_reconstruction_algorithm(request).volume)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 4096)
    result = run_reconstruction_algorithm(request)
    assert isinstance(result.volume, np.ndarray) and result.algorithm_config["host_slabs"]
    # On CUDA, slabs' shifted detector windows round the texture unit's 1/256
    # interpolation weights differently (see the slab test above).
    tolerance = 2e-3 if use_cuda_cone() else 1e-5
    np.testing.assert_allclose(result.volume, on_device, atol=tolerance * np.abs(on_device).max())


def test_export_writes_volume_slices_and_raw_files(tmp_path: Path):
    import imageio.v3 as iio

    from tomojax.cli.main import main
    from tomojax.io import ProjectionDataset

    volume = np.random.default_rng(5).random((6, 5, 4)).astype(np.float32)  # (x, y, z)
    detector = Detector(6, 4, 1.0, 1.0)
    dataset = ProjectionDataset(
        projections=np.zeros((2, 4, 6), np.float32),
        angles_deg=np.asarray([0.0, 90.0], np.float32),
        volume=volume,
        detector=detector,
        grid=Grid(6, 5, 4, 0.5, 0.5, 0.25),
    )
    save_dataset(tmp_path / "recon.nxs", dataset)
    assert main(["export", str(tmp_path / "recon.nxs"), "-o", str(tmp_path / "tif")]) == 0
    np.testing.assert_array_equal(
        iio.imread(tmp_path / "tif" / "slice_00002.tif"), volume[:, :, 2].T
    )
    raw = tmp_path / "recon.raw"
    args = ["export", str(tmp_path / "recon.nxs"), "-o", str(raw)]
    assert main([*args, "--dtype", "uint16", "--range", "0", "1"]) == 0
    stored = np.fromfile(raw, "<u2").reshape(4, 5, 6)
    np.testing.assert_array_equal(stored, np.round(volume.transpose(2, 1, 0) * 65535))
    info = json.loads(raw.with_suffix(".json").read_text())
    assert info["shape_zyx"] == [4, 5, 6] and info["voxel_size_xyz"] == [0.5, 0.5, 0.25]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("n", "nz", "tilt"), [(8, 8, 0.0), (8, 40, 0.0), (12, 12, 5.0), (8, 40, 5.0)]
)
def test_cuda_cone_transpose_handles_volumes_smaller_than_a_tile(n, nz, tilt):
    # The adjoint kernels work in tiles of 32 or 64 voxels; their parts beyond a
    # small volume, close to the source, project through infinity.
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(n, n, nz, 1.0, 1.0, 1.0)
    detector = Detector(int(1.5 * n), int(1.5 * max(n, nz)), 1.0, 1.0)
    geometry = ConeGeometry(
        grid, detector, np.linspace(0, 360, 60, endpoint=False), ConeBeam(3 * n, 4.5 * n),
        tilt_deg=tilt,
    )  # fmt: skip
    coeff = cone_coefficients(
        jnp.asarray(geometry.poses(), jnp.float32), grid, detector, geometry.beam
    )
    y = jnp.asarray(np.random.default_rng(0).random((60, detector.nv, detector.nu)), jnp.float32)
    expected = cone_backproject(y, coeff, grid, detector, backend="jax")
    actual = cone_backproject(y, coeff, grid, detector, backend="cuda")
    assert float(jnp.abs(actual - expected).max() / jnp.abs(expected).max()) < 1e-5
