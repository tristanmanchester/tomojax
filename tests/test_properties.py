"""Properties that hold for every valid geometry, checked on generated geometries.

The projector pair is an exact transpose, saved files give back the geometry they
were saved with, and the automatic reconstruction runs the method its documented
policy names. Hypothesis draws small parallel-beam, laminography and cone-beam
scans with shifted centers, tilts about x and z, anisotropic pitches, odd and even
sizes, uneven angles and per-view poses. Examples are derandomized so every run
checks the same cases.
"""

import dataclasses
import math

from hypothesis import HealthCheck, given, settings, strategies as st
import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import grid_volume_origin

pytestmark = pytest.mark.numerical

# Each example compiles its own projector, so a few cover the space at CPU-gate cost.
EXAMPLES = settings(max_examples=16, deadline=None, derandomize=True, database=None)

# |<Ax, y> - <x, A^T y>| over (|Ax| |y| + |x| |A^T y|). Both inner products are
# float32 sums over at most a few thousand terms: over 150 drawn geometries the
# matched pair's residual stayed below 2e-8, and a transpose a tenth of a pixel off
# scores 2e-3.
TRANSPOSE_TOLERANCE = 1e-6


def quantities(low, high):
    """Values in [low, high] to a thousandth: the magnitudes real scans record."""
    return st.integers(round(low * 1000), round(high * 1000)).map(lambda n: n / 1000)


pitches = quantities(0.5, 2.0)
offsets = quantities(-1.5, 1.5)


@st.composite
def grids(draw):
    sizes = [draw(st.integers(3, 8)) for _ in "xyz"]
    spacing = [draw(pitches) for _ in "xyz"]
    placement = tuple(draw(offsets) for _ in "xyz")
    if draw(st.booleans()):
        return tj.Grid(*sizes, *spacing, vol_center=placement)
    center = np.subtract(placement, np.multiply(np.subtract(sizes, 1), spacing) / 2)
    return tj.Grid(*sizes, *spacing, vol_origin=tuple(center.tolist()))


def covering_detector(draw, grid, magnification=1.0):
    """A detector at a drawn pitch and center that sees most of ``grid`` in every view."""
    extent = magnification * math.hypot(grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz)
    du, dv = draw(pitches), draw(pitches)
    nu = min(24, math.ceil(extent / du) + draw(st.integers(0, 1)))
    nv = min(24, math.ceil(extent / dv) + draw(st.integers(0, 1)))
    return tj.Detector(nu, nv, du, dv, (draw(offsets), draw(offsets)))


uneven_angles = st.lists(quantities(0.0, 360.0), min_size=2, max_size=9).map(sorted)


@st.composite
def geometries(draw, *, kinds=("parallel", "lamino", "cone")):
    """A scan geometry of a drawn kind, with or without a per-view pose table."""
    kind = draw(st.sampled_from(kinds))
    grid = draw(grids())
    angles = np.asarray(draw(uneven_angles))
    tilt_about = draw(st.sampled_from(["x", "z"]))
    if kind == "parallel":
        detector = covering_detector(draw, grid)
        geometry = tj.ParallelGeometry(
            grid, detector, angles, detector_roll=draw(quantities(-8.0, 8.0))
        )
    elif kind == "lamino":
        detector = covering_detector(draw, grid)
        geometry = tj.LaminographyGeometry(
            grid,
            detector,
            angles,
            tilt=draw(quantities(5.0, 60.0)),
            tilt_about=tilt_about,
            detector_roll=draw(quantities(-8.0, 8.0)),
        )
    else:
        extent = math.hypot(grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz)
        source_to_axis = extent * draw(quantities(3.0, 6.0))
        source_to_detector = source_to_axis * draw(quantities(1.2, 2.0))
        beam = tj.ConeBeam(
            source_to_axis,
            source_to_detector,
            detector_roll=draw(quantities(-5.0, 5.0)),
            detector_pitch=draw(quantities(-5.0, 5.0)),
            detector_yaw=draw(quantities(-5.0, 5.0)),
            axis_offset=draw(offsets),
            source_height=draw(offsets),
        )
        detector = covering_detector(draw, grid, source_to_detector / source_to_axis)
        geometry = tj.ConeGeometry(
            grid,
            detector,
            angles,
            beam,
            tilt=draw(st.sampled_from([0.0, draw(quantities(-20.0, 20.0))])),
            tilt_about=tilt_about,
        )
    if not draw(st.booleans()):
        return geometry
    views = len(angles)
    rotations = draw(st.lists(quantities(-3.0, 3.0), min_size=3 * views, max_size=3 * views))
    shifts = draw(st.lists(offsets, min_size=3 * views, max_size=3 * views))
    poses = np.c_[np.reshape(rotations, (views, 3)), np.reshape(shifts, (views, 3))]
    return unposed_scan(geometry).with_poses(poses).geometry


def unposed_scan(geometry):
    shape = (len(geometry.angles), geometry.detector.nv, geometry.detector.nu)
    return tj.Scan(np.zeros(shape, np.float32), geometry)


def transpose_residual(geometry, seed):
    """The normalized adjoint residual of ``tj.Projector(geometry)`` on random inputs."""
    operator = tj.Projector(geometry)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(operator.volume_shape, dtype=np.float32)
    y = rng.standard_normal(operator.projection_shape, dtype=np.float32)
    ax = np.asarray(operator(x), np.float64)
    aty = np.asarray(operator.adjoint(y), np.float64)
    scale = np.linalg.norm(ax) * np.linalg.norm(y) + np.linalg.norm(x) * np.linalg.norm(aty)
    # A geometry that misses the detector entirely has nothing to compare.
    assert scale > 0, "the drawn detector sees none of the volume"
    return abs(np.vdot(ax, y) - np.vdot(x, aty)) / scale


@EXAMPLES
@given(geometry=geometries(), seed=st.integers(0, 2**32 - 1))
def test_projector_is_the_exact_transpose_for_any_geometry(geometry, seed):
    assert transpose_residual(geometry, seed) < TRANSPOSE_TOLERANCE


def test_the_transpose_check_detects_a_tenth_of_a_pixel_mismatch():
    # The tolerance has to separate a matched pair from a nearly matched one.
    grid = tj.Grid(7, 6, 5, 0.8, 1.1, 1.3, vol_center=(0.2, -0.1, 0.3))
    detector = tj.Detector(11, 9, 0.9, 1.2, (0.17, -0.23))
    geometry = tj.LaminographyGeometry(grid, detector, [3.0, 47.0, 151.0], tilt=25)
    shifted = tj.LaminographyGeometry(
        grid, tj.Detector(11, 9, 0.9, 1.2, (0.17 + 0.09, -0.23)), [3.0, 47.0, 151.0], tilt=25
    )
    rng = np.random.default_rng(1)
    forward, transpose = tj.Projector(geometry), tj.Projector(shifted)
    x = rng.standard_normal(forward.volume_shape, dtype=np.float32)
    y = rng.standard_normal(forward.projection_shape, dtype=np.float32)
    ax = np.asarray(forward(x), np.float64)
    aty = np.asarray(transpose.adjoint(y), np.float64)
    scale = np.linalg.norm(ax) * np.linalg.norm(y) + np.linalg.norm(x) * np.linalg.norm(aty)
    assert abs(np.vdot(ax, y) - np.vdot(x, aty)) / scale > 100 * TRANSPOSE_TOLERANCE
    assert transpose_residual(geometry, 1) < TRANSPOSE_TOLERANCE


@pytest.mark.gpu
@settings(EXAMPLES, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(geometry=geometries(), seed=st.integers(0, 2**32 - 1))
def test_cuda_and_cpu_projectors_are_one_matched_pair(projector_device, geometry, seed):
    # The fixture runs each lowering in turn; both must give the CPU pair's values. Over
    # 100 drawn geometries CUDA differed from the CPU by at most 4e-6 of the peak value.
    rng = np.random.default_rng(seed)
    operator = tj.Projector(geometry)
    x = rng.standard_normal(operator.volume_shape, dtype=np.float32)
    y = rng.standard_normal(operator.projection_shape, dtype=np.float32)
    assert transpose_residual(geometry, seed) < TRANSPOSE_TOLERANCE
    with jax.default_device(jax.devices("cpu")[0]):
        reference = tj.Projector(geometry)
        expected_forward = np.asarray(reference(x))
        expected_adjoint = np.asarray(reference.adjoint(y))
    scale = np.abs(expected_forward).max()
    np.testing.assert_allclose(operator(x), expected_forward, rtol=0, atol=2e-5 * scale)
    scale = np.abs(expected_adjoint).max()
    np.testing.assert_allclose(operator.adjoint(y), expected_adjoint, rtol=0, atol=2e-5 * scale)


def nominal(geometry):
    """The calibrated geometry beneath any per-view poses."""
    while hasattr(geometry, "base"):
        geometry = geometry.base
    return geometry


def assert_same_geometry(actual, expected):
    """Every view's pose agrees, and so does each declared calibration value."""
    for view in range(len(expected.angles)):
        np.testing.assert_allclose(
            actual.pose_matrix(view), expected.pose_matrix(view), rtol=0, atol=1e-6
        )
    actual, expected = nominal(actual), nominal(expected)
    assert type(actual) is type(expected)
    for field in dataclasses.fields(expected):
        value, wanted = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "angles":  # Files store angles, like the projector, in float32.
            np.testing.assert_array_equal(np.float32(value), np.float32(wanted))
        elif field.name != "tilt_about" or expected.tilt != 0:  # an untilted axis has no plane
            assert value == wanted, field.name
    assert grid_volume_origin(actual.grid) == grid_volume_origin(expected.grid)


@settings(EXAMPLES, max_examples=12)
@given(geometry=geometries(), seed=st.integers(0, 2**32 - 1), unit=st.sampled_from(["mm", "um"]))
def test_saved_scans_and_reconstructions_reload_with_their_geometry_and_units(
    tmp_path_factory, geometry, seed, unit
):
    directory = tmp_path_factory.mktemp("saved")
    rng = np.random.default_rng(seed)
    shape = (len(geometry.angles), geometry.detector.nv, geometry.detector.nu)
    scan = tj.Scan(rng.random(shape, dtype=np.float32), geometry)
    tj.save(directory / "scan.nxs", scan)
    loaded = tj.load(directory / "scan.nxs")
    assert_same_geometry(loaded.geometry, scan.geometry)
    np.testing.assert_array_equal(loaded.projections, scan.projections)
    if scan.poses is None:
        assert loaded.poses is None
    else:
        np.testing.assert_array_equal(loaded.poses, scan.poses)

    grid = geometry.grid
    volume = rng.random((grid.nz, grid.ny, grid.nx), dtype=np.float32)
    recon = tj.Reconstruction(volume, geometry, "fbp", {}, scan, length_unit=unit)
    tj.save(directory / "recon.nxs", recon, include_scan=True)
    reloaded = tj.load_reconstruction(directory / "recon.nxs")
    assert reloaded.length_unit == unit
    assert reloaded.grid == grid
    np.testing.assert_array_equal(reloaded.volume, volume)
    assert_same_geometry(reloaded.geometry, geometry)
    assert reloaded.scan is not None
    np.testing.assert_array_equal(reloaded.scan.projections, scan.projections)


@settings(EXAMPLES, max_examples=12)
@given(
    kind=st.sampled_from(["parallel", "lamino", "cone"]),
    grid=grids(),
    data=st.data(),
    n_views=st.integers(2, 9),
    translation_sigma=st.sampled_from([0.0, 0.4]),
    seed=st.integers(0, 1000),
)
def test_simulations_reload_with_their_placement_spacing_and_motion(
    tmp_path_factory, kind, grid, data, n_views, translation_sigma, seed
):
    detector = covering_detector(data.draw, grid, 2.0 if kind == "cone" else 1.0)
    tilt = data.draw(quantities(5.0, 45.0)) if kind != "parallel" else None
    sim = tj.simulate(
        "random_shapes",
        geometry=kind,
        grid=grid,
        detector=detector,
        n_views=n_views,
        tilt=tilt,
        translation_sigma=translation_sigma,
        seed=seed,
    )
    # The simulation keeps what it was asked for, before any file is involved.
    assert sim.scan.grid == grid and sim.scan.detector == detector
    assert sim.phantom.shape == (grid.nz, grid.ny, grid.nx)
    assert sim.scan.projections.shape == (n_views, detector.nv, detector.nu)
    if kind != "parallel":
        assert nominal(sim.scan.geometry).tilt == tilt

    path = tmp_path_factory.mktemp("simulated") / "simulation.nxs"
    tj.save(path, sim)
    loaded = tj.load_simulation(path)
    assert_same_geometry(loaded.scan.geometry, sim.scan.geometry)
    np.testing.assert_array_equal(loaded.scan.projections, sim.scan.projections)
    np.testing.assert_array_equal(loaded.phantom, sim.phantom)
    if sim.true_poses is None:
        assert loaded.true_poses is None
    else:
        np.testing.assert_array_equal(loaded.true_poses, sim.true_poses)

    report = tj.inspect(path)
    assert report["projection"]["shape"] == [n_views, detector.nv, detector.nu]
    assert report["angles"]["count"] == n_views and report["angles"]["units"] == "degree"
    assert report["volume"]["shape"] == [grid.nz, grid.ny, grid.nx]
    assert report["volume"]["axes"] == "zyx"
    described = report["detector_metadata"]
    assert (described["du"], described["dv"]) == (detector.du, detector.dv)
    assert tuple(described["center"]) == detector.center
    assert report["valid"], report["issues"]


@st.composite
def spread_angles(draw, *, arc=180.0):
    """16 to 30 views evenly over ``arc`` degrees, each a little off."""
    views = draw(st.integers(16, 30))
    step = arc / views
    start = draw(quantities(0.0, 360.0))
    jitter = draw(st.lists(quantities(-0.1, 0.1), min_size=views, max_size=views))
    return start + step * (np.arange(views) + np.asarray(jitter))


@st.composite
def automatic_cases(draw):
    """A scan and the method the documented policy of ``method="auto"`` names for it.

    Fourier inversion takes an unposed axial parallel scan that measures 16 or more
    directions spread over the half turn. A cone-beam arc FDK cannot weight, shorter
    than 180 degrees plus the fan angle, gets CGLS. Every other scan gets FBP or FDK.
    """
    case = draw(
        st.sampled_from(
            ["half-turn", "few views", "missing wedge", "posed", "lamino", "cone", "cone arc"]
        )
    )
    if case == "half-turn":
        geometry = draw(geometries(kinds=("parallel",)).map(nominal))
        geometry = dataclasses.replace(geometry, angles=draw(spread_angles()))
        expected = "fourier"
    elif case == "few views":
        geometry = dataclasses.replace(
            draw(geometries(kinds=("parallel",)).map(nominal)),
            angles=np.asarray(draw(st.lists(quantities(0.0, 360.0), min_size=2, max_size=15))),
        )
        expected = "fbp"
    elif case == "missing wedge":
        geometry = dataclasses.replace(
            draw(geometries(kinds=("parallel",)).map(nominal)),
            angles=draw(spread_angles(arc=draw(quantities(90.0, 140.0)))),
        )
        expected = "fbp"
    elif case == "posed":
        geometry = dataclasses.replace(
            draw(geometries(kinds=("parallel",)).map(nominal)), angles=draw(spread_angles())
        )
        views = len(geometry.angles)
        shifts = draw(st.lists(quantities(0.1, 1.0), min_size=views, max_size=views))
        poses = np.zeros((views, 6))
        poses[:, 3] = shifts
        geometry = unposed_scan(geometry).with_poses(poses).geometry
        expected = "fbp"
    elif case == "lamino":
        geometry = draw(geometries(kinds=("lamino",)))
        expected = "fbp"
    else:
        cone = draw(geometries(kinds=("cone",)).map(nominal))
        full_turn = case == "cone"
        angles = draw(spread_angles(arc=360.0 if full_turn else draw(quantities(20.0, 170.0))))
        geometry = dataclasses.replace(cone, angles=angles)
        expected = "fbp" if full_turn else "cgls"
    return geometry, expected


@EXAMPLES
@given(case=automatic_cases(), seed=st.integers(0, 2**32 - 1))
def test_automatic_reconstruction_runs_the_method_its_policy_names(case, seed):
    geometry, expected = case
    grid = geometry.grid
    phantom = np.random.default_rng(seed).random((grid.nz, grid.ny, grid.nx), dtype=np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, phantom)), geometry)

    automatic = tj.reconstruct(scan)
    assert automatic.method == expected
    assert automatic.info["automatic"]["method"] == expected
    explicit = tj.reconstruct(scan, expected)
    np.testing.assert_array_equal(automatic.volume, explicit.volume)
    assert np.isfinite(automatic.volume).all()
    # Method-specific options are refused rather than dropped.
    with pytest.raises(tj.InvalidParameter, match="iterations"):
        tj.reconstruct(scan, iterations=3)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
@settings(EXAMPLES, max_examples=2)  # each fit compiles for about 10 s on a CPU
@given(data=st.data(), seed=st.integers(0, 2**32 - 1))
def test_default_alignment_fits_translations_and_holds_everything_else(kind, data, seed):
    geometry = data.draw(geometries(kinds=(kind,)))
    grid = geometry.grid
    phantom = np.random.default_rng(seed).random((grid.nz, grid.ny, grid.nx), dtype=np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, phantom)), geometry)
    views = len(geometry.angles)

    result = tj.align(scan)
    # The joint fit is the default, and fits translations: two components across
    # parallel rays and three for a cone beam, which magnifies with depth.
    cone = isinstance(nominal(geometry), tj.ConeGeometry)
    assert result.info["fit"] == ["translations"]
    assert result.info["pose_columns"] == (["x", "z", "y"] if cone else ["x", "z"])
    assert result.poses.shape == (views, 6) and np.isfinite(result.poses).all()
    declared = np.zeros((views, 6)) if scan.poses is None else scan.poses
    held = [0, 1, 2] if cone else [0, 1, 2, 5]  # alpha, beta, phi and, across parallel rays, y
    np.testing.assert_allclose(result.poses[:, held], declared[:, held], rtol=0, atol=1e-5)
    assert_same_geometry(nominal(result.scan.geometry), nominal(geometry))


@EXAMPLES
@given(geometry=geometries(kinds=("lamino", "cone")))
def test_opposite_views_refuses_scans_without_an_axial_parallel_beam(geometry):
    # A method that does not apply to the scan is a bad ``method``, as for FDK's arcs.
    with pytest.raises(tj.InvalidParameter, match="opposite-views") as refused:
        tj.align(unposed_scan(geometry), method="opposite-views")
    assert refused.value.path == "method"
