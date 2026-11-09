"""Joint image, global-geometry and per-view motion fitting.

Two bounded working scales preserve the physical field of view. Only parameters
selected by ``fit`` have columns or unknowns; shared centre, tilt and roll form a
small exact Schur border. Local motion uses a drift spline and robust jumps;
sample rotations additionally have a Gaussian prior in physical radians.

Linear interpolation and four image-CG iterations serve centre/translation
fits. Angular parameters use cubic interpolation for parallel rays and up to
eight CG iterations. These are internal discretization/work
choices for the requested model, never data-dependent route selection. Each
working pixel is one ray through its centre; a block with missing input pixels
is the mean of its finite ones, weighted by their fraction. Image TV is
disabled to avoid imposing a shape preference on weak geometry modes.
"""

from dataclasses import asdict
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.joseph import validate_plane_geometry
from tomojax.recon.fista_tv import resolve_tv_metric

from ._noise import estimate_noise
from ._operators import Operators
from ._parameters import Parameters
from ._pose import MotionPrior
from ._projection import cone_gauge, cone_parameter_scales, cropped_frames, rigid_gauge
from ._refine import refine
from ._solver import Budget, solve

_MAX_VOXELS = 16_000_000
_MAX_DETECTOR = 320
# (binning, maximum Gauss-Newton steps). The active model is the same at both
# scales; held coordinates have neither unknowns nor derivative columns.
_LEVELS = ((2, 12), (1, 8))
_LOSS_SIGMA = 2.5
_LOSS_CONTRAST = 0.5
_TV_WEIGHT = 0.0
_TV_HUBER = 1.0
# Views between spline knots for each active local coordinate.
_KNOT_SPACING = 10
_POSE_HUBER_PX = 0.02
# The fitted image extends 1/_Z_MARGIN of the working grid's height past each end.
_Z_MARGIN = 16
_ROTATION_SIGMA_DEG = 0.5


def _resized(grid, shape_xyz):
    """``grid`` resampled to ``shape_xyz`` cells over the same six cell faces."""
    old = np.asarray((grid.nx, grid.ny, grid.nz))
    new = np.asarray(shape_xyz, dtype=int)
    h = np.asarray((grid.vx, grid.vy, grid.vz))
    pitch = old * h / new
    origin = np.asarray(grid_volume_origin(grid)) - h / 2 + pitch / 2
    return Grid(
        *map(int, new), *map(float, pitch), vol_origin=tuple(origin), vol_center=grid.vol_center
    )


def _magnification(poses, frames):
    """Mean detector over object distance along the detector normal; 1 for parallel rays."""
    if frames is None:
        return 1.0
    normal = np.cross(frames[:, 1], frames[:, 2])
    distance = np.einsum("ni,ni->n", frames[:, 0] - frames[:, 3], normal)
    depth = np.einsum("ni,ni->n", poses[:, :3, 3] - frames[:, 3], normal)
    return float(np.mean(distance / depth))


def _padded(grid, bins):
    """``grid`` extended along z past each end, and the cells added at each end.

    Moving views bring material from above and below the field into the outer
    rows; the extension gives it somewhere to be, and is cropped from the
    returned volume. Cells no ray reaches stay out of the fit. Whole
    coarse-level cells keep every level's voxel centres where they were
    relative to the detector rows.
    """
    step = math.lcm(*bins)
    cells = step * math.ceil(grid.nz / (_Z_MARGIN * step)) if grid.nz > 1 else 0
    x, y, z = grid_volume_origin(grid)
    extended = Grid(
        grid.nx,
        grid.ny,
        grid.nz + 2 * cells,
        grid.vx,
        grid.vy,
        grid.vz,
        vol_origin=(x, y, z - cells * grid.vz),
        vol_center=grid.vol_center,
    )
    return extended, cells


def _coarsened(measured, b, batch_views=16):
    """Block means over complete b x b pixel blocks, and the observed fraction.

    Each block mean uses only its finite pixels and its precision is their
    fraction; precision is None when every pixel is finite. Input conversion
    and finite-pixel checks use one view slab at a time, including memory maps.
    """
    n, nv, nu = measured.shape
    y = np.empty((n, nv // b, nu // b), np.float32)
    w = np.ones_like(y)
    for s in range(0, n, batch_views):
        src = np.asarray(measured[s : s + batch_views], np.float32)
        blocks = (len(src), nv // b, b, nu // b, b)
        finite = np.isfinite(src)
        count = finite.reshape(blocks).sum((2, 4))
        total = np.where(finite, src, 0).reshape(blocks).sum((2, 4), dtype=np.float64)
        y[s : s + len(src)] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        w[s : s + len(src)] = count / (b * b)
        del src  # release this converted slab before allocating the next one
    return y, None if w.min() == 1 else w


def _capped_contrast(y, rms, precision):
    """The contrast RMS, capped at three central interdecile scales of offset-free samples."""
    total = y.size
    rng = np.random.default_rng(19871)
    indices = np.arange(total) if total <= 65536 else rng.integers(total, size=65536)
    values = y.reshape(-1)[indices].astype(np.float64)
    views = indices // (y.shape[1] * y.shape[2])
    valid = np.isfinite(values)
    if precision is not None:
        valid &= np.asarray(precision).reshape(-1)[indices] > 0
    values, views = values[valid], views[valid]
    if not len(values):
        raise ValueError("no valid samples for contrast calibration")
    for v in np.unique(views):
        values[views == v] -= np.median(values[views == v])
    low, high = np.quantile(values, [0.1, 0.9])
    return min(float(rms), 3.0 * float((high - low) / 2.5631031310892007))


def _level(
    measured,
    poses,
    base,
    detector,
    center,
    total_bin,
    b,
    *,
    lever,
    native,
    frames=None,
    scales=None,
    parameters=None,
    detector_roll_deg=0.0,
    interpolation="linear",
):
    """Observations, grid, operators and calibration at one binning."""
    nv, nu = measured.shape[1:]
    y, precision = _coarsened(measured, total_bin)
    g = _resized(base, tuple(math.ceil(n / b) for n in (base.nx, base.ny, base.nz)))
    d = Detector(
        nu // total_bin,
        nv // total_bin,
        detector.du * total_bin,
        detector.dv * total_bin,
        center=center,
    )
    metric = resolve_tv_metric(
        None, g, d, y, weights=precision, tv_weight=_TV_WEIGHT, huber_delta=_TV_HUBER
    )
    contrast = _capped_contrast(y, float(metric.contrast_scale), precision)
    # Estimate on the supplied finest samples, before coarsening turns smooth
    # texture into apparent noise. With precision=count/b**2, sigma/b gives
    # the same independent-pixel noise model for partially observed blocks.
    sigma = estimate_noise(measured, dtype=np.float32) / total_bin
    ops = Operators(
        g,
        d,
        poses,
        lever=lever,
        native=native,
        frames=frames,
        scales=scales,
        interpolation=interpolation,
        parameters=parameters,
        detector_roll_deg=detector_roll_deg,
        detector_pivot=detector.center,
    )
    h = (g.vx * g.vy * g.vz) ** (1 / 3)
    calibration = {
        "tv_weight": float(metric.weight_over_contrast) * contrast,
        "eps": max(1e-8, _TV_HUBER * contrast * h / float(metric.length_scale) ** 2),
        "delta": max(_LOSS_SIGMA * sigma, _LOSS_CONTRAST * contrast, 1e-8),
        "sigma": sigma,
        "scales": (1.0, 1.0, 1.0) if g.vx == g.vy == g.vz else (h / g.vz, h / g.vy, h / g.vx),
    }
    return y, precision, g, ops, calibration


def _gauge(poses, *, lever, native):
    """(5 views, 6): each view's parameter change under the six rigid motions of the object.

    Moving the object by G and every pose by G^-1 predicts the same data. To
    first order the poses change by -M G-hat; each view's parameters reproduce
    that in the nine rotation entries and the two translations a parallel beam
    sees (the one along the beam is invisible).
    """
    return rigid_gauge(poses, [1 / lever] * 3 + list(native))


def _finest(grid):
    """The finest voxel pitch along axes with more than one cell."""
    pitches = ((grid.nx, grid.vx), (grid.ny, grid.vy), (grid.nz, grid.vz))
    return min((h for n, h in pitches if n > 1), default=math.inf)


def working_scale(grid, detector, *, keep_grid=False, magnification=1.0):
    """The bounded working grid, the detector binning and each level's binning.

    The grid keeps its six cell faces and has at most 16 M cells, unless
    ``keep_grid``; the binned detector has at most 320 pixels a side. Each
    working pixel is one ray, so its footprint at the object (its pitch over
    ``magnification``) must not exceed a voxel: otherwise voxels between the
    rays are unobserved and the image can trade them for pose. The grid is
    coarsened to the detector's binning; a kept grid instead limits that
    binning. No binning goes below one pixel, so a thin detector keeps its
    short axis and its levels share one scale.
    """
    short = min(detector.nu, detector.nv)
    dbin = min(short, max(1, math.ceil(max(detector.nu, detector.nv) / _MAX_DETECTOR)))
    # One detector pixel's footprint at the object, against the finest voxel.
    pixel = max(detector.du, detector.dv) / magnification
    if keep_grid:
        dbin = max(1, min(dbin, math.floor(_finest(grid) / pixel * (1 + 1e-9))))
    gbin, base = 1, grid
    while not keep_grid and (
        base.nx * base.ny * base.nz > _MAX_VOXELS or _finest(base) * (1 + 1e-9) < dbin * pixel
    ):
        gbin += 1
        base = _resized(grid, tuple(math.ceil(n / gbin) for n in (grid.nx, grid.ny, grid.nz)))
    bins = tuple(max(1, min(level[0], short // dbin)) for level in _LEVELS)
    return base, dbin, bins


def fit_poses(
    projections,
    nominal_poses,
    grid,
    detector,
    *,
    fit=("translations",),
    keep_grid=False,
    frames=None,
    detector_roll_deg=0.0,
):
    """Fit a volume and the selected geometry parameters to line-integral projections.

    ``projections`` are (views, nv, nu), NaN where unobserved; ``nominal_poses``
    are (views, 4, 4) world-from-object transforms. Pose units are the detector
    pitches (du, dv). Host arrays and memory maps are converted to FP32 in
    bounded view slabs, not copied as a whole stack. ``keep_grid`` fits on
    ``grid`` itself instead of a bounded resampling.
    ``frames`` supplies cone lab frames (views, 4, 3), with rows detector centre,
    unit u, unit v, source; None selects parallel rays. A shared (4, 3) frame is
    also accepted. ``fit`` selects shared centre/tilt/roll and local translation/
    rotation blocks. ``detector_roll_deg`` supplies an existing parallel detector
    roll, which is absent from the object pose matrices. Cone frames already
    contain any existing roll.
    Returns the working-scale volume (nz, ny, nx), its grid, the (views, 6)
    pose table (alpha, beta, phi in radians; dx, dz, dy along lab x, z, y,
    in the grid's length unit; dy is zero only for parallel rays) and a
    record containing the global corrections in native pixels and degrees.
    """
    started = time.perf_counter()
    with jax.enable_x64(True):
        y0 = np.asarray(projections)
        poses = np.asarray(nominal_poses, np.float32)
        if y0.ndim != 3 or y0.shape[1:] != (detector.nv, detector.nu):
            raise ValueError("projection shape does not match the detector")
        if len(y0) < 2 or poses.shape != (len(y0), 4, 4) or not np.isfinite(poses).all():
            raise ValueError("pose fitting needs finite (views, 4, 4) poses for two or more views")
        native = (float(detector.du), float(detector.dv))
        # Native pixels per radian: the object's half width, so a unit of any
        # parameter moves its edge by about one pixel.
        lever = 0.5 * max(grid.nx * grid.vx, grid.ny * grid.vy) / native[0]
        parameters = Parameters(len(y0), fit, cone=frames is not None, lever=lever)
        pose_scales = np.asarray([1 / lever] * 3 + list(native))
        if frames is None:
            validate_plane_geometry(jnp.asarray(poses), grid, detector)
        else:
            frames = np.broadcast_to(np.asarray(frames, np.float64), (len(poses), 4, 3)).copy()
            if not np.isfinite(frames).all():
                raise ValueError("cone fitting needs finite (views, 4, 3) lab frames")
            pose_scales = cone_parameter_scales(poses, frames, grid, native)
        base, dbin, bins = working_scale(
            grid, detector, keep_grid=keep_grid, magnification=_magnification(poses, frames)
        )
        padded, margin = _padded(base, bins)
        multiple = dbin * math.lcm(*bins)
        drop_v, drop_u = detector.nv % multiple, detector.nu % multiple
        measured = y0[:, : detector.nv - drop_v, : detector.nu - drop_u]
        center = (
            detector.center[0] - 0.5 * drop_u * detector.du,
            detector.center[1] - 0.5 * drop_v * detector.dv,
        )
        if frames is not None:
            frames = cropped_frames(frames, detector, high_u=drop_u, high_v=drop_v)
        gauge = (
            rigid_gauge(poses, pose_scales)
            if frames is None
            else cone_gauge(poses, pose_scales, frames)
        )
        # Angular columns need smooth interpolation as sample points rotate
        # through voxel cells. Centre/translation fits retain the cheaper linear
        # basis.
        angular = any(name in parameters.fit for name in ("tilt", "roll", "rotations"))
        interpolation = "cubic" if angular and frames is None else "linear"
        inner = 8 if angular else 4
        x = p = None
        prior = None
        levels = []
        for b, (_, outer) in zip(bins, _LEVELS, strict=True):
            y, precision, g, ops, cal = _level(
                measured,
                poses,
                padded,
                detector,
                center,
                dbin * b,
                b,
                lever=lever,
                native=native,
                frames=frames,
                scales=pose_scales,
                parameters=parameters,
                detector_roll_deg=detector_roll_deg,
                interpolation=interpolation,
            )
            # Only the angular size prior depends on the level's noise scale.
            # Reuse the same physical translation prior and its compiled solves
            # across levels when those rotation columns are absent.
            if prior is None or "rotations" in parameters.fit:
                prior = MotionPrior(
                    len(y0),
                    gauge,
                    knot_spacing=(_KNOT_SPACING,) * pose_scales.shape[-1],
                    delta=_POSE_HUBER_PX,
                    parameters=parameters,
                    rotation_scales=pose_scales,
                    rotation_sigma_deg=_ROTATION_SIGMA_DEG,
                    noise_sigma=cal["sigma"],
                )
            if x is not None and x.shape != ops.shape:
                x = jax.image.resize(x, ops.shape, method="linear", antialias=False).astype(
                    jnp.float32
                )
            x, p, y, record = solve(
                y,
                precision,
                prior,
                cal["tv_weight"],
                cal["eps"],
                ops=ops,
                budget=Budget(outer, inner),
                x0=x,
                p0=p,
                pose_sigma=cal["sigma"],
                delta=cal["delta"],
                scales=cal["scales"],
            )
            levels.append(
                {
                    "binning": dbin * b,
                    "grid": asdict(g),
                    **record,
                }
            )
        assert x is not None  # every schedule has at least one level
        p, refinement = refine(x, p, y, precision, prior, ops=ops, sigma=cal["sigma"])
        table, corrections = parameters.physical(np.asarray(p, np.float64), pose_scales)
        x = x[margin : margin + base.nz]
    record = {
        "fit": list(parameters.fit),
        "corrections": corrections,
        "rotation_prior_sigma_deg": _ROTATION_SIGMA_DEG if "rotations" in parameters.fit else None,
        "working_grid": asdict(base),
        "detector_binning": dbin,
        "lever_px_per_radian": lever,
        "ray_geometry": "parallel" if frames is None else "cone",
        "pose_dofs": [["alpha", "beta", "phi", "dx", "dz", "dy"][i] for i in parameters.local],
        "image_preconditioner": ops.projection.diagonal_kind,
        "interpolation": ops.projection.interpolation,
        "noise_estimator": "normalized_fourth_difference_mad",
        "motion_penalty": "group_student_t_innovations",
        "gauge": "rigid" if frames is None else "rigid_and_scale",
        "trimmed_pixels": {"bottom": drop_v, "right": drop_u},
        "levels": levels,
        "refinement": refinement,
        "seconds": time.perf_counter() - started,
    }
    return x, base, table, record
