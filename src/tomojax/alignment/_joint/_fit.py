"""Joint image, global-geometry and per-view motion fitting.

Bounded working scales (``AlignConfig.levels``) preserve the physical field of view.
Only parameters selected by ``fit`` have columns or unknowns; shared center, tilt and
roll form a small exact Schur border. Local motion uses a drift spline and robust
jumps; sample rotations additionally have a Gaussian prior in physical radians.

By default linear interpolation serves center/translation fits and cubic interpolation
angular fits for parallel rays; every fit's image steps use the same CG iteration
count. These are discretization/work choices for the requested model, never
data-dependent route selection. Each working pixel is one ray through its center; a
block with missing input pixels is the mean of its finite ones, weighted by their
fraction. Model and data are compared through ``residual_filter`` (K = [1, 2, 1] / 4
along u and v) at every level, so the objective does not ask the sampled model to be
right at Nyquist. Image TV is off by default, so the fit imposes no shape preference
on weak geometry modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import TYPE_CHECKING, TypedDict

import jax
import numpy as np

from tomojax import progress
from tomojax.alignment._config import AlignConfig, resolve_align_config
from tomojax.core.devices import announce
from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.registration import scaled_images
from tomojax.core.validation import validate_physical_poses
from tomojax.recon.fista_tv import resolve_tv_metric

from ._noise import MAX_SAMPLES, Windows, estimate_noise
from ._operators import Operators
from ._parameters import Parameters
from ._pose import MotionPrior, ViewOrder
from ._projection import (
    cone_parameter_scales,
    cropped_frames,
    filtered_precision,
    fit_gauge,
    residual_filter,
)
from ._refine import refine
from ._report import axial as axial_geometry, build as build_report
from ._solver import Budget, solve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike

    from ._noise import ProjectionSource
    from ._parameters import CalibrationCorrections
    from ._refine import RefinementRecord
    from ._report import AlignmentReport
    from ._solver import LevelRecord
    from ._stacks import Place

    class _LevelCalibration(TypedDict):
        tv_weight: float
        eps: float
        delta: _DataScale
        sigma: _DataScale
        scale: _DataScale
        scales: tuple[float, float, float]

    class FitLevelRecord(LevelRecord):
        binning: int
        grid: dict[str, object]

    class FitRecord(TypedDict):
        fit: list[str]
        calibration: CalibrationCorrections
        config: dict[str, object]
        working_grid: dict[str, object]
        detector_binning: int
        lever: float
        ray_geometry: str
        pose_columns: list[str]
        image_preconditioner: str
        noise_estimator: str
        motion_penalty: str
        gauge: str
        trimmed_pixels: dict[str, int]
        levels: list[FitLevelRecord]
        refinement: RefinementRecord
        offsets: list[list[float]]
        report: AlignmentReport
        seconds: float


type _Binned = tuple[np.ndarray, np.ndarray | None]
type _DataScale = float | np.float64


# Every number a user may set is a field of AlignConfig. The active model is the same at
# every level; held coordinates have neither unknowns nor derivative columns.


def _resized(grid: Grid, shape_xyz: tuple[int, ...]) -> Grid:
    """``grid`` resampled to ``shape_xyz`` cells over the same six cell faces."""
    old = np.asarray((grid.nx, grid.ny, grid.nz))
    new = np.asarray(shape_xyz, dtype=int)
    h = np.asarray((grid.vx, grid.vy, grid.vz))
    pitch = old * h / new
    origin = np.asarray(grid_volume_origin(grid)) - h / 2 + pitch / 2
    return Grid(
        *map(int, new), *map(float, pitch), vol_origin=tuple(origin), vol_center=grid.vol_center
    )


def _magnification(poses: np.ndarray, frames: np.ndarray) -> float:
    """Least detector over object distance along the cone detector normal.

    The least magnification gives the widest pixel footprint at the object, so
    a working pixel bounded by it is bounded in every view and segment.
    """
    frames = np.broadcast_to(frames, (len(poses), 4, 3))
    normal = np.cross(frames[:, 1], frames[:, 2])
    distance = np.einsum("ni,ni->n", frames[:, 0] - frames[:, 3], normal)
    depth = np.einsum("ni,ni->n", poses[:, :3, 3] - frames[:, 3], normal)
    return float(np.min(distance / depth))


def _padded(grid: Grid, bins: Sequence[int], margin: float) -> tuple[Grid, int]:
    """``grid`` extended along z past each end, and the cells added at each end.

    Moving views bring material from above and below the field into the outer
    rows; the extension gives it somewhere to be, and is cropped from the
    returned volume. Cells no ray reaches stay out of the fit. Whole
    coarse-level cells keep every level's voxel centers where they were
    relative to the detector rows. ``margin`` is the extension's share of the height.
    """
    step = math.lcm(*bins)
    cells = step * math.ceil(grid.nz * margin / step) if grid.nz > 1 else 0
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


def _read(
    measured: ProjectionSource, bins: Sequence[int], views_per_batch: int = 16
) -> tuple[list[_Binned], _DataScale, float]:
    """What the levels take from the frames, in one front-to-back pass over them.

    Returns each binning's block means over complete b x b pixel blocks and observed
    fraction, the power of two bringing the observations near one, and the native-pixel
    noise SD (``estimate_noise``). Each slab of views is converted to FP32 once and gives
    all of them: a block mean uses only its finite pixels and its precision is their
    fraction, None when every pixel is finite; the scale comes from about 16 evenly spaced
    views; the noise windows in the slab are kept from it. Nothing reads the frames at
    random or twice: random page faults on a memory map stop the kernel reading ahead on
    it for its lifetime, and a sweep after them reads a page per fault.
    """
    n, nv, nu = measured.shape
    means = [np.empty((n, nv // b, nu // b), np.float32) for b in bins]
    weights = [np.ones_like(y) for y in means]
    step = max(1, n // 16)
    peak = np.float32(0)
    noise = Windows(measured.shape)
    for s in range(0, n, views_per_batch):
        progress.check_canceled()
        src = np.asarray(measured[s : s + views_per_batch], np.float32)
        finite = np.isfinite(src)
        filled = np.where(finite, src, 0)
        for b, y, w in zip(bins, means, weights, strict=True):
            blocks = (len(src), nv // b, b, nu // b, b)
            count = finite.reshape(blocks).sum((2, 4))
            total = filled.reshape(blocks).sum((2, 4), dtype=np.float64)
            y[s : s + len(src)] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            w[s : s + len(src)] = count / (b * b)
        sample = src[-s % step :: step]
        peak = max(peak, np.max(np.abs(sample[np.isfinite(sample)]), initial=0.0))
        noise.gather(src, start=s)
        del src, finite, filled  # release this converted slab before reading the next one
    scale = 2.0 ** -np.round(np.log2(peak)) if peak > 0 else 1.0
    sd = noise.sd()
    if sd is None:  # no first-order window wholly finite: lower orders, gathered directly
        sd = estimate_noise(measured, dtype=np.float32)
    binned = [(y, None if w.min() == 1 else w) for y, w in zip(means, weights, strict=True)]
    return binned, scale, sd


def _capped_contrast(y: np.ndarray, rms: float, precision: np.ndarray | None) -> float:
    """The contrast RMS, capped at three central interdecile scales of offset-free samples."""
    total = y.size
    rng = np.random.default_rng(19871)
    indices = np.arange(total) if total <= MAX_SAMPLES else rng.integers(total, size=MAX_SAMPLES)
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
    binned: _Binned,
    scale: _DataScale,
    noise: float,
    poses: np.ndarray,
    g: Grid,
    detector: Detector,
    center: tuple[float, float],
    total_bin: int,
    *,
    lever: float,
    native: tuple[float, float],
    frames: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    parameters: Parameters | None = None,
    detector_roll: float = 0.0,
    config: AlignConfig,
) -> tuple[np.ndarray, np.ndarray | None, Grid, Operators, _LevelCalibration]:
    """Observations, grid, operators and calibration at one binning.

    The observations are multiplied by a power of two bringing them near one,
    the same at every level: the fit is scale covariant and this is exact, and
    it keeps the step's FP16 storage clear of its subnormals. Fitted volumes
    are in these units until divided by ``calibration["scale"]``. ``binned`` is this
    level's block means and observed fraction and ``noise`` the native-pixel noise SD
    (``_read``).
    """
    y, precision = binned
    y *= np.float32(scale)
    # The objective compares K A x with K y (residual_filter); the filtered data are known
    # only where every tap was measured. K acts within each view: filter on the host, a
    # slab of views at a time, as the binning above does.
    for s in range(0, len(y), 16):
        y[s : s + 16] = residual_filter(y[s : s + 16])
    precision = filtered_precision(precision)
    d = Detector(
        y.shape[2],
        y.shape[1],
        detector.du * total_bin,
        detector.dv * total_bin,
        center=center,
    )
    metric = resolve_tv_metric(
        None,
        g,
        d,
        y,
        weights=precision,
        tv_weight=config.image_tv_weight,
        huber_delta=config.image_tv_huber,
    )
    contrast = _capped_contrast(y, float(metric.contrast_scale), precision)
    # Estimate on the supplied finest samples, before coarsening turns smooth
    # texture into apparent noise. With precision=count/b**2, sigma/b gives
    # the same independent-pixel noise model for partially observed blocks.
    sigma = noise * scale / total_bin
    ops = Operators(
        g,
        d,
        poses,
        lever=lever,
        native=native,
        frames=frames,
        scales=scales,
        interpolation=config.interpolation,  # resolved by fit_poses
        parameters=parameters,
        detector_roll=detector_roll,
        detector_pivot=detector.center,
    )
    h = (g.vx * g.vy * g.vz) ** (1 / 3)
    calibration: _LevelCalibration = {
        "tv_weight": float(metric.weight_over_contrast) * contrast,
        "eps": max(1e-8, config.image_tv_huber * contrast * h / float(metric.length_scale) ** 2),
        "delta": max(config.loss_width * sigma, config.loss_contrast_floor * contrast, 1e-8),
        "sigma": sigma,
        "scale": scale,
        "scales": (1.0, 1.0, 1.0) if g.vx == g.vy == g.vz else (h / g.vz, h / g.vy, h / g.vx),
    }
    return y, precision, g, ops, calibration


def _finest(grid: Grid) -> float:
    """The finest voxel pitch along axes with more than one cell."""
    pitches = ((grid.nx, grid.vx), (grid.ny, grid.vy), (grid.nz, grid.vz))
    return min((h for n, h in pitches if n > 1), default=math.inf)


def working_scale(
    grid: Grid,
    detector: Detector,
    *,
    keep_grid: bool = False,
    magnification: float = 1.0,
    config: AlignConfig | None = None,
) -> tuple[Grid, int, tuple[int, ...]]:
    """The bounded working grid, the detector binning and each level's binning.

    The grid keeps its six cell faces and has at most ``config.max_voxels`` cells,
    unless ``keep_grid``; the binned detector has at most ``config.resolution``
    pixels a side (``AlignConfig``'s defaults when None). Each
    working pixel is one ray, so its footprint at the object (its pitch over
    ``magnification``) must not exceed a voxel: otherwise voxels between the
    rays are unobserved and the image can trade them for pose. The grid is
    coarsened to the detector's binning; a kept grid instead limits that
    binning. No binning goes below one pixel, so a thin detector keeps its
    short axis and its levels share one scale.
    """
    config = AlignConfig() if config is None else config
    short = min(detector.nu, detector.nv)
    dbin = min(short, max(1, math.ceil(max(detector.nu, detector.nv) / config.resolution)))
    # One detector pixel's footprint at the object, against the finest voxel.
    pixel = max(detector.du, detector.dv) / magnification
    if keep_grid:
        dbin = max(1, min(dbin, math.floor(_finest(grid) / pixel * (1 + 1e-9))))
    gbin, base = 1, grid
    while not keep_grid and (
        base.nx * base.ny * base.nz > config.max_voxels or _finest(base) * (1 + 1e-9) < dbin * pixel
    ):
        gbin += 1
        base = _resized(grid, tuple(math.ceil(n / gbin) for n in (grid.nx, grid.ny, grid.nz)))
    bins = tuple(max(1, min(level, short // dbin)) for level in config.levels)
    return base, dbin, bins


@dataclass(frozen=True)
class FitLevel:
    """One actual detector binning, padded grid and solve budget."""

    binning: int
    grid: Grid
    budget: Budget


@dataclass(frozen=True)
class FitSchedule:
    """Grid returned to the caller and the ordered working grids of the joint fit."""

    base: Grid
    detector_binning: int
    margin: int
    levels: tuple[FitLevel, ...]


def fit_schedule(
    grid: Grid,
    detector: Detector,
    cone: tuple[np.ndarray, np.ndarray] | None,
    *,
    keep_grid: bool,
    config: AlignConfig,
) -> FitSchedule:
    """Plan independently of fitted corrections, from declared (poses, frames) for cone."""
    base, dbin, bins = working_scale(
        grid,
        detector,
        keep_grid=keep_grid,
        magnification=1.0 if cone is None else _magnification(*cone),
        config=config,
    )
    padded, margin = _padded(base, bins, config.margin)
    iterations = (config.coarse_iterations,) * (len(bins) - 1) + (config.iterations,)
    levels = tuple(
        FitLevel(
            dbin * b,
            _resized(padded, tuple(math.ceil(n / b) for n in (padded.nx, padded.ny, padded.nz))),
            Budget.of(config, outer),
        )
        for b, outer in zip(bins, iterations, strict=True)
    )
    return FitSchedule(base, dbin, margin, levels)


def _physical_frame(
    poses: np.ndarray,
    grid: Grid,
    native: tuple[float, float],
    lever: float,
    frames: ArrayLike | jax.Array | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The pose scales, and the per-view lab frame for cone-beam scans."""
    if frames is None:
        validate_physical_poses(poses, "tj.align", rigid=True)
        return np.asarray([1 / lever] * 3 + list(native)), None
    frames = np.broadcast_to(np.asarray(frames, np.float64), (len(poses), 4, 3)).copy()
    if not np.isfinite(frames).all():
        raise ValueError("cone fitting needs finite (views, 4, 3) lab frames")
    return cone_parameter_scales(poses, frames, grid, native), frames


def _fit_inputs(
    projections: ArrayLike | jax.Array, nominal_poses: ArrayLike | jax.Array, detector: Detector
) -> tuple[np.ndarray, np.ndarray]:
    """Read the projection stack without copying it and validate the matching pose table."""
    y0 = np.asanyarray(projections)
    poses = np.asarray(nominal_poses, np.float32)
    if y0.ndim != 3 or y0.shape[1:] != (detector.nv, detector.nu):
        raise ValueError("projection shape does not match the detector")
    if len(y0) < 2 or poses.shape != (len(y0), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("pose fitting needs finite (views, 4, 4) poses for two or more views")
    return y0, poses


def fit_poses(
    projections: ArrayLike | jax.Array,
    nominal_poses: ArrayLike | jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    fit: str | tuple[str, ...] = ("translations",),
    keep_grid: bool = False,
    frames: ArrayLike | jax.Array | None = None,
    detector_roll: float = 0.0,
    view_positions: ArrayLike | None = None,
    declared_poses: np.ndarray | None = None,
    initial_center: float = 0.0,
    schedule: FitSchedule | None = None,
    placement: Mapping[str, Place] | None = None,
    config: AlignConfig | None = None,
) -> tuple[jax.Array, Grid, np.ndarray | None, FitRecord]:
    """Fit a volume and the selected geometry parameters to line-integral projections.

    ``projections`` are (views, nv, nu), NaN where unobserved; ``nominal_poses``
    are (views, 4, 4) world-from-object transforms. Pose units are the detector
    pitches (du, dv). Host arrays and memory maps are converted to FP32 in
    bounded view slabs, not copied as a whole stack. ``keep_grid`` fits on
    ``grid`` itself instead of a bounded resampling.
    ``frames`` supplies cone lab frames (views, 4, 3), with rows detector center,
    unit u, unit v, source; None selects parallel rays. A shared (4, 3) frame is
    also accepted. ``initial_center`` is the measured detector-u correction in native
    pixels at which to start a selected center fit. ``fit`` selects shared
    center/tilt/roll and local translation/rotation blocks. ``detector_roll``
    supplies an existing parallel detector
    roll, which is absent from the object pose matrices. Cone frames already
    contain any existing roll. ``view_positions`` place the views in acquisition
    time (see :class:`ViewOrder`); the motion prior and the report's neighbor
    tests follow them, and None means consecutive. ``declared_poses`` are the
    geometry's poses before any corrections, which set a cone's magnification and so
    the working scale: the problem's size is a property of the scan as declared, so a
    start or restart from a different pose table fits on the same grid. None uses
    ``nominal_poses``, which are the declared ones when no corrections are attached.
    ``placement`` forces the levels' stacks to "device" or "host" by name
    (``_stacks.ORDER``); None plans each level from the device's room. ``config`` is
    the ``AlignConfig`` (None: its defaults).
    Returns the working-scale volume (nz, ny, nx), its grid, the (views, 6)
    pose table (alpha, beta, phi in radians; dx, dz, dy along lab x, z, y,
    in the grid's length unit; dy is zero only for parallel rays), or None when
    ``fit`` selects no per-view parameters, and a
    record with the change in each fitted global, in native pixels and degrees, under
    ``calibration``.
    """
    started = time.perf_counter()
    config = AlignConfig() if config is None else config
    announce()
    with jax.enable_x64(True):
        y0, poses = _fit_inputs(projections, nominal_poses, detector)
        native = (float(detector.du), float(detector.dv))
        # Native pixels per radian at the object edge.
        lever = 0.5 * max(grid.nx * grid.vx, grid.ny * grid.vy) / native[0]
        parameters = Parameters(len(y0), fit, cone=frames is not None, lever=lever)
        pose_scales, frames = _physical_frame(poses, grid, native, lever, frames)
        if schedule is None:
            declared = poses if declared_poses is None else declared_poses
            cone = None if frames is None else (declared, frames)
            schedule = fit_schedule(grid, detector, cone, keep_grid=keep_grid, config=config)
        base, dbin, margin = schedule.base, schedule.detector_binning, schedule.margin
        bins = [level.binning for level in schedule.levels]
        multiple = math.lcm(*bins)
        drop_v, drop_u = detector.nv % multiple, detector.nu % multiple
        measured = y0[:, : detector.nv - drop_v, : detector.nu - drop_u]
        center = (
            detector.center[0] - 0.5 * drop_u * detector.du,
            detector.center[1] - 0.5 * drop_v * detector.dv,
        )
        if frames is not None:
            frames = cropped_frames(frames, detector, high_u=drop_u, high_v=drop_v)
        gauge, constraint = fit_gauge(poses, pose_scales, parameters.fit, detector.center, frames)
        config = resolve_align_config(config, parameters.fit, cone=frames is not None)
        x = last = None
        p = None
        if "center_u" in parameters.globals:
            p = np.zeros(parameters.size, np.float64)
            p[parameters.local_size] = initial_center
        order = ViewOrder(len(y0), view_positions)
        prior = None
        levels: list[FitLevelRecord] = []
        binned, scale, noise = _read(measured, bins)
        for level in schedule.levels:
            last = None  # Release the previous level before building the next one.
            with (
                scaled_images(scale),
                progress.task(
                    "level", f"resolution 1/{level.binning}", level.budget.iterations, "steps"
                ),
            ):
                y, precision, g, ops, cal = _level(
                    binned.pop(0),
                    scale,
                    noise,
                    poses,
                    level.grid,
                    detector,
                    center,
                    level.binning,
                    lever=lever,
                    native=native,
                    frames=frames,
                    scales=pose_scales,
                    parameters=parameters,
                    detector_roll=detector_roll,
                    config=config,
                )
                # Without angular columns, reuse the translation prior across levels.
                if prior is None or "rotations" in parameters.fit:
                    prior = MotionPrior(
                        len(y0),
                        gauge,
                        constraint=constraint,
                        knot_spacing=(config.knot_spacing,) * pose_scales.shape[-1],
                        delta=config.slip_scale,
                        independent=config.motion == "independent",
                        parameters=parameters,
                        rotation_scales=pose_scales,
                        rotation_sigma_deg=config.rotation_sigma,
                        noise_sigma=cal["sigma"],
                        order=order,
                    )
                x, p, y, record = solve(
                    y,
                    precision,
                    prior,
                    cal["tv_weight"],
                    cal["eps"],
                    ops=ops,
                    budget=level.budget,
                    x0=x,
                    p0=p,
                    pose_sigma=cal["sigma"],
                    delta=cal["delta"],
                    scales=cal["scales"],
                    placement=placement,
                )
                last = x, p, y, precision, ops, cal
                levels.append({"binning": level.binning, "grid": asdict(g), **record})
        if last is None or prior is None:
            raise ValueError("expected a fit schedule with at least one level")
        x, p, y, precision, ops, cal = last
        del last
        total = config.refine_iterations if prior.nlocal else 0
        with scaled_images(scale), progress.task("refine", "refine poses", total, "updates"):
            p, refinement, (offsets, energy) = refine(
                x,
                p,
                y,
                precision,
                prior,
                ops=ops,
                sigma=cal["sigma"],
                last=levels[-1],
                placement=placement,
                config=config,
            )
        table, changes = parameters.physical(np.asarray(p, np.float64), pose_scales)
        x = x[margin : margin + base.nz] / np.float32(cal["scale"])
    info: FitRecord = {
        "fit": list(parameters.fit),
        "calibration": changes,
        "config": asdict(config),
        "working_grid": asdict(base),
        "detector_binning": dbin,
        "lever": lever * math.pi / 180,  # native pixels per degree
        "ray_geometry": "parallel" if frames is None else "cone",
        "pose_columns": [["alpha", "beta", "phi", "x", "z", "y"][i] for i in parameters.local],
        "image_preconditioner": ops.projection.diagonal_kind,
        "noise_estimator": "normalized_fourth_difference_mad",
        "motion_penalty": "group_student_t_innovations",
        "gauge": "rigid" if frames is None else "rigid_and_scale",
        "trimmed_pixels": {"bottom": drop_v, "right": drop_u},
        "levels": levels,
        "refinement": refinement,
        # Fitted for information: each view's offset and ramp (top row -1, bottom +1) less
        # the level all views share, in the projections' units.
        "offsets": (offsets / cal["scale"]).tolist(),
        "report": build_report(
            table=table,
            local=parameters.local,
            fit=parameters.fit,
            axial=axial_geometry(
                poses, (0.0, 1.0, 0.0) if frames is None else frames[0, 0] - frames[0, 3]
            ),
            calibration=changes,
            native=native,
            binning=dbin,
            levels=levels,
            refinement=refinement,
            offsets=offsets / cal["scale"],
            energy=energy,
            spacing=config.knot_spacing,
            order=order,
        ),
        "seconds": time.perf_counter() - started,
    }
    return x, base, table if parameters.local else None, info
