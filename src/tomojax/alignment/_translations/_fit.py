"""Joint fit of a volume and per-view detector translations, then edge refinement.

The fit runs on a bounded working scale: at most 16 M volume cells and 320
detector pixels a side, keeping every face of the volume's cells (so the field
of view is unchanged and pitches may become unequal), and two resolution levels
(2x then 1x binning of that scale). Translations are always in native detector
pixels, so the prior's widths and the gauge mean the same at every scale. The
data loss is quadratic out to max(2.5 sigma, 0.5 contrast) and logarithmic
beyond; the volume prior is anisotropic physical Huber TV.
"""

from dataclasses import asdict
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.joseph import plane_coefficients, validate_plane_geometry
from tomojax.recon.fista_tv import estimate_projection_noise, resolve_tv_metric

from ._operators import Operators
from ._pose import MotionPrior
from ._refine import refine
from ._solver import Budget, solve

_MAX_VOXELS = 16_000_000
_MAX_DETECTOR = 320
# (binning, outer Gauss-Newton steps, CG iterations) per level.
_LEVELS = ((2, 12, 4), (1, 8, 4))
_LOSS_SIGMA = 2.5
_LOSS_CONTRAST = 0.5
_TV_WEIGHT = 5e-5
_TV_HUBER = 1.0
_KNOT_SPACING = 10
_POSE_HUBER_PX = 0.02


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


def _coarsened(measured, b, batch_views=16):
    """Block means over complete b x b pixel blocks, and the observed fraction.

    Precision is None when every pixel is finite; otherwise each block mean uses
    only its finite pixels and its precision is their fraction.
    """
    n, nv, nu = measured.shape
    y = np.empty((n, nv // b, nu // b), np.float32)
    w = np.ones_like(y)
    missing = any(
        not np.isfinite(measured[s : s + batch_views]).all() for s in range(0, n, batch_views)
    )
    for s in range(0, n, batch_views):
        src = np.asarray(measured[s : s + batch_views], np.float32)
        blocks = (len(src), nv // b, b, nu // b, b)
        if not missing:
            y[s : s + len(src)] = src.reshape(blocks).mean((2, 4))
            continue
        finite = np.isfinite(src)
        count = finite.reshape(blocks).sum((2, 4))
        total = np.where(finite, src, 0).reshape(blocks).sum((2, 4), dtype=np.float64)
        y[s : s + len(src)] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        w[s : s + len(src)] = count / (b * b)
    return y, (w if missing else None)


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


def _level(measured, poses, base, detector, center, total_bin, b, native):
    """Observations, grid, detector, coefficients and calibration at one binning."""
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
    cf = np.asarray(plane_coefficients(jnp.asarray(poses, jnp.float64), g, d), np.float32)
    metric = resolve_tv_metric(
        None, g, d, y, weights=precision, tv_weight=_TV_WEIGHT, huber_delta=_TV_HUBER
    )
    contrast = _capped_contrast(y, float(metric.contrast_scale), precision)
    sigma = float(estimate_projection_noise(y, weights=precision))
    ops = Operators(g, d, len(y), ratios=(native[0] / d.du, native[1] / d.dv))
    weights = ops.pad_images(np.ones_like(y) if precision is None else precision)
    noise_weight = sigma * float(ops.noise_rms(ops.pad_coefficients(cf), weights))
    h = (g.vx * g.vy * g.vz) ** (1 / 3)
    calibration = {
        "tv_weight": max(float(metric.weight_over_contrast) * contrast, noise_weight),
        "eps": max(1e-8, _TV_HUBER * contrast * h / float(metric.length_scale) ** 2),
        "delta": max(_LOSS_SIGMA * sigma, _LOSS_CONTRAST * contrast, 1e-8),
        "sigma": sigma,
        "scales": (1.0, 1.0, 1.0) if g.vx == g.vy == g.vz else (h / g.vz, h / g.vy, h / g.vx),
    }
    return y, precision, g, cf, ops, calibration


def working_scale(grid, detector, *, keep_grid=False):
    """The bounded working grid, the detector binning and each level's binning.

    The grid keeps its six cell faces and has at most 16 M cells, unless
    ``keep_grid``; the binned detector has at most 320 pixels a side. No
    binning goes below one pixel, so a thin detector keeps its short axis and
    its levels share one scale.
    """
    short = min(detector.nu, detector.nv)
    dbin = min(short, max(1, math.ceil(max(detector.nu, detector.nv) / _MAX_DETECTOR)))
    gbin = 1
    while (
        not keep_grid
        and math.prod(math.ceil(n / gbin) for n in (grid.nx, grid.ny, grid.nz)) > _MAX_VOXELS
    ):
        gbin += 1
    base = _resized(grid, tuple(math.ceil(n / gbin) for n in (grid.nx, grid.ny, grid.nz)))
    bins = tuple(max(1, min(level[0], short // dbin)) for level in _LEVELS)
    return base, dbin, bins


def fit_translations(
    projections, nominal_poses, grid, detector, *, native_pixel_size=None, keep_grid=False
):
    """Fit a volume and per-view detector translations to line-integral projections.

    ``projections`` are (views, nv, nu), NaN where unobserved; ``nominal_poses``
    are (views, 4, 4) world-from-object transforms. ``native_pixel_size`` gives
    the original detector pitches (du, dv) when the input was already binned;
    ``keep_grid`` fits on ``grid`` itself instead of a bounded resampling.
    Returns the working-scale volume (nz, ny, nx), its grid, the (views, 2)
    translations in native detector pixels along u and v, and a record.
    """
    started = time.perf_counter()
    with jax.enable_x64(True):
        y0 = np.asarray(projections, np.float32)
        poses = np.asarray(nominal_poses, np.float32)
        if y0.ndim != 3 or y0.shape[1:] != (detector.nv, detector.nu):
            raise ValueError("projection shape does not match the detector")
        if len(y0) < 2 or poses.shape != (len(y0), 4, 4) or not np.isfinite(poses).all():
            raise ValueError(
                "translation fitting needs finite (views, 4, 4) poses for two or more views"
            )
        if not np.isfinite(y0).any():
            raise ValueError("no observed pixels")
        validate_plane_geometry(jnp.asarray(poses), grid, detector)
        native = tuple(map(float, native_pixel_size or (detector.du, detector.dv)))
        base, dbin, bins = working_scale(grid, detector, keep_grid=keep_grid)
        multiple = dbin * math.lcm(*bins)
        drop_v, drop_u = detector.nv % multiple, detector.nu % multiple
        measured = y0[:, : detector.nv - drop_v, : detector.nu - drop_u]
        center = (
            detector.center[0] - 0.5 * drop_u * detector.du,
            detector.center[1] - 0.5 * drop_v * detector.dv,
        )
        gauge = (
            poses[:, [0, 2], :3].astype(np.float64)
            * np.asarray([1.0, native[0] / native[1]])[None, :, None]
        )
        prior = MotionPrior(
            len(y0), gauge.reshape(-1, 3), knot_spacing=_KNOT_SPACING, delta=_POSE_HUBER_PX
        )
        x = p = None
        levels = []
        for b, (_, outer, inner) in zip(bins, _LEVELS, strict=True):
            y, precision, g, cf, ops, cal = _level(
                measured, poses, base, detector, center, dbin * b, b, native
            )
            if x is not None and x.shape != ops.shape:
                x = jax.image.resize(x, ops.shape, method="linear", antialias=False).astype(
                    jnp.float32
                )
            x, p, record = solve(
                y,
                cf,
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
            levels.append({"binning": dbin * b, "grid": asdict(g), **record})
        assert x is not None  # every schedule has at least one level
        p, refinement = refine(x, p, y, precision, cf, prior, ops=ops, sigma=cal["sigma"])
        shifts = np.asarray(p, np.float64)
    record = {
        "working_grid": asdict(base),
        "detector_binning": dbin,
        "trimmed_pixels": {"bottom": drop_v, "right": drop_u},
        "levels": levels,
        "refinement": refinement,
        "seconds": time.perf_counter() - started,
    }
    return x, base, shifts, record
