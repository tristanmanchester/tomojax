"""Native-texture translation matching for axial parallel-beam scans."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import time
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
from scipy import fft, interpolate, ndimage, optimize

from tomojax._data.geometry_meta import AugmentedGeometry
from tomojax.geometry import (
    Detector,
    Grid,
    ParallelGeometry,
    grid_volume_origin,
    stack_view_poses,
)
from tomojax.recon import FBPConfig, backproject, fbp, project

from ._modes import alignment_plan
from ._pose._coupled_program import JointRefinement
from ._pose._pose_loop import align as refine_poses
from ._prealign import correlation_peaks, reprojector, shift_views

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import ScanGeometry

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionMatchingConfig:
    """Joint translation alignment of axial parallel-beam scans.

    Axial mass profiles estimate vertical motion; five native detector rows
    estimate horizontal motion by refining correlation and first-moment starts
    against one joint volume/pose likelihood, profiling detector offsets and
    slopes without discarding low-frequency texture. The full scan is retained
    without resampling.
    This estimator can fit noise or model mismatch as motion, including on
    already aligned input; reconstruction consistency alone does not establish
    pose accuracy or image preservation.

    No trajectory prior or supplied aligned reference enters inference.

    Parameters
    ----------
    iterations : int, optional
        Cap on each start's joint volume/pose refinement, not on the
        conjugate-gradient inner solve; exhausting it does not imply convergence.
        Default is 20.
    """

    iterations: int = 20


def _profile_seed(profiles: np.ndarray) -> np.ndarray:
    measured = cast("np.ndarray", ndimage.gaussian_filter1d(profiles, 1, axis=1, order=1))
    nv = measured.shape[1]
    radius = nv // 4
    lags = np.arange(-radius, radius + 1)
    size = cast("int", fft.next_fast_len(nv + radius))
    low, high = np.maximum(lags, 0), np.minimum(nv + lags, nv)
    lower, upper = np.maximum(-lags, 0), np.minimum(nv - lags, nv)
    count = high - low
    sums = np.pad(np.cumsum(measured, axis=1), ((0, 0), (1, 0)))
    squares = np.pad(np.cumsum(measured**2, axis=1), ((0, 0), (1, 0)))
    total = sums[:, high] - sums[:, low]
    variance = squares[:, high] - squares[:, low] - total**2 / count
    transformed = np.asarray(fft.rfft(measured, size, axis=1))
    reference = measured[0].copy()
    for _ in range(3):
        sr = np.r_[0, np.cumsum(reference)]
        srr = np.r_[0, np.cumsum(reference**2)]
        rsum = sr[upper] - sr[lower]
        rvar = srr[upper] - srr[lower] - rsum**2 / count
        corr = np.asarray(
            fft.irfft(transformed * np.asarray(fft.rfft(reference, size)).conj(), size, axis=1)
        )
        corr = (corr[:, lags % size] - total * rsum / count) / np.sqrt(
            np.maximum(variance * rvar, 1e-20)
        )
        peak = np.argmax(corr, axis=1)
        interior = np.clip(peak, 1, len(lags) - 2)
        rows = np.arange(len(measured))
        a, b, c = (corr[rows, interior + delta] for delta in (-1, 0, 1))
        shifts = lags[peak] + np.where(peak == interior, 0.5 * (a - c) / (a - 2 * b + c), 0)
        shifts -= shifts.mean()
        aligned = np.asarray(
            [
                ndimage.shift(p, -s, order=3, mode="constant", cval=np.nan)
                for p, s in zip(measured, shifts, strict=True)
            ]
        )
        reference = np.nan_to_num(np.nanmedian(aligned, axis=0))
    return shifts


def _vertical_shifts(projections: np.ndarray) -> np.ndarray:
    """Continuous common-profile fit, retaining a per-profile gain and affine background."""
    profiles = projections.sum(axis=2, dtype=np.float64)
    shifts = _profile_seed(profiles)
    measured = cast("np.ndarray", ndimage.gaussian_filter1d(profiles, 2, axis=1, order=1))
    nv = measured.shape[1]
    z = np.arange(nv)
    margin = nv // 8
    fixed = z[margin:-margin]
    basis = np.column_stack([np.ones(len(fixed)), np.linspace(-1, 1, len(fixed))])
    inverse = np.linalg.pinv(basis)

    def detrend(x):
        return x - basis @ (inverse @ x)

    targets = np.asarray([detrend(p[fixed]) for p in measured])
    splines = [interpolate.CubicSpline(z, p, extrapolate=False) for p in measured]
    for _ in range(3):
        aligned = np.asarray([s(z + d) for s, d in zip(splines, shifts, strict=True)])
        reference = interpolate.CubicSpline(z, np.nan_to_num(np.nanmedian(aligned, axis=0)))
        for i, target in enumerate(targets):

            def objective(s, reference=reference, target=target):
                p = detrend(reference(fixed - s))
                residual = target - (target @ p) / (p @ p) * p
                return residual @ residual

            shifts[i] = cast(
                "optimize.OptimizeResult",
                optimize.minimize_scalar(
                    objective,
                    bounds=(shifts[i] - 2, shifts[i] + 2),
                    method="bounded",
                    options={"xatol": 1e-7},
                ),
            ).x
        shifts -= shifts.mean()
    return shifts


def _xy_center(grid: Grid) -> tuple[float, float]:
    ox, oy, _ = grid_volume_origin(grid)
    return ox + (grid.nx - 1) * grid.vx / 2, oy + (grid.ny - 1) * grid.vy / 2


def _sample_rows(geometry: ParallelGeometry, data: np.ndarray, vertical: np.ndarray):
    n, nv, nu = data.shape
    detector = geometry.detector
    stride = nv // 8
    row_ids = nv // 2 + stride * np.arange(-2, 3)
    coords = row_ids[None, :] + vertical[:, None]
    indices = np.floor(coords).astype(int)[:, :, None] + np.arange(-3, 5)
    weights = np.sinc(coords[:, :, None] - indices) * np.sinc((coords[:, :, None] - indices) / 4)
    weights /= weights.sum(-1, keepdims=True)
    selected = data[np.arange(n)[:, None, None], np.clip(indices, 0, nv - 1), :]
    samples = np.einsum("nst,nstu->nsu", weights, selected).astype(np.float32)
    del selected
    centre_xy = _xy_center(geometry.grid)
    centre_z = detector.center[1] + (row_ids.mean() - (nv - 1) / 2) * detector.dv
    grid = Grid(
        nu, nu, 5, detector.du, detector.du, stride * detector.dv, vol_center=(*centre_xy, centre_z)
    )
    sampled_detector = Detector(
        nu, 5, detector.du, stride * detector.dv, center=(detector.center[0], centre_z)
    )
    sparse = ParallelGeometry(grid, sampled_detector, geometry.angles)
    return sparse, samples, row_ids


def _correlation_seed(geometry: ParallelGeometry, samples: np.ndarray) -> np.ndarray:
    """Search horizontal texture with the independently registered axial rows fixed."""
    n, nv, width = samples.shape
    original = geometry.detector
    factor = 4
    nu = width // factor
    du = factor * original.du
    centre_u = original.center[0] + original.du * (factor * nu - width) / 2
    detector = Detector(nu, nv, du, original.dv, (centre_u, original.center[1]))
    grid = Grid(
        nu,
        nu,
        nv,
        du,
        du,
        original.dv,
        vol_center=(*_xy_center(geometry.grid), original.center[1]),
    )
    coarse = ParallelGeometry(grid, detector, geometry.angles)
    nominal = stack_view_poses(coarse, n)
    reproject = reprojector(coarse, nominal, grid, detector)
    frame = np.asarray(nominal)[:, 0, :3]
    inverse = np.linalg.pinv(frame)
    measured = jnp.asarray(samples[:, :, : nu * factor].reshape(n, nv, nu, factor).mean(-1))
    shifts = np.zeros((n, 2), np.float32)
    chunks = [slice(i, min(i + 32, n)) for i in range(0, n, 32)]
    for _ in range(16):
        corrected = jnp.concatenate(
            [shift_views(measured[c], jnp.asarray(-shifts[c])) for c in chunks]
        )
        volume = fbp(coarse, grid, detector, corrected, config=FBPConfig(filter="hann"))
        peaks = np.asarray(
            jnp.concatenate(
                [
                    correlation_peaks(
                        measured[c], reproject(c, volume), max_u=max(1, nu // 4), max_v=0
                    )
                    for c in chunks
                ]
            )
        )
        horizontal = peaks[:, 0].astype(np.float64)
        horizontal -= frame @ (inverse @ horizontal)
        updated = np.column_stack([horizontal, np.zeros(n)]).astype(np.float32)
        change = float(np.max(np.abs(updated - shifts)))
        shifts = updated
        if change <= 0.05:
            break
    return shifts[:, 0] * du


def _moment_seed(geometry: ParallelGeometry, data: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """First Radon moments supply a global start, not a final pose estimate."""
    n, nv, nu = data.shape
    z = np.arange(nv)
    rows = z[nv // 4 : 3 * nv // 4]
    mass = data.sum(axis=2, dtype=np.float64)
    moment = np.einsum("nvu,u->nv", data, np.arange(nu) - (nu - 1) / 2, dtype=np.float64)
    totals = np.stack(
        [
            interpolate.CubicSpline(z, np.column_stack([mass[i], moment[i]]))(
                rows + vertical[i]
            ).sum(axis=0)
            for i in range(n)
        ]
    )
    return geometry.detector.du * totals[:, 1] / totals[:, 0] + geometry.detector.center[0]


def _horizontal_fit(
    sparse: ParallelGeometry,
    samples: np.ndarray,
    seeds: tuple[np.ndarray, np.ndarray],
    *,
    iterations: int,
):
    """Refine both physical starts against one likelihood, retaining its minimum."""
    grid, detector = sparse.grid, sparse.detector
    measured = jnp.asarray(samples)
    ones = jnp.ones((grid.nx, grid.ny, grid.nz), jnp.float32)
    bound = float(jnp.max(backproject(sparse, project(sparse, ones))))
    del ones
    plan = alignment_plan("pose", grid, freeze=("alpha", "beta", "phi", "dz"), levels=(1,))
    solver = replace(
        plan.config,
        iterations=1,
        outer_iterations=iterations,
        gn_joint_iterations=20,
        early_stop_rel_impr=1e-4,
        seed_translations=False,
        gn_volume_damping=1e-3 * bound,
    )
    nominal = np.asarray(stack_view_poses(sparse, len(samples)))
    frame = nominal[:, 0, :3]
    inverse = np.linalg.pinv(frame)
    fits = []
    for name, horizontal in zip(("correlation", "moment"), seeds, strict=True):
        seed = np.zeros((len(samples), 6), np.float32)
        seed[:, 3] = horizontal - frame @ (inverse @ horizontal)
        posed = AugmentedGeometry(sparse, seed, translation_frame="detector")
        initial = jnp.maximum(fbp(posed, grid, detector, measured, config=FBPConfig()), 0)
        volume, increment, info = refine_poses(
            posed,
            grid,
            detector,
            measured,
            cfg=solver,
            init_x=initial,
            joint_refinement=JointRefinement(ramp_damping=1e-3),
        )
        fits.append((name, seed + np.asarray(increment), info))
        del volume, initial
    best = min(fits, key=lambda fit: fit[2]["loss"][-1])
    return (
        best[1],
        best[2],
        {
            "selected": best[0],
            "losses": {name: info["loss"][-1] for name, _, info in fits},
        },
    )


def match_projections(
    geometry: ScanGeometry, projections: np.ndarray | jax.Array, *, config: ProjectionMatchingConfig
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate translations from native projection texture.

    Owns the sparse native reconstruction grid and temporary volumes. Only the
    coupled likelihood profiles detector backgrounds; input projections and the
    final inverse are unchanged.

    Parameters
    ----------
    geometry : ParallelGeometry
        Unposed axial parallel-beam geometry of the scan.
    projections : array-like
        Line integrals of shape ``(views, nv, nu)``, at least 20 rows and columns.
    config : ProjectionMatchingConfig
        Refinement settings.

    Returns
    -------
    poses : numpy.ndarray
        Per-view pose table ``(views, 6)`` in the detector translation frame.
    info : dict
        Solver record: timings, iterations and the estimator's settings.

    Raises
    ------
    ValueError
        If the geometry is not unposed axial parallel beam, or the detector is too small.
    """
    if type(geometry) is not ParallelGeometry:
        raise ValueError("projection matching requires an unposed axial ParallelGeometry")
    data = np.asarray(projections, np.float32)
    n, nv, nu = data.shape
    if min(nv, nu) < 20:
        raise ValueError("native projection matching needs at least 20 detector rows and columns")
    start = time.perf_counter()
    vertical = _vertical_shifts(data)
    sparse, samples, row_ids = _sample_rows(geometry, data, vertical)
    seeds = (_correlation_seed(sparse, samples), _moment_seed(geometry, data, vertical))
    poses, joint_info, initialization = _horizontal_fit(
        sparse, samples, seeds, iterations=config.iterations
    )
    poses[:, 4] = vertical * geometry.detector.dv
    # Choose the minimum-motion object-frame gauge using estimates only.
    nominal = np.asarray(stack_view_poses(geometry, n))
    frame = nominal[:, [0, 2], :3].reshape(-1, 3)
    translation = np.linalg.lstsq(frame, poses[:, 3:5].reshape(-1), rcond=None)[0]
    poses[:, 3:5] -= (frame @ translation).reshape(-1, 2)
    return poses, {
        "method": "projection_matching",
        "config": config,
        "estimation_seconds": time.perf_counter() - start,
        "sample_rows": row_ids.tolist(),
        "joint": dict(joint_info),
        "initialization": initialization,
        "pose_translation_frame": "detector",
    }
