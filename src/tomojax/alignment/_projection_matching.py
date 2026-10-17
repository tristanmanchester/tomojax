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
from tomojax.core.array_ops import mean_detector_pixels
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
from ._prealign import estimate_view_shifts

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import ScanGeometry

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionMatchingConfig:
    """Joint translation alignment of axial parallel-beam scans.

    Axial mass profiles estimate vertical motion; five native detector rows
    estimate horizontal motion using a positive joint volume/pose solve with
    a matched high-pass objective. The full scan is retained without resampling.
    This estimator can fit noise or model mismatch as motion, including on
    already aligned input; reconstruction consistency alone does not establish
    pose accuracy or image preservation.

    No trajectory prior or supplied aligned reference enters inference.

    Parameters
    ----------
    iterations : int, optional
        Cap on joint volume/pose refinement, not on the conjugate-gradient inner
        solve; exhausting it does not imply convergence. Default is 20.
    """

    iterations: int = 20


def _profile_seed(profiles: np.ndarray) -> np.ndarray:
    measured = np.asarray(ndimage.gaussian_filter1d(profiles, 1, axis=1, order=1))
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
    measured = np.asarray(ndimage.gaussian_filter1d(profiles, 2, axis=1, order=1))
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


def _capture(geometry: ParallelGeometry, data: np.ndarray) -> np.ndarray:
    _, nv, nu = data.shape
    detector = geometry.detector
    # The established global capture search uses block means only for its seed;
    # all subsequent updates retain the native detector texture.
    centre_xy = _xy_center(geometry.grid)
    factor = 4
    binned = mean_detector_pixels(data, factor)
    cu = detector.center[0] + detector.du * (factor * (nu // factor) - nu) / 2
    cv = detector.center[1] + detector.dv * (factor * (nv // factor) - nv) / 2
    coarse_detector = Detector(
        nu // factor, nv // factor, factor * detector.du, factor * detector.dv, (cu, cv)
    )
    coarse_grid = Grid(
        nu // factor,
        nu // factor,
        nv // factor,
        factor * detector.du,
        factor * detector.du,
        factor * detector.dv,
        vol_center=(*centre_xy, 0),
    )
    coarse = ParallelGeometry(coarse_grid, coarse_detector, geometry.angles)
    seed = estimate_view_shifts(coarse, coarse_grid, coarse_detector, binned)
    poses = np.zeros((len(data), 6), np.float32)
    poses[:, 3] = seed[:, 0]
    return poses


def match_projections(
    geometry: ScanGeometry, projections: np.ndarray | jax.Array, *, config: ProjectionMatchingConfig
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate translations from native projection texture.

    Owns the sparse native reconstruction grid and temporary volumes. Only the
    coupled objective is filtered; input projections and final inverse are not.

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
    grid, detector = sparse.grid, sparse.detector
    seed = _capture(geometry, data)
    posed = AugmentedGeometry(sparse, seed, translation_frame="detector")
    measured = jnp.asarray(samples)
    initial = jnp.maximum(fbp(posed, grid, detector, measured, config=FBPConfig()), 0)
    ones = jnp.ones((grid.nx, grid.ny, grid.nz), jnp.float32)
    bound = float(jnp.max(backproject(sparse, project(sparse, ones))))
    del ones
    plan = alignment_plan("pose", grid, freeze=("alpha", "beta", "phi", "dz"), levels=(1,))
    solver = replace(
        plan.config,
        iterations=1,
        outer_iterations=config.iterations,
        gn_joint_iterations=20,
        early_stop_rel_impr=1e-4,
        seed_translations=False,
        gn_volume_damping=1e-3 * bound,
    )
    volume, increment, joint_info = refine_poses(
        posed,
        grid,
        detector,
        measured,
        cfg=solver,
        init_x=initial,
        joint_refinement=JointRefinement(highpass_cycles=5.12, ramp_damping=1e-3),
    )
    poses = seed + np.asarray(increment)
    poses[:, 4] = vertical * geometry.detector.dv
    del volume, initial, measured
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
        "pose_translation_frame": "detector",
    }
