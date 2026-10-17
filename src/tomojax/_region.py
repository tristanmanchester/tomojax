"""A data-selected reconstruction region for calibrated laminography scans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal.windows import tukey

from tomojax._scan import Scan, describe, with_grid
from tomojax.backends import device_free_memory_bytes
from tomojax.geometry import Grid
from tomojax.geometry.api import stack_view_poses
from tomojax.recon import backproject, project
from tomojax.recon.api import reconstruct_arrays

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax._typed_arrays import Device


@dataclass(frozen=True)
class RegionEstimate:
    """The resized scan and the evidence/uncertainty behind its reconstruction grid.

    Projection samples, detector calibration and per-view poses are unchanged.
    ``info`` distinguishes a conditionally selected region from a broad survey;
    neither is a certificate of the object's physical support.

    Parameters
    ----------
    scan : Scan
        Scan with the proposed reconstruction grid; measurements are unchanged.
    info : mapping
        Selection evidence, uncertainty, memory budget and limitations.
    """

    scan: Scan
    info: Mapping[str, object]

    @property
    def grid(self) -> Grid:
        """The proposed reconstruction grid, also carried by ``scan``.

        Returns
        -------
        Grid
            Proposed reconstruction grid carried by the scan.
        """
        return self.scan.grid


@dataclass(frozen=True)
class _Evidence:
    grid: Grid
    z: np.ndarray
    focus: np.ndarray
    height: float
    isolated: bool
    fwhm: float
    competitor: float


@dataclass(frozen=True)
class _Score:
    width: float
    mse: np.ndarray
    hp_mse: np.ndarray
    seconds: float


def _detector_corners(scan: Scan) -> np.ndarray:
    d = scan.detector
    roll = np.deg2rad(float(getattr(scan.geometry, "detector_roll_deg", 0.0)))
    c, s = np.cos(roll), np.sin(roll)
    offsets = np.array(
        [
            [u, v]
            for u in (-d.nu * d.du / 2, d.nu * d.du / 2)
            for v in (-d.nv * d.dv / 2, d.nv * d.dv / 2)
        ]
    )
    coordinates = offsets @ np.array([[c, s], [-s, c]]) + np.asarray(d.center)
    return np.column_stack([coordinates[:, 0], np.zeros(4), coordinates[:, 1]])


def _posed_rays(scan: Scan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(stack_view_poses(scan.geometry, len(scan.angles)), np.float64)
    rotation, shift = matrix[:, :3, :3], matrix[:, :3, 3]
    origins = np.einsum("nci,nij->ncj", _detector_corners(scan)[None] - shift[:, None], rotation)
    return origins, rotation[:, 1, :], matrix


def _ray_grid(scan: Scan, height: float, width: float, pitch: float) -> Grid:
    origins, beam, _ = _posed_rays(scan)
    intersections = np.stack(
        [
            origins + (z - origins[:, :, 2])[:, :, None] / beam[:, None, 2, None] * beam[:, None]
            for z in (height - width / 2, height + width / 2)
        ]
    )
    radius = float(np.max(np.linalg.norm(intersections[..., :2], axis=-1)))
    side = int(np.ceil((2 * radius / pitch + 4) / 8)) * 8
    depth = int(np.ceil(width / pitch / 8)) * 8
    return Grid(side, side, depth, pitch, pitch, pitch, vol_center=(0, 0, height))


def _broad_grid(scan: Scan, pitch: float) -> Grid:
    _, beam, matrix = _posed_rays(scan)
    vertical = matrix[:, 2, 2]
    if np.any(np.isclose(beam[:, 2], 0)) or np.any(np.isclose(vertical, 0)):
        raise ValueError("estimate_region needs an oblique laminography axis, not axial tomography")
    corners = _detector_corners(scan)
    bounds = (corners[None, :, 2] - matrix[:, None, 2, 3]) / vertical[:, None]
    half_depth = float(np.max(np.abs(bounds)))
    depth = int(np.ceil(2 * half_depth / pitch / 8)) * 8
    return _ray_grid(scan, 0, depth * pitch, pitch)


def _focus_evidence(scan: Scan, voxel_budget: float) -> _Evidence:
    count = min(64, len(scan.angles))
    indices = np.linspace(0, len(scan.angles) - 1, count).astype(int)
    small = scan.selected(indices).binned(2)
    d = small.detector
    pitch = 2 * max(d.du, d.dv)
    grid = _broad_grid(small, pitch)
    while grid.nx * grid.ny * grid.nz > voxel_budget:
        pitch *= 2
        grid = _broad_grid(small, pitch)
    window = (tukey(d.nv, 1)[:, None] * tukey(d.nu, 1)[None]).astype(np.float32)
    volume = np.asarray(
        reconstruct_arrays("fbp", small.geometry, grid, small.detector, small.projections * window)[
            0
        ]
    )
    xy = (np.arange(grid.nx) - (grid.nx - 1) / 2) * grid.vx
    disk = xy[:, None] ** 2 + xy[None, :] ** 2 < (d.nu * d.du / 4) ** 2
    gradient = np.gradient(volume, grid.vx, axis=0) ** 2 + np.gradient(volume, grid.vy, axis=1) ** 2
    energy = gaussian_filter1d(
        np.sum(gradient * disk[..., None], axis=(0, 1), dtype=np.float64), 0.5
    )
    z = (np.arange(grid.nz) - (grid.nz - 1) / 2) * grid.vz
    k = int(np.argmax(energy))
    lo = hi = k
    while lo > 0 and energy[lo - 1] >= energy[k] / 2:
        lo -= 1
    while hi + 1 < len(z) and energy[hi + 1] >= energy[k] / 2:
        hi += 1
    width = float(z[hi] - z[lo] + grid.vz)
    outside = np.abs(z - z[k]) > 2 * width
    competitor = float(np.max(energy[outside], initial=0) / energy[k]) if energy[k] > 0 else 1.0
    isolated = bool(energy[k] > 0 and lo > 0 and hi < len(z) - 1 and competitor <= 0.25)
    return _Evidence(grid, z, energy, float(z[k]), isolated, width, competitor)


def _score_supports(scan: Scan, evidence: _Evidence) -> list[_Score]:
    count = min(64, len(scan.angles) // 2)
    indices = np.linspace(0, len(scan.angles) - 1, 2 * count).astype(int)
    fit, validation = scan.selected(indices[::2]).binned(2), scan.selected(indices[1::2]).binned(2)
    pitch = evidence.grid.vz
    span = evidence.grid.nz * pitch
    grid = _ray_grid(scan, 0, span, pitch)
    z = (np.arange(grid.nz) - (grid.nz - 1) / 2) * pitch
    center = pitch * round(evidence.height / pitch)
    data, target = jnp.asarray(fit.projections), np.asarray(validation.projections)
    sigma = (0, 1.5 * pitch / validation.detector.dv, 1.5 * pitch / validation.detector.du)

    def high(a: np.ndarray) -> np.ndarray:
        return a - gaussian_filter(a, sigma)

    target_hp = high(target)
    zeros = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)

    def forward(x: jax.Array) -> jax.Array:
        return project(fit.geometry, x, grid=grid)

    def adjoint(y: jax.Array) -> jax.Array:
        return backproject(fit.geometry, y, grid=grid)

    validate = jax.jit(lambda x: project(validation.geometry, x, grid=grid))

    @jax.jit
    def solve(mask: jax.Array) -> jax.Array:
        support = jnp.broadcast_to(mask, zeros.shape)
        bound = 1.2 * jnp.max(adjoint(forward(support)) * support)

        def step(
            _: int, state: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            previous, extrapolated, momentum = state
            gradient = adjoint(forward(extrapolated) - data) * mask
            current = jnp.maximum(extrapolated - gradient / bound, 0) * mask
            next_momentum = (1 + jnp.sqrt(1 + 4 * momentum * momentum)) / 2
            extrapolated = (current + (momentum - 1) / next_momentum * (current - previous)) * mask
            return current, extrapolated, next_momentum

        return jax.lax.fori_loop(0, 60, step, (zeros, zeros, jnp.float32(1)))[0]

    records: list[_Score] = []

    def evaluate(width: float) -> None:
        mask = (
            np.ones(grid.nz, np.float32)
            if width >= span
            else (np.abs(z - center) <= width / 2).astype(np.float32)
        )
        started = time.perf_counter()
        prediction = np.asarray(validate(solve(jnp.asarray(mask[None, None]))))
        mse = np.mean((prediction - target) ** 2, axis=(1, 2), dtype=np.float64)
        hp_mse = np.mean((high(prediction) - target_hp) ** 2, axis=(1, 2), dtype=np.float64)
        records.append(
            _Score(float(width), np.asarray(mse), np.asarray(hp_mse), time.perf_counter() - started)
        )

    widths = sorted({pitch * n for n in (4, 8, 16, 32, 64) if pitch * n < span} | {span})
    for width in widths:
        evaluate(width)
    ranking = np.argsort([r.mse.mean() for r in records])
    if len(ranking) > 1 and records[ranking[0]].width < span:
        first, second = (records[i].width for i in ranking[:2])
        midpoint = pitch * round((first + second) / (2 * pitch))
        if midpoint not in widths:
            evaluate(midpoint)
    return records


def _select_width(
    records: list[_Score], span: float
) -> tuple[float | None, list[dict[str, object]]]:
    rows: list[dict[str, object]] = [
        {
            "width": r.width,
            "mse": float(r.mse.mean()),
            "highpass_mse": float(r.hp_mse.mean()),
            "seconds": r.seconds,
        }
        for r in records
    ]
    for name in ("mse", "hp_mse"):
        values = np.array([getattr(r, name) for r in records])
        best = np.argmin(values.mean(axis=1))
        for row, difference in zip(rows, values - values[best], strict=True):
            se = difference.std(ddof=1) / np.sqrt(len(difference))
            row[name + "_within_one_se"] = bool(difference.mean() <= se)
    choices = [
        r.width
        for r, row in zip(records, rows, strict=True)
        if row["mse_within_one_se"] and row["hp_mse_within_one_se"] and r.width < span
    ]
    return (min(choices) if choices else None), rows


def _working_device(scan: Scan) -> Device:
    if isinstance(scan.projections, jax.Array):
        devices = scan.projections.devices()
        if len(devices) == 1:
            return next(iter(devices))
    return next(iter(jnp.zeros(()).devices()))


def estimate_region(scan: Scan) -> RegionEstimate:
    """Select a reconstruction region from an already calibrated laminography scan.

    A broad FBP preview tests axial focus isolation. Sixty-step positive reconstructions
    on balanced fitting/validation views then compare candidate depths using ordinary
    and high-pass projection error. Disagreement or multiple strong focus peaks returns
    a broad detector-derived survey instead of inventing a thin object.

    Detector-pitch isotropic voxels are preferred. The output may be coarser to budget
    eight volume work arrays plus measured data on the working device; ``info`` reports
    this explicitly. The region is a conditional reconstruction choice, not physical
    support or an accuracy certificate. Inspect the reconstruction and independent views.
    No measured projections, poses, detector calibration or solver defaults are changed.

    Parameters
    ----------
    scan : Scan
        Calibrated oblique laminography scan of corrected line integrals.

    Returns
    -------
    RegionEstimate
        Resized scan and evidence distinguishing a selected region from a broad survey.

    Raises
    ------
    ValueError
        If the geometry is not calibrated oblique laminography.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("laminography.nxs")  # doctest: +SKIP
    >>> region = tj.estimate_region(scan)  # doctest: +SKIP
    >>> recon = tj.reconstruct(region.scan, "cgls")  # doctest: +SKIP
    """
    if describe(scan.geometry)[0] != "lamino":
        raise ValueError("estimate_region requires a calibrated laminography scan")
    started = time.perf_counter()
    device = _working_device(scan)
    with jax.default_device(device):
        available = device_free_memory_bytes(device=device) or 4 * 2**30
        resident = isinstance(scan.projections, jax.Array) and scan.projections.devices() == {
            device
        }
        # Resident data is already charged to free memory; large host scans can stream.
        data_bytes = 0 if resident else min(scan.projections.size * 4, available / 4)
        budget = (0.6 * available - data_bytes) / 32
        evidence = _focus_evidence(scan, budget)
        span = evidence.grid.nz * evidence.grid.vz
        width, probes = None, []
        if evidence.isolated:
            width, probes = _select_width(_score_supports(scan, evidence), span)
        pitch = max(scan.detector.du, scan.detector.dv)
        height = pitch * round(evidence.height / pitch) if width is not None else 0.0
        depth = width if width is not None else span
        grid = _ray_grid(scan, height, depth, pitch)
        spacing = pitch
        while grid.nx * grid.ny * grid.nz > budget:
            spacing *= 2
            grid = _ray_grid(scan, height, depth, spacing)
    info = {
        "status": "conditional_cv_region" if width is not None else "broad_uncertain",
        "isolated": evidence.isolated,
        "focus_fwhm": evidence.fwhm,
        "competitor_ratio": evidence.competitor,
        "focus_z": evidence.z.tolist(),
        "focus": evidence.focus.tolist(),
        "probes": probes,
        "requested_pitch": pitch,
        "selected_pitch": spacing,
        "preview_pitch": evidence.grid.vz,
        "grid": asdict(grid),
        "seconds": time.perf_counter() - started,
        "warning": (
            "Conditional reconstruction region, not a physical-support or accuracy certificate."
        ),
    }
    return RegionEstimate(with_grid(scan, grid), info)
