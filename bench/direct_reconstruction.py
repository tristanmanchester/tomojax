"""External direct-reconstruction adapters with explicit physical normalization.

The ASTRA 3D workflow combines an independently constructed discrete Ram-Lak
filter (CuPy/cuFFT) with ASTRA's voxel-driven CUDA backprojection. ASTRA's native
2D FBP and TIGRE's public FBP are separate baselines, not emulated TomoJAX calls.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def astra_geometries(case: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert physical poses to ASTRA vectors without importing another solver."""
    import astra

    from tomojax.core.geometry.base import grid_volume_origin

    grid, det, poses = case.grid, case.detector, case.poses
    origin = np.asarray(grid_volume_origin(grid))
    voxel = np.asarray([grid.vx, grid.vy, grid.vz])
    lower = origin - voxel / 2
    upper = lower + np.asarray([grid.nx, grid.ny, grid.nz]) * voxel
    vol_geom = astra.create_vol_geom(
        grid.ny, grid.nx, grid.nz, lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]
    )
    center_world = np.asarray([det.det_center[0], 0.0, det.det_center[1]])
    center = np.einsum("ni,nij->nj", center_world - poses[:, :3, 3], poses[:, :3, :3])
    vectors = np.concatenate(
        [poses[:, 1, :3], center, poses[:, 0, :3] * det.du, poses[:, 2, :3] * det.dv], axis=1
    )
    proj_geom = astra.create_proj_geom("parallel3d_vec", det.nv, det.nu, vectors)
    return vol_geom, proj_geom


def extend_filter_support(case: Any) -> tuple[Any, int]:
    """Zero-extend measured rows so filtered tails cover all projected corners.

    Independent of TomoJAX's padding helper. Padding is part of each measured
    external workflow; no additional measured rays or phantom truth are used.
    """
    from tomojax.core.geometry.base import grid_volume_origin

    grid, detector = case.grid, case.detector
    origin = np.asarray(grid_volume_origin(grid))
    far = origin + (np.array([grid.nx, grid.ny, grid.nz]) - 1) * [grid.vx, grid.vy, grid.vz]
    corners = np.array(np.meshgrid(*zip(origin, far, strict=True), indexing="ij")).reshape(3, -1).T
    coordinates = np.einsum("ni,vi->vn", corners, case.poses[:, 0, :3]) + case.poses[:, 0, 3, None]
    radius = np.max(np.abs(coordinates - detector.det_center[0])) / detector.du
    padding = max(0, int(np.ceil(radius - (detector.nu - 1) / 2)))
    if padding == 0:
        return case, 0
    return replace(
        case,
        detector=replace(detector, nu=detector.nu + 2 * padding),
        analytic=np.pad(case.analytic, ((0, 0), (0, 0), (padding, padding))),
    ), padding


def ramp_impulse(detector_width: int, detector_spacing: float) -> np.ndarray:
    """Build the physical finite-ramp convolution kernel in a zero-padded domain."""
    if detector_width < 1 or not np.isfinite(detector_spacing) or detector_spacing <= 0:
        raise ValueError("Detector width and finite spacing must be positive")
    length = 1 << (2 * detector_width - 1).bit_length()
    distance = np.minimum(np.arange(length), length - np.arange(length))
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 0.25 / detector_spacing
    odd = distance % 2 == 1
    impulse[odd] = -1 / (np.pi**2 * distance[odd] ** 2 * detector_spacing)
    return impulse


def _astra_backproject_batch(
    volume_geometry: dict[str, Any],
    projection_geometry: dict[str, Any],
    filtered: Any,
    result: Any,
    volume_id: int | None,
) -> None:
    """Own the projector and temporary data handle for one synchronized batch."""
    import astra
    from astra.experimental import accumulate_BP
    import cupy as cp

    projector = astra.create_projector("cuda3d", projection_geometry, volume_geometry)
    projection_id = None
    try:
        # Separate CUDA streams: synchronize both sides of the handoff.
        cp.cuda.runtime.deviceSynchronize()
        if volume_id is None:
            # Keep the cheaper DLPack call when only one batch is needed.
            astra.projector3d.direct_BP(projector, filtered, out=result)
        else:
            projection_id = astra.data3d.link("-sino", projection_geometry, filtered)
            accumulate_BP(projector, volume_id, projection_id)
        cp.cuda.runtime.deviceSynchronize()
    finally:
        if projection_id is not None:
            astra.data3d.delete(projection_id)
        astra.projector3d.delete(projector)


def astra_fbp3d(case: Any, *, filter_batch: int | None = None) -> tuple[np.ndarray, dict]:
    """Filter on the GPU and apply ASTRA BP with physical angular/volume weights."""
    import astra
    import cupy as cp

    case, padding = extend_filter_support(case)
    grid, detector = case.grid, case.detector
    impulse = cp.asarray(ramp_impulse(detector.nu, detector.du))
    response = cp.fft.rfft(impulse)
    n_views = len(case.poses)
    if filter_batch is None:
        capacity = max(1, (512 * 1024**2) // (16 * detector.nv * len(impulse)))
        filter_batch = n_views if capacity >= n_views else 1 << (capacity.bit_length() - 1)
    if filter_batch < 1:
        raise ValueError("filter_batch must be positive")
    filter_batch = min(filter_batch, n_views)
    result = cp.zeros((grid.nz, grid.ny, grid.nx), dtype=cp.float32)
    volume_geometry, complete_projection_geometry = astra_geometries(case)
    volume_id = (
        astra.data3d.link("-vol", volume_geometry, result) if filter_batch < n_views else None
    )
    try:
        for start in range(0, n_views, filter_batch):
            stop = min(start + filter_batch, n_views)
            data = cp.asarray(case.analytic[start:stop])
            spectrum = cp.fft.rfft(data, n=len(impulse), axis=-1)
            spectrum *= response
            filtered = cp.fft.irfft(spectrum, n=len(impulse), axis=-1)[..., : detector.nu]
            filtered = cp.ascontiguousarray(filtered.transpose(1, 0, 2))
            if volume_id is None:
                projection_geometry = complete_projection_geometry
            else:
                batch = replace(case, poses=case.poses[start:stop])
                _, projection_geometry = astra_geometries(batch)
            _astra_backproject_batch(
                volume_geometry, projection_geometry, filtered, result, volume_id
            )
        # ASTRA's parallel BP carries voxel_volume / detector_area weighting.
        # Cancel it, then apply the half-turn angular quadrature; no fitted scale.
        scale = (
            (np.pi / len(case.poses)) * detector.du * detector.dv / (grid.vx * grid.vy * grid.vz)
        )
        result *= np.float32(scale)
        host = cp.asnumpy(result).transpose(2, 1, 0).copy()
    finally:
        if volume_id is not None:
            astra.data3d.delete(volume_id)
    return host, {
        "backend": "CuPy_RamLak_ASTRA_accumulate_BP3D_CUDA"
        if filter_batch < n_views
        else "CuPy_RamLak_ASTRA_BP3D_CUDA",
        "filter": "ram-lak",
        "filter_support": "volume",
        "detector_zero_padding_each_side": padding,
        "filter_views_per_batch": filter_batch,
        "angular_scale": "pi/n_views",
        "regulariser": "none",
        "positivity": False,
        "approximate_for_tilted_scans": True,
    }


def astra_fbp2d(case: Any) -> tuple[np.ndarray, dict]:
    """Run native FBP_CUDA slice by slice on centred, isotropic parallel fixtures."""
    import astra

    case, padding = extend_filter_support(case)
    grid, detector = case.grid, case.detector
    volume_geometry = astra.create_vol_geom(
        grid.ny,
        grid.nx,
        -grid.nx * grid.vx / 2,
        grid.nx * grid.vx / 2,
        -grid.ny * grid.vy / 2,
        grid.ny * grid.vy / 2,
    )
    projection_geometry = astra.create_proj_geom(
        "parallel", detector.du, detector.nu, -np.deg2rad(case.angles_deg)
    )
    data_ids = []
    algorithm = None
    try:
        sino = astra.data2d.create("-sino", projection_geometry, 0.0)
        data_ids.append(sino)
        volume = astra.data2d.create("-vol", volume_geometry, 0.0)
        data_ids.append(volume)
        config = astra.astra_dict("FBP_CUDA")
        config.update(ProjectionDataId=sino, ReconstructionDataId=volume)
        config["option"] = {"FilterType": "ram-lak"}
        algorithm = astra.algorithm.create(config)
        result = np.empty((grid.nx, grid.ny, grid.nz), dtype=np.float32)
        for z in range(grid.nz):
            astra.data2d.store(sino, np.ascontiguousarray(case.analytic[:, z, :]))
            astra.algorithm.run(algorithm)
            # ASTRA 2D image rows run downwards in y; 3D array rows run upwards.
            result[:, :, z] = astra.data2d.get(volume).T[:, ::-1]
    finally:
        if algorithm is not None:
            astra.algorithm.delete(algorithm)
        astra.data2d.delete(data_ids)
    return result, {
        "backend": "FBP_CUDA_per_slice",
        "filter": "ram-lak",
        "filter_support": "volume",
        "detector_zero_padding_each_side": padding,
        "regulariser": "none",
        "positivity": False,
    }
