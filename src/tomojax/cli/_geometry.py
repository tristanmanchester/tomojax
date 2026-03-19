from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import jax.numpy as jnp

from ..align.parametrizations import se3_from_5d
from ..core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry


GridOverride = tuple[int, int, int] | list[int] | None


@dataclass
class AugmentedGeometry:
    """Geometry wrapper that applies saved per-view 5-DOF alignment params."""

    base: Any
    align_params: np.ndarray

    def pose_for_view(self, i: int):
        T_nom = np.asarray(self.base.pose_for_view(i), dtype=np.float32)
        T_delta = np.asarray(
            se3_from_5d(jnp.asarray(self.align_params[i], dtype=jnp.float32)),
            dtype=np.float32,
        )
        T = T_nom @ T_delta
        return tuple(map(tuple, T))

    def rays_for_view(self, i: int):
        return self.base.rays_for_view(i)


def _detector_from_meta(meta: dict) -> Detector:
    det_d = meta["detector"]
    return Detector(
        **{k: det_d[k] for k in ("nu", "nv", "du", "dv")},
        det_center=tuple(det_d.get("det_center", (0.0, 0.0))),
    )


def _grid_from_meta(meta: dict, detector: Detector, grid_override: GridOverride) -> Grid:
    grid_d = meta.get("grid")
    if grid_d is None:
        if grid_override is not None:
            nx, ny, nz = map(int, grid_override)
        else:
            nx = int(detector.nu)
            ny = int(detector.nu)
            nz = int(detector.nv)
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(detector.du),
            vy=float(detector.du),
            vz=float(detector.dv),
        )

    if grid_override is not None:
        nx, ny, nz = map(int, grid_override)
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(grid_d["vx"]),
            vy=float(grid_d["vy"]),
            vz=float(grid_d["vz"]),
        )

    kwargs: dict[str, Any] = {
        k: grid_d[k] for k in ("nx", "ny", "nz", "vx", "vy", "vz")
    }
    if grid_d.get("vol_origin") is not None:
        kwargs["vol_origin"] = tuple(grid_d["vol_origin"])
    if grid_d.get("vol_center") is not None:
        kwargs["vol_center"] = tuple(grid_d["vol_center"])
    return Grid(**kwargs)


def _resolve_thetas_deg(meta: dict, *, apply_saved_angle_offset: bool) -> np.ndarray:
    thetas = np.asarray(meta["thetas_deg"], dtype=np.float32)
    if not apply_saved_angle_offset:
        return thetas

    angle_offset = meta.get("angle_offset_deg")
    if angle_offset is None:
        return thetas

    offset = np.asarray(angle_offset, dtype=np.float32)
    if offset.shape != thetas.shape:
        return thetas
    if not np.isfinite(offset).any() or np.allclose(offset, 0.0):
        return thetas

    # TomoJAX's misalign CLI already bakes scheduled angle offsets into
    # `thetas_deg` and stores the raw schedule separately for provenance.
    if meta.get("misalign_spec") is not None:
        return thetas

    return thetas + offset


def _base_geometry(
    *,
    meta: dict,
    grid: Grid,
    detector: Detector,
    thetas_deg: Sequence[float],
):
    gtype = meta.get("geometry_type", "parallel")
    if gtype == "parallel":
        return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas_deg)

    tilt_deg = float(meta.get("tilt_deg", 30.0))
    tilt_about = str(meta.get("tilt_about", "x"))
    return LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas_deg,
        tilt_deg=tilt_deg,
        tilt_about=tilt_about,
    )


def build_geometry_from_meta(
    meta: dict,
    *,
    grid_override: GridOverride = None,
    apply_saved_alignment: bool = False,
) -> tuple[Grid, Detector, Any]:
    """Build geometry from NXtomo metadata with sensible fallbacks.

    When `grid` metadata is missing, the grid is inferred from detector dimensions
    (or from `grid_override` if supplied) using detector pixel spacings as voxel
    spacings. When `apply_saved_alignment` is True, any saved `align_params` are
    composed onto the nominal poses, and `angle_offset_deg` is applied unless it
    is known to have already been baked into `thetas_deg`.
    """

    detector = _detector_from_meta(meta)
    grid = _grid_from_meta(meta, detector, grid_override)
    thetas_deg = _resolve_thetas_deg(
        meta,
        apply_saved_angle_offset=apply_saved_alignment,
    )
    geom = _base_geometry(meta=meta, grid=grid, detector=detector, thetas_deg=thetas_deg)

    if apply_saved_alignment and meta.get("align_params") is not None:
        align_params = np.asarray(meta["align_params"], dtype=np.float32)
        if align_params.ndim == 2 and align_params.shape[0] == len(thetas_deg):
            geom = AugmentedGeometry(base=geom, align_params=align_params[:, :5])

    return grid, detector, geom
