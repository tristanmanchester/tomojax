"""ASTRA cone baselines with bounded residual and subset-normalizer storage.

These are explicit FP32 recurrences over ASTRA's native CUDA projectors. They
are not timings of ASTRA's named algorithms; :func:`native` supplies those.
Volumes use (z, y, x), matching ASTRA's native layout.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np


def volume_geometry(grid: Any) -> dict[str, Any]:
    """Return an ASTRA grid with the same physical voxel centres."""
    from tomojax.geometry import grid_volume_origin

    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    low = np.asarray(grid_volume_origin(grid)) - spacing / 2
    high = low + spacing * np.asarray([grid.nx, grid.ny, grid.nz])
    return {
        "GridColCount": grid.nx,
        "GridRowCount": grid.ny,
        "GridSliceCount": grid.nz,
        "option": {f"WindowMin{a}": float(v) for a, v in zip("XYZ", low, strict=True)}
        | {f"WindowMax{a}": float(v) for a, v in zip("XYZ", high, strict=True)},
    }


def _geometry(vg: dict[str, Any]) -> dict[str, Any]:
    import astra

    geom = astra.create_vol_geom(vg["GridRowCount"], vg["GridColCount"], vg["GridSliceCount"])
    geom["option"].update(vg["option"])
    return geom


def native(
    data: np.ndarray,
    vectors: np.ndarray,
    vg: dict[str, Any],
    *,
    method: str,
    iterations: int = 1,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Run ASTRA FDK, CGLS or nonnegative SIRT, returning a host volume."""
    import astra

    pg = astra.create_proj_geom("cone_vec", data.shape[1], data.shape[2], vectors)
    shape = (vg["GridSliceCount"], vg["GridRowCount"], vg["GridColCount"])
    out = np.zeros(shape, np.float32) if initial is None else np.ascontiguousarray(initial)
    with ExitStack() as cleanup:
        sid = astra.data3d.link("-sino", pg, np.ascontiguousarray(data.transpose(1, 0, 2)))
        cleanup.callback(astra.data3d.delete, sid)
        vid = astra.data3d.link("-vol", _geometry(vg), out)
        cleanup.callback(astra.data3d.delete, vid)
        cfg = astra.astra_dict(
            {"fdk": "FDK_CUDA", "cgls": "CGLS3D_CUDA", "sirt": "SIRT3D_CUDA"}[method]
        )
        cfg.update(ProjectionDataId=sid, ReconstructionDataId=vid)
        if method == "sirt":
            cfg["option"] = {"MinConstraint": 0}
        aid = astra.algorithm.create(cfg)
        cleanup.callback(astra.algorithm.delete, aid)
        astra.algorithm.run(aid, iterations)
    return out


class Operators(ExitStack):
    """Native CUDA FP/BP with explicit groups of views and counted work."""

    def __init__(
        self, data: np.ndarray, vectors: np.ndarray, vg: dict[str, Any], groups: list[Any]
    ) -> None:
        super().__init__()
        import astra
        import cupy as cp

        workspace = cp.cuda.MemoryPool()
        self.callback(workspace.free_all_blocks)
        self.enter_context(cp.cuda.using_allocator(workspace.malloc))
        self.cp, self.astra, self.data = cp, astra, data
        self.shape = (vg["GridSliceCount"], vg["GridRowCount"], vg["GridColCount"])
        self.parts = []
        self.work = {
            "forward_views": 0,
            "backprojected_views": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
        }
        for group in groups:
            geom = astra.create_proj_geom("cone_vec", data.shape[1], data.shape[2], vectors[group])
            projector = astra.create_projector("cuda3d", geom, _geometry(vg))
            self.callback(astra.projector3d.delete, projector)
            self.parts.append(
                (group, projector, (data.shape[1], len(vectors[group]), data.shape[2]))
            )

    def upload(self, array: np.ndarray) -> Any:
        """Copy a host array and count its payload."""
        self.work["host_to_device_bytes"] += array.nbytes
        return self.cp.asarray(array)

    def download(self, array: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        """Copy a device array and count its payload."""
        self.work["device_to_host_bytes"] += array.nbytes
        result = array.get(out=out)
        return result if out is None else out

    def measured(self, part: Any) -> Any:
        """Upload one group in ASTRA's projection layout."""
        return self.upload(np.ascontiguousarray(self.data[part[0]].transpose(1, 0, 2)))

    def forward(self, volume: Any, part: Any) -> Any:
        """Project one view group."""
        output = self.cp.empty(part[2], self.cp.float32)
        self.astra.projector3d.direct_FP(part[1], volume, out=output)
        self.cp.cuda.runtime.deviceSynchronize()
        self.work["forward_views"] += part[2][1]
        return output

    def back(self, images: Any, part: Any, output: Any) -> None:
        """Backproject one view group into the supplied array."""
        self.astra.projector3d.direct_BP(part[1], images, out=output)
        self.cp.cuda.runtime.deviceSynchronize()
        self.work["backprojected_views"] += part[2][1]


def cgls(
    ops: Operators,
    initial: np.ndarray | None,
    *,
    iterations: int,
    resident_residual: bool = False,
) -> np.ndarray:
    """CGLS with host search projections and host or device residuals."""
    cp = ops.cp
    x = (
        cp.zeros(ops.shape, cp.float32)
        if initial is None
        else ops.upload(np.ascontiguousarray(initial))
    )
    scratch, gradient, residual = cp.empty_like(x), cp.zeros_like(x), []
    for part in ops.parts:
        r = ops.measured(part)
        if initial is not None:
            r -= ops.forward(x, part)
        residual.append(r if resident_residual else ops.download(r))
        ops.back(r, part, scratch)
        gradient += scratch
        del r
    direction, gamma = gradient.copy(), cp.sum(gradient * gradient)
    for _ in range(iterations):
        projections, denominator = [], cp.zeros((), cp.float32)
        for part in ops.parts:
            q = ops.forward(direction, part)
            denominator += cp.sum(q * q)
            projections.append(ops.download(q))
            del q
        alpha = gamma / denominator
        x += alpha * direction
        gradient.fill(0)
        for part, stored_r, q in zip(ops.parts, residual, projections, strict=True):
            r = stored_r if resident_residual else ops.upload(stored_r)
            device_q = ops.upload(q)
            r -= alpha * device_q
            del device_q
            ops.back(r, part, scratch)
            gradient += scratch
            if not resident_residual:
                ops.download(r, out=stored_r)
            del r
        gamma_new = cp.sum(gradient * gradient)
        direction *= gamma_new / gamma
        direction += gradient
        gamma = gamma_new
    return ops.download(x)


def ordered_subsets(
    ops: Operators,
    initial: np.ndarray | None,
    *,
    method: str,
    iterations: int,
    norms: str,
) -> np.ndarray:
    """Nonnegative OS-SART or OSEM; resident, host or recomputed column norms."""
    cp = ops.cp
    x = (
        cp.full(ops.shape, int(method == "osem"), cp.float32)
        if initial is None
        else cp.maximum(ops.upload(np.ascontiguousarray(initial)), 0)
    )
    correction, columns, rows = cp.empty_like(x), [], []
    seen = cp.zeros(ops.shape, cp.bool_) if method == "osem" else None

    def column(part: Any) -> Any:
        result = cp.empty_like(x)
        ops.back(cp.ones(part[2], cp.float32), part, result)
        return result

    for part in ops.parts:
        if norms != "recompute":
            norm = column(part)
            columns.append(ops.download(norm) if norms == "host" else norm)
            del norm
        if method == "ossart":
            rows.append(ops.download(ops.forward(cp.ones_like(x), part)))
    for iteration in range(iterations):
        for group, part in enumerate(ops.parts):
            norm = (
                column(part)
                if norms == "recompute"
                else (ops.upload(columns[group]) if norms == "host" else columns[group])
            )
            if seen is not None and iteration == 0:
                seen |= norm > 0
            measured, predicted = ops.measured(part), ops.forward(x, part)
            if method == "ossart":
                measured -= predicted
                row = ops.upload(rows[group])
                measured /= cp.where(row > 0, row, 1)
                measured[row <= 0] = 0
                del row
            else:
                cp.divide(measured, cp.where(predicted > 0, predicted, 1), out=measured)
                measured[predicted <= 0] = 0
            ops.back(measured, part, correction)
            if method == "ossart":
                x += cp.where(norm > 0, correction / cp.where(norm > 0, norm, 1), 0)
                cp.maximum(x, 0, out=x)
            else:
                x *= cp.where(norm > 0, correction / cp.where(norm > 0, norm, 1), 1)
            del measured, predicted, norm
    if seen is not None:
        x *= seen
    return ops.download(x)
