"""Working-set estimates shared by dataset and resolved acquisition inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._inspection_types import MemoryEstimatesReport, ProjectionReport, WorkingSetEstimate


def memory_estimates(
    grid_shape: list[int] | None, projection: ProjectionReport
) -> MemoryEstimatesReport:
    """Estimate resident working sets for the resolved reconstruction grid."""
    if not projection.get("found"):
        return {
            "feasible": False,
            "reconstruction_grid_shape": None,
            "input_projection_bytes": None,
            "modes": {},
            "notes": "Projection data not found; memory estimates are unavailable.",
        }

    projection_bytes = int(projection["storage_bytes"] or 0)
    if grid_shape is None:
        return {
            "feasible": False,
            "reconstruction_grid_shape": None,
            "input_projection_bytes": projection_bytes,
            "modes": {},
            "notes": "Could not infer reconstruction grid shape.",
        }

    voxels = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
    volume_bytes = voxels * np.dtype(np.float32).itemsize
    modes: dict[str, WorkingSetEstimate] = {
        "fbp_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 2 * volume_bytes),
        },
        "fista_tv_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 5 * volume_bytes),
        },
        "spdhg_tv_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 5 * volume_bytes),
        },
    }
    return {
        "feasible": True,
        "reconstruction_grid_shape": grid_shape,
        "input_projection_bytes": projection_bytes,
        "modes": modes,
        "notes": "Heuristic fp32 working-set estimates; actual JAX/XLA peak memory may differ.",
    }
