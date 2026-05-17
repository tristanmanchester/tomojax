"""Run context helpers for real-laminography developer workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.bench.real_laminography_artifacts import write_real_lamino_stage_products
from tomojax.bench.real_laminography_planning import parse_real_lamino_z_range

if TYPE_CHECKING:
    from tomojax.geometry import Grid


class RealLaminoRunContext:
    """Mutable per-run state shared by real-laminography stage helpers."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.run_root = Path(args.out)
        self.status_path = self.run_root / "status.json"
        self.preview_global_z = int(args.preview_z)
        self.stack_z_range = parse_real_lamino_z_range(args.stack_z_range)
        self.stage_records: list[dict[str, Any]] = []
        self.naive_slice: np.ndarray | None = None
        self.final_volume: np.ndarray | None = None
        self.final_grid: Grid | None = None

    def stage_dir(self, name: str) -> Path:
        """Return the directory for a named real-lamino stage."""
        return self.run_root / name

    def save_stage_products(
        self,
        *,
        stage_dir: Path,
        volume: np.ndarray,
        grid: Grid,
        full_nz: int,
        input_reference: np.ndarray | None,
        suffix: str = "aligned",
    ) -> dict[str, str]:
        """Write standard stage image products using this run's preview policy."""
        return write_real_lamino_stage_products(
            stage_dir=stage_dir,
            volume=volume,
            grid=grid,
            full_nz=full_nz,
            preview_global_z=self.preview_global_z,
            stack_z_range=self.stack_z_range,
            snapshot_max_cols=int(self.args.snapshot_max_cols),
            input_reference=input_reference,
            fallback_reference=self.naive_slice,
            suffix=suffix,
        )


__all__ = ["RealLaminoRunContext"]
