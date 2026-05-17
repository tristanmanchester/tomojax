from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tomojax.align.api import (
    AlignConfig,
    AlignmentLossConfig,
    AlignMultiresResumeState,
    AlignResumeState,
)
from tomojax.geometry import Detector, Geometry, Grid
from tomojax.io import ProjectionDataset

from ._align_command import AlignCommand


@dataclass(frozen=True, slots=True)
class AlignCliRunPlan:
    """Resolved inputs needed to execute an alignment CLI run."""

    command: AlignCommand
    cli_args: argparse.Namespace
    config_metadata: dict[str, Any]
    loss_config: AlignmentLossConfig
    loss_params: dict[str, float]
    levels: list[int] | None
    run_levels: list[int] | None
    meta: ProjectionDataset
    geometry_meta: dict[str, Any]
    grid: Grid
    recon_grid: Grid
    detector: Detector
    geometry: Geometry
    projections: jnp.ndarray
    cfg: AlignConfig
    gather_dtype: str
    geometry_dofs: tuple[str, ...]
    schedule_metadata: dict[str, object] | None
    checkpoint_path: str | None
    checkpoint_every: int | None
    resume_state: AlignResumeState | AlignMultiresResumeState | None
    apply_cyl_mask: bool


@dataclass(frozen=True, slots=True)
class AlignCliExecutionResult:
    """Alignment result returned by the CLI execution helper."""

    x: jnp.ndarray
    params5: jnp.ndarray
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AlignCliCheckpointCallbacks:
    """Checkpoint callbacks for single-level and multires alignment runs."""

    single: Callable[..., None]
    multires: Callable[[AlignMultiresResumeState], None]
