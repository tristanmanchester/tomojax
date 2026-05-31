from __future__ import annotations

# ruff: noqa: D100,TC001,TC002,TC003
import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tomojax.align.api import (
    AlignConfig,
    AlignInfo,
    AlignmentLossConfig,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
)
from tomojax.geometry import Detector, Geometry, Grid
from tomojax.io import ProjectionDataset

from .command import AlignCommand

type AlignCliInfo = AlignInfo | AlignMultiresInfo


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
    info: AlignCliInfo


@dataclass(frozen=True, slots=True)
class AlignCliCheckpointCallbacks:
    """Checkpoint callbacks for single-level and multires alignment runs."""

    single: Callable[..., None]
    multires: Callable[[AlignMultiresResumeState], None]
