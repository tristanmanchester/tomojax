"""Run reconstruction workflows from the public TomoJAX CLI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tomojax.cli._recon_command import ReconCommand, parse_recon_command
from tomojax.cli._recon_outputs import write_reconstruction_outputs
from tomojax.cli._recon_plan import ReconRuntimePlan, build_recon_runtime_plan
from tomojax.cli._runtime import transfer_guard_context
from tomojax.core import log_jax_env, setup_logging
from tomojax.recon.api import ReconstructionResult, run_reconstruction_algorithm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.cli.config import ConfigValue


def _run_reconstruction(command: ReconCommand, config_metadata: dict[str, ConfigValue]) -> None:
    """Run reconstruction from a typed command plan."""
    setup_logging()
    log_jax_env()
    if command.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    plan = build_recon_runtime_plan(command)
    result = _execute_reconstruction(command, plan)
    write_reconstruction_outputs(
        command,
        config_metadata=config_metadata,
        meta=plan.meta,
        geometry_meta=plan.geometry_meta,
        input_grid=plan.input_grid,
        recon_grid=plan.recon_grid,
        detector=plan.detector,
        detector_center_override=plan.detector_center_override,
        det_grid=plan.detector_grid,
        roi_mode=plan.roi_mode,
        is_parallel=plan.is_parallel,
        resolved_views_per_batch=plan.views_per_batch,
        views_per_batch_mode=plan.views_per_batch_mode,
        gather_dtype=plan.gather_dtype,
        volume_mask=plan.volume_mask,
        algorithm_config=result.algorithm_config,
        volume=result.volume,
    )


def _execute_reconstruction(command: ReconCommand, plan: ReconRuntimePlan) -> ReconstructionResult:
    with transfer_guard_context(command.transfer_guard):
        return run_reconstruction_algorithm(plan.algorithm_request)


def main(argv: Sequence[str] | None = None) -> None:
    """Run reconstruction from the public CLI."""
    command, config_metadata = parse_recon_command(argv)
    _run_reconstruction(command, config_metadata)


if __name__ == "__main__":  # pragma: no cover
    main()
