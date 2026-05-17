from __future__ import annotations

import os
from typing import cast

from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging

from ._align_checkpoint import _make_align_cli_checkpoint_callbacks
from ._align_command import _build_parser
from ._align_outputs import _write_alignment_outputs
from ._align_plan import (
    _build_align_cli_run_plan,
    _execute_alignment_plan,
    _init_jax_compilation_cache,
)


def main() -> None:
    """Run alignment from the public CLI."""
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    _init_jax_compilation_cache()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"
    plan = _build_align_cli_run_plan(p, args, config_metadata)
    checkpoint_callbacks = _make_align_cli_checkpoint_callbacks(plan)
    execution = _execute_alignment_plan(
        plan,
        single_checkpoint_callback=checkpoint_callbacks.single,
        multires_checkpoint_callback=checkpoint_callbacks.multires,
    )
    _write_alignment_outputs(plan, execution)
