from __future__ import annotations

import os
from typing import cast

from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging

from ._align_checkpoint import make_align_cli_checkpoint_callbacks
from ._align_command import build_parser
from ._align_outputs import write_alignment_outputs
from ._align_plan import (
    build_align_cli_run_plan,
    execute_alignment_plan,
    init_jax_compilation_cache,
)


def main() -> None:
    """Run alignment from the public CLI."""
    p = build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    init_jax_compilation_cache()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"
    plan = build_align_cli_run_plan(p, args, config_metadata)
    checkpoint_callbacks = make_align_cli_checkpoint_callbacks(plan)
    execution = execute_alignment_plan(
        plan,
        single_checkpoint_callback=checkpoint_callbacks.single,
        multires_checkpoint_callback=checkpoint_callbacks.multires,
    )
    write_alignment_outputs(plan, execution)
