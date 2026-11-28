"""Run the native MCP server for one TomoJAX connection."""

import argparse
import os
from pathlib import Path
import shutil
import sys


def build_parser() -> argparse.ArgumentParser:
    """Describe the MCP launcher and its connection option."""
    parser = argparse.ArgumentParser(prog="tomojax mcp", description=__doc__)
    parser.add_argument(
        "--connection", metavar="NAME", default="local", help="Connection profile (default: local)"
    )
    return parser


def main() -> int:
    """Replace the launcher with the worker, keeping the client's stdio pipes."""
    args = build_parser().parse_args()
    name = "tomojax-worker"
    installed = Path(sys.executable).with_name(name)
    worker = str(installed) if installed.is_file() else shutil.which(name)
    if worker is None:
        raise ValueError("Install the matching tomojax-studio wheel or put tomojax-worker on PATH.")
    os.execve(
        worker,
        [worker, "mcp", "--connection", args.connection],
        {**os.environ, "TOMOJAX_STUDIO_PYTHON": sys.executable},
    )
