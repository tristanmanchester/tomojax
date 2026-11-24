"""Launch the native project workspace with this Python environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    """Describe the native workspace launcher."""
    parser = argparse.ArgumentParser(
        prog="tomojax studio",
        description="Open the native tomography workspace, an existing project or an acquisition.",
    )
    _ = parser.add_argument("input", nargs="?", metavar="PROJECT_OR_INPUT")
    _ = parser.add_argument(
        "--build", action="store_true", help="Build the native application first"
    )
    _ = parser.add_argument("--demo", action="store_true", help="Start a sample project workflow")
    return parser


def main() -> int:
    """Launch Studio and return zero on a clean native exit."""
    args = build_parser().parse_args()
    source = None if args.input is None else Path(args.input).expanduser().resolve(strict=True)
    native = Path(__file__).resolve().parents[3] / "studio"
    manifest = native / "Cargo.toml"
    executable = "tomojax-studio"
    built = native / "target" / "release" / executable
    if args.build:
        if not manifest.is_file():
            raise ValueError(
                "Building Studio requires a TomoJAX source checkout with studio/Cargo.toml"
            )
        cargo = shutil.which("cargo")
        if cargo is None:
            raise ValueError(
                "Install the Rust toolchain to build Studio, then run tomojax studio --build"
            )
        result = subprocess.run(
            [
                cargo,
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                str(manifest),
                "--target-dir",
                str(native / "target"),
            ],
            cwd=native,
            check=False,
        )
        if result.returncode:
            return 1
    binary = str(built) if built.is_file() else shutil.which(executable)
    if binary is None:
        raise ValueError(
            "Studio is not built. Run tomojax studio --build from a source checkout, "
            "or install tomojax-studio on PATH."
        )
    environment = os.environ.copy()
    environment["TOMOJAX_STUDIO_PYTHON"] = sys.executable
    for key in ("TOMOJAX_STUDIO_INPUT", "TOMOJAX_STUDIO_DEMO"):
        environment.pop(key, None)
    if source is not None:
        environment["TOMOJAX_STUDIO_INPUT"] = str(source)
    if args.demo:
        environment["TOMOJAX_STUDIO_DEMO"] = "1"
    return int(subprocess.run([binary], env=environment, check=False).returncode != 0)
