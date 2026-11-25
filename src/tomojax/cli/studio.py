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
    _ = parser.add_argument(
        "--demo", action="store_true", help="Create a demo project and simulate a scan in it"
    )
    return parser


def main() -> int:
    """Launch Studio and return zero on a clean native exit."""
    parser = build_parser()
    args = parser.parse_args()
    if args.demo and args.input is not None:
        parser.error("--demo creates its own project; omit PROJECT_OR_INPUT")
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
        cmd = [cargo, "build", "--locked", "--release", "--manifest-path", str(manifest)]
        cmd += ["--target-dir", str(native / "target")]
        if subprocess.run(cmd, cwd=native, check=False).returncode:
            return 1
    candidates = (built, Path(sys.executable).parent / executable)
    binary = next((str(p) for p in candidates if p.is_file()), shutil.which(executable))
    if binary is None:
        raise ValueError(
            "Studio is not installed. Install tomojax-studio, or run tomojax studio --build "
            "from a source checkout."
        )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"TOMOJAX_STUDIO_INPUT", "TOMOJAX_STUDIO_DEMO"}
    }
    env["TOMOJAX_STUDIO_PYTHON"] = sys.executable
    if source is not None:
        env["TOMOJAX_STUDIO_INPUT"] = str(source)
    if args.demo:
        env["TOMOJAX_STUDIO_DEMO"] = "1"
    return int(subprocess.run([binary], env=env, check=False).returncode != 0)
