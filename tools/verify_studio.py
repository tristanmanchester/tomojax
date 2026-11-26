"""Render and verify the Studio workspaces with `uv run --no-sync python tools/verify_studio.py`."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import jax
import msgspec
import numpy as np
from PIL import Image, ImageStat

import tomojax as tj
from tomojax.geometry import grid_volume_origin
from tomojax.wire.arrays import Array, Axis, Hdf5, Uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "studio"
EXECUTABLE = "tomojax-studio"
PROJECT_MANIFEST = "project.tomojax.json"

# The start screen has no open project. The remaining views restore saved state.
PROJECT_VIEWS = {
    "Prepare": ("scan", "Projection", "correct"),
    "Reconstruct": ("scan", "Sinogram", "reconstruct"),
    "Explore": ("ramp", "Volume", "reconstruct"),
    "Compare": ("ramp", "Axial", "reconstruct"),
    "Jobs": ("ramp", "Axial", "reconstruct"),
}
VIEWS = ("start", *(workspace.lower() for workspace in PROJECT_VIEWS))


def locate_binary(*, binary: Path | None = None, build: bool = False) -> Path | None:
    """Use an explicit executable, a Cargo release build, or an installed Studio."""
    if binary is not None:
        return binary.resolve() if binary.is_file() and os.access(binary, os.X_OK) else None

    target = Path(os.environ.get("CARGO_TARGET_DIR", STUDIO / "target")).resolve()
    built = target / "release" / EXECUTABLE
    if build:
        _ = subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                str(STUDIO / "Cargo.toml"),
                "--target-dir",
                str(target),
                "--bin",
                EXECUTABLE,
            ],
            check=True,
        )
    candidates = (built,) if build else (built, Path(sys.executable).parent / EXECUTABLE)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    installed = None if build else shutil.which(EXECUTABLE)
    return Path(installed).resolve() if installed else None


def write_fixture(root: Path) -> dict[str, Path]:
    """Write one parallel-beam acquisition and saved project state for each workspace."""
    root.mkdir()
    size, views = 32, 64
    with jax.default_device("cpu"):
        scan = tj.simulate("shepp", size=size, n_views=views, seed=0).scan
        tj.save(root / "scan.nxs", scan)
        results = {
            "ramp": tj.reconstruct(scan, "fbp", filter="ramp"),
            "hann": tj.reconstruct(scan, "fbp", filter="hann"),
        }
    detector = scan.detector
    projections = Array(
        storage=Hdf5(path="scan.nxs", dataset="/entry/instrument/detector/data"),
        shape=(views, detector.nv, detector.nu),
        dtype="<f4",
        axes=(
            Axis("view", "acquisition", "degree", Uniform(0.0, 180.0 / views)),
            Axis(
                "v",
                "detector",
                "scan_length",
                Uniform(-(detector.nv - 1) * detector.dv / 2, detector.dv),
            ),
            Axis(
                "u",
                "detector",
                "scan_length",
                Uniform(-(detector.nu - 1) * detector.du / 2, detector.du),
            ),
        ),
        unit="1",
        domain="line_integral",
    )
    scan_input: dict[str, object] = {
        "name": "Simulated Shepp scan",
        "path": "scan.nxs",
        "kind": "scan",
        "arrays": {"projections": projections},
    }
    assets: list[dict[str, object]] = [
        dict(scan_input, id="scan", managed=True, created=0, simulated=True)
    ]
    jobs: list[dict[str, object]] = []
    for name, result in results.items():
        path = f"{name}.nxs"
        tj.save(root / path, result)
        grid = result.grid
        x, y, z = grid_volume_origin(grid)
        volume = Array(
            storage=Hdf5(path=path, dataset="/entry/processing/tomojax/volume"),
            shape=grid.shape,
            dtype="<f4",
            axes=(
                Axis("z", "volume", "scan_length", Uniform(z, grid.vz)),
                Axis("y", "volume", "scan_length", Uniform(y, grid.vy)),
                Axis("x", "volume", "scan_length", Uniform(x, grid.vx)),
            ),
            unit="1/scan_length",
            domain="attenuation",
        )
        assets.append(
            {
                "id": name,
                "name": f"FBP · {name.title()}",
                "path": path,
                "kind": "reconstruction",
                "parent": "scan",
                "managed": True,
                "created": len(assets),
                "simulated": False,
                "method": "fbp",
                "arrays": {"volume": volume},
            }
        )
        jobs.append(
            {
                "id": f"recon-{name}",
                "name": f"Reconstruct · {name.title()}",
                "operation": "reconstruct",
                "parameters": {"method": "fbp", "filter": name},
                "inputs": [dict(scan_input, asset="scan")],
                "submitted": 0,
                "status": "completed",
                "stage": "Complete",
                "result": name,
                "destination": path,
            }
        )

    projects: dict[str, Path] = {}
    for workspace, (active, view, operation) in PROJECT_VIEWS.items():
        label = workspace.lower()
        project = root / label
        project.mkdir()
        for asset in ("scan", *results):
            # Independent manifests and shared immutable file contents.
            _ = shutil.copyfile(root / f"{asset}.nxs", project / f"{asset}.nxs")
        state: dict[str, object] = {
            "active": active,
            "selected": ["ramp", "hann"] if workspace == "Compare" else [active],
            "compare": ["ramp", "hann"] if workspace == "Compare" else [None, None],
            "view": view,
            "workspace": workspace,
            "indices": [size // 2] * 3,
            "projection": views // 8,
            "sinogram": detector.nv // 2,
            "window": None,
            "colormap": "",
            "invert": False,
            "camera": [0, 0],
            "clip": [0, 0, 0],
            "density": 0,
            "lighting": 0,
            "zoom": 0,  # Studio restores its default camera and display settings.
            "pan": [0, 0],
            "operation": operation,
            "parameters": {"method": "fbp"} if operation == "reconstruct" else {},
        }
        document: dict[str, object] = {
            "version": 2,
            "id": "studio-verification",
            "name": "Studio verification",
            "created": 0,
            "modified": 0,
            "assets": assets,
            "jobs": jobs,
            "recipes": [],
            "workspace": state,
        }
        _ = (project / PROJECT_MANIFEST).write_bytes(msgspec.json.encode(document))
        projects[label] = project
    return projects


@dataclass(frozen=True)
class ImageStats:
    """PNG dimensions and mean spatial variance across visible RGB channels."""

    width: int
    height: int
    variance: float


def validate_png(path: Path, expected: tuple[int, int]) -> ImageStats:
    """Check a rendered PNG's format, dimensions and visible variation."""
    if not path.is_file():
        raise ValueError(f"{path.name} was not produced")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path.name} is not a PNG")
        if image.size != expected:
            raise ValueError(
                f"{path.name} is {image.width}x{image.height}, expected {expected[0]}x{expected[1]}"
            )
        rgba = image.convert("RGBA")
        visible = Image.alpha_composite(Image.new("RGBA", image.size, "black"), rgba).convert("RGB")
    variance = sum(cast("list[float]", ImageStat.Stat(visible).var)) / 3
    if variance < 1.0:
        raise ValueError(f"{path.name} is blank or nearly uniform (RGB variance {variance:.3g})")
    return ImageStats(*expected, round(variance, 3))


@dataclass(frozen=True)
class Viewport:
    """Logical viewport dimensions and device scale passed to Rune."""

    width: int
    height: int
    scale: float

    @property
    def size(self) -> str:
        """The logical size in Rune's width-by-height syntax."""
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> tuple[int, int]:
        """Output dimensions using Rune's float32 multiplication and rounding."""
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.float32(self.scale)
            return (
                math.floor(float(np.float32(self.width) * scale) + 0.5),
                math.floor(float(np.float32(self.height) * scale) + 0.5),
            )


def capture(binary: Path, out_dir: Path, viewport: Viewport) -> Path:
    """Stage the six renders and publish the output manifest after validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    with TemporaryDirectory(prefix=".verify-", dir=out_dir) as temporary:
        stage = Path(temporary).resolve()
        runtime = stage / "runtime"
        runtime.mkdir(mode=0o700)
        env = os.environ.copy()
        for key in (
            "TOMOJAX_STUDIO_INPUT",
            "TOMOJAX_STUDIO_DEMO",
            "RUNE_CONTRAST",
            "RUNE_INCREASE_CONTRAST",
            "RUNE_MOTION_GATE",
            "RUNE_HEADLESS_WAIT",
        ):
            _ = env.pop(key, None)
        env["XDG_RUNTIME_DIR"] = str(runtime)
        # The navigator truncates the middle of the absolute project path.
        # Keep the visible suffix independent of the temporary directory name.
        projects = write_fixture(stage / "studio-visual-verification-fixture")
        views: dict[str, dict[str, object]] = {}
        expected = viewport.pixels
        for label in VIEWS:
            path = stage / f"{label}.png"
            env["TOMOJAX_STUDIO_SETTINGS"] = str(stage / f"{label}-settings.json")
            if label in projects:
                env["TOMOJAX_STUDIO_INPUT"] = str(projects[label])
            try:
                # Studio preloads the saved arrays before frame zero. Capture there
                # to minimise wall-clock autosave changes in the status bar.
                _ = subprocess.run(
                    [
                        str(binary),
                        "--snap",
                        str(path),
                        "--size",
                        viewport.size,
                        "--scale",
                        str(viewport.scale),
                        "--theme",
                        "light",
                        "--at",
                        "0",
                    ],
                    env=env,
                    check=True,
                    timeout=120,
                )
                views[label] = {"file": path.name, **asdict(validate_png(path, expected))}
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                raise ValueError(f"Studio capture failed for {label}: {error}") from error

        summary: dict[str, object] = {
            "size": viewport.size,
            "scale": viewport.scale,
            "views": views,
        }
        _ = (stage / manifest.name).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        for label in views:
            filename = f"{label}.png"
            _ = (stage / filename).replace(out_dir / filename)
        _ = (stage / manifest.name).replace(manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Capture the Studio workspaces and report the validated manifest path."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    _ = source.add_argument("--binary", type=Path, help="Path to a compiled tomojax-studio")
    _ = source.add_argument("--build", action="store_true", help="Build the locked release binary")
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / ".artifacts" / "studio-snap",
        help="Output directory (default .artifacts/studio-snap in the checkout)",
    )
    _ = parser.add_argument(
        "--size", default="1440x900", help="Logical WIDTHxHEIGHT (default 1440x900)"
    )
    _ = parser.add_argument("--scale", type=float, default=2.0, help="Device scale (default 2)")
    args = parser.parse_args(argv)
    size = cast("str", args.size)
    scale = cast("float", args.scale)
    try:
        width, height = (int(part) for part in size.split("x"))
        if min(width, height) < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError
        viewport = Viewport(width, height, scale)
        if min(viewport.pixels) < 1:
            raise ValueError
    except (ValueError, OverflowError):
        parser.error(
            "--size must be positive WIDTHxHEIGHT and --scale must produce positive pixels"
        )
    try:
        binary = locate_binary(
            binary=cast("Path | None", args.binary), build=cast("bool", args.build)
        )
        if binary is None:
            raise ValueError("Studio executable not found. Use --binary PATH or --build.")
        manifest = capture(binary, cast("Path", args.out_dir), viewport)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"verify_studio: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(VIEWS)} Studio screenshots. See {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
