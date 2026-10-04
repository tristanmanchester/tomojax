"""Shared first-use workflow for source and installed-wheel validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def workflow_steps(root: Path) -> list[list[str]]:
    """Return CLI steps that produce and inspect a small reconstructed volume."""
    scan, recon, slices = root / "synthetic.nxs", root / "recon.nxs", root / "slices"
    return [
        [
            "simulate",
            "--out",
            str(scan),
            "--nx",
            "16",
            "--ny",
            "16",
            "--nz",
            "16",
            "--nu",
            "16",
            "--nv",
            "16",
            "--n-views",
            "16",
        ],
        ["inspect", str(scan), "--json", str(root / "inspect.json")],
        ["validate", str(scan)],
        ["recon", "--data", str(scan), "--out", str(recon), "--algo", "fbp", "--roi", "off"],
        ["validate", str(recon)],
        ["slices", "--data", str(recon), "--out", str(slices)],
    ]


def verify_workflow_outputs(root: Path) -> None:
    """Require datasets, inspection output, and all three labelled slices."""
    expected = [
        root / "synthetic.nxs",
        root / "recon.nxs",
        root / "inspect.json",
        root / "slices/slice_slices.json",
        root / "slices/slice_x0008.png",
        root / "slices/slice_y0008.png",
        root / "slices/slice_z0008.png",
    ]
    missing = [str(path) for path in expected if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(
            f"smoke workflow did not create expected output(s): {', '.join(missing)}"
        )
