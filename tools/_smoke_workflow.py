"""Shared first-use workflow for source and installed-wheel validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def workflow_steps(root: Path) -> list[list[str]]:
    """Return CLI steps that produce and inspect a small reconstructed volume."""
    scan, recon = root / "synthetic.nxs", root / "recon.nxs"
    return [
        ["simulate", "-o", str(scan), "--size", "16", "--views", "16"],
        ["inspect", str(scan)],
        ["recon", str(scan), "-o", str(recon), "--method", "fbp", "--roi", "off"],
        ["inspect", str(recon), "--preview", str(root / "previews")],
    ]


def verify_workflow_outputs(root: Path) -> None:
    """Require both datasets and the projection and three slice previews."""
    expected = [
        root / "synthetic.nxs",
        root / "recon.nxs",
        *(
            root / "previews" / f"{name}.png"
            for name in ("projection", "slice_x", "slice_y", "slice_z")
        ),
    ]
    missing = [str(path) for path in expected if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(
            f"smoke workflow did not create expected output(s): {', '.join(missing)}"
        )
