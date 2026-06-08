"""Smoke-test the documented TomoJAX CLI workflow on a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path
import tempfile

from tomojax.cli.main import main as run_tomojax


def _run_step(args: list[str]) -> None:
    exit_code = run_tomojax(args)
    if exit_code != 0:
        raise RuntimeError(f"tomojax {' '.join(args)} exited {exit_code}")


def main() -> int:
    """Run a small end-to-end CLI workflow without leaving artifacts in the repo."""
    with tempfile.TemporaryDirectory(prefix="tomojax-smoke-") as tmp:
        root = Path(tmp)
        scan = root / "synthetic.nxs"
        recon = root / "recon.nxs"
        slices = root / "slices"

        _run_step(
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
            ]
        )
        _run_step(["validate", str(scan)])
        _run_step(["recon", "--data", str(scan), "--out", str(recon), "--roi", "off"])
        _run_step(["validate", str(recon)])
        _run_step(["slices", "--data", str(recon), "--out", str(slices)])

        expected_outputs = [
            scan,
            recon,
            slices / "slice_slices.json",
            slices / "slice_x0008.png",
            slices / "slice_y0008.png",
            slices / "slice_z0008.png",
        ]
        missing = [path for path in expected_outputs if not path.exists()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise RuntimeError(f"smoke workflow did not create expected output(s): {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
