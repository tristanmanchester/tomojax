"""Smoke-test the documented TomoJAX CLI workflow on a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path
import tempfile

from _smoke_workflow import verify_workflow_outputs, workflow_steps
from tomojax.cli.main import main as run_tomojax


def _run_step(args: list[str]) -> None:
    exit_code = run_tomojax(args)
    if exit_code != 0:
        raise RuntimeError(f"tomojax {' '.join(args)} exited {exit_code}")


def main() -> int:
    """Run a small end-to-end CLI workflow without leaving artifacts in the repo."""
    with tempfile.TemporaryDirectory(prefix="tomojax-smoke-") as tmp:
        root = Path(tmp)
        for step in workflow_steps(root):
            _run_step(step)
        verify_workflow_outputs(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
