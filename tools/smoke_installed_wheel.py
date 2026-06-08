"""Smoke-test the built wheel from a fresh environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    _ = env.pop("PYTHONPATH", None)
    _ = env.pop("PYTHONHOME", None)
    return env


def _run(args: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        env=_clean_env(),
        text=True,
    )
    if completed.returncode != 0:
        payload: dict[str, object] = {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        raise RuntimeError(json.dumps(payload, indent=2))


def _script_path(venv: Path, name: str) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / scripts_dir / f"{name}{suffix}"


def _built_wheel(dist: Path) -> Path:
    wheels = sorted(dist.glob("tomojax-*.whl"))
    if len(wheels) != 1:
        rendered = ", ".join(str(path) for path in wheels) or "<none>"
        raise RuntimeError(f"expected exactly one built TomoJAX wheel in {dist}, found {rendered}")
    return wheels[0]


def main() -> int:
    """Install the built wheel in a clean venv and exercise the console script."""
    repo = Path(__file__).resolve().parents[1]
    wheel = _built_wheel(repo / "dist")
    with tempfile.TemporaryDirectory(prefix="tomojax-wheel-smoke-") as tmp:
        root = Path(tmp)
        venv = root / "venv"
        work = root / "work"
        work.mkdir()
        _run(["uv", "venv", "--python", sys.executable, str(venv)], cwd=root)
        python = _script_path(venv, "python")
        tomojax = _script_path(venv, "tomojax")
        _run(["uv", "pip", "install", "--python", str(python), str(wheel)], cwd=root)
        _run(["uv", "pip", "check", "--python", str(python)], cwd=root)
        _run(
            [
                str(python),
                "-c",
                (
                    "from pathlib import Path; import tomojax; "
                    f"repo = Path({str(repo)!r}).resolve(); "
                    "module = Path(tomojax.__file__).resolve(); "
                    "assert repo not in module.parents, module"
                ),
            ],
            cwd=work,
        )
        _run([str(tomojax), "--help"], cwd=work)

        scan = work / "wheel_synthetic.nxs"
        _run(
            [
                str(tomojax),
                "simulate",
                "--out",
                str(scan),
                "--nx",
                "8",
                "--ny",
                "8",
                "--nz",
                "8",
                "--nu",
                "8",
                "--nv",
                "8",
                "--n-views",
                "8",
            ],
            cwd=work,
        )
        _run([str(tomojax), "validate", str(scan)], cwd=work)
        if not scan.exists():
            raise RuntimeError(f"installed wheel smoke did not create {scan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
