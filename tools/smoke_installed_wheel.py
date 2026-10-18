"""Install a wheel or sdist without a cache and run the README workflow outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from _smoke_workflow import verify_workflow_outputs, workflow_steps


def _clean_env(*, extra: str, root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        _ = env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        JAX_PLATFORMS="cuda" if extra == "cuda12" else "cpu",
        TOMOJAX_JAX_CACHE_DIR=str(root / "jax-cache"),
        JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache"),
        CUPY_CACHE_DIR=str(root / "cupy-cache"),
    )
    return env


def _run(args: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(args, cwd=cwd, check=False, capture_output=True, env=env, text=True)
    if completed.returncode != 0:
        payload: dict[str, object] = {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        raise RuntimeError(json.dumps(payload, indent=2))
    if completed.stdout:
        print(completed.stdout, end="", flush=True)


def _built_artifact(dist: Path, kind: str) -> Path:
    artifacts = sorted(dist.glob("tomojax-*.whl" if kind == "wheel" else "tomojax-*.tar.gz"))
    if len(artifacts) != 1:
        rendered = ", ".join(str(path) for path in artifacts) or "<none>"
        raise RuntimeError(f"expected exactly one built {kind} in {dist}, found {rendered}")
    return artifacts[0].resolve()


def main(argv: list[str] | None = None) -> int:
    """Install without cached packages, require the chosen device, and check CLI outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--python", default=sys.executable, help="Interpreter path or version")
    _ = parser.add_argument("--extra", choices=("cpu", "cuda12"), default="cpu")
    _ = parser.add_argument("--artifact", choices=("wheel", "sdist"), default="wheel")
    _ = parser.add_argument(
        "--dist", type=Path, default=Path(__file__).resolve().parents[1] / "dist"
    )
    _ = parser.add_argument(
        "--find-links", type=Path, help="Offline dependency wheelhouse; disables package indexes"
    )
    args = parser.parse_args(argv)
    artifact = _built_artifact(args.dist, args.artifact)
    with tempfile.TemporaryDirectory(prefix="tomojax-wheel-smoke-") as tmp:
        root = Path(tmp)
        venv, work = root / "venv", root / "work"
        work.mkdir()
        env = _clean_env(extra=args.extra, root=root)
        offline = ["--offline"] if args.find_links is not None else []
        _run(["uv", "venv", *offline, "--python", args.python, str(venv)], cwd=root, env=env)
        python, tomojax = venv / "bin/python", venv / "bin/tomojax"
        install = ["uv", "pip", "install", "--no-cache", "--python", str(python)]
        if args.find_links is not None:
            install += ["--offline", "--no-index", "--find-links", str(args.find_links.resolve())]
        _run([*install, f"{artifact}[{args.extra}]"], cwd=root, env=env)
        _run(["uv", "pip", "check", "--python", str(python)], cwd=root, env=env)
        _run(
            [
                str(python),
                "-c",
                "from pathlib import Path; import jax, tomojax; "
                f"assert Path({str(venv)!r}) in Path(tomojax.__file__).resolve().parents; "
                f"assert jax.default_backend() == {'gpu' if args.extra == 'cuda12' else 'cpu'!r}; "
                "print('Installed:', tomojax.__file__); print('Devices:', jax.devices())",
            ],
            cwd=work,
            env=env,
        )
        _run([str(tomojax), "--help"], cwd=work, env=env)
        for step in workflow_steps(work):
            _run([str(tomojax), *step], cwd=work, env=env)
        verify_workflow_outputs(work)
        _run(
            [
                str(python),
                "-c",
                "import numpy as np; import tomojax as tj; "
                "r = tj.load_reconstruction('recon.nxs'); "
                "v = np.asarray(r.volume); "
                "truth = np.asarray(tj.load_reconstruction('synthetic.nxs').volume); "
                "assert v.shape == truth.shape and np.isfinite(v).all(); "
                "error = float(np.linalg.norm(v - truth) / np.linalg.norm(truth)); "
                "assert error < 0.5, error; "
                "print('Reconstruction:', v.shape, r.method, 'relative_l2=', error)",
            ],
            cwd=work,
            env=env,
        )
        print(f"Installed {artifact.name}[{args.extra}] and reconstructed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
