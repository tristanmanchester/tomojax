from __future__ import annotations

from pathlib import Path
import runpy
import subprocess

import pytest


@pytest.mark.parametrize(("extra", "kind"), [("cpu", "wheel"), ("cuda12", "sdist")])
def test_installed_package_smoke_isolated_and_uncached(tmp_path, monkeypatch, extra, kind):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    module = runpy.run_path(str(tools / "smoke_installed_wheel.py"))
    suffix = "whl" if kind == "wheel" else "tar.gz"
    artifact = tmp_path / f"tomojax-0.3.0.{suffix}"
    artifact.touch()
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "home"))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu" if extra == "cuda12" else "cuda")
    monkeypatch.setitem(module["main"].__globals__, "verify_workflow_outputs", lambda root: None)
    assert (
        module["main"](
            [
                "--dist",
                str(tmp_path),
                "--python",
                "3.12",
                "--extra",
                extra,
                "--artifact",
                kind,
                "--find-links",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "--offline" in calls[0][0]
    install = calls[1][0]
    assert all(flag in install for flag in ("--no-cache", "--offline", "--no-index"))
    assert install[-1] == f"{artifact}[{extra}]"
    for _, kwargs in calls:
        assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]
        assert kwargs["env"]["JAX_PLATFORMS"] == ("cuda" if extra == "cuda12" else "cpu")
        assert kwargs["cwd"] != tools.parent
    commands = [args[1:] for args, _ in calls if str(args[0]).endswith("/tomojax")]
    assert commands[1][3:] == ["--size", "32", "--n-views", "60"]
    assert len(commands[2]) == 4  # README's automatic reconstruction, not forced FBP.


def test_installed_package_smoke_fails_on_install_error(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    module = runpy.run_path(str(tools / "smoke_installed_wheel.py"))
    (tmp_path / "tomojax-0.3.0.whl").touch()

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            int(args[:3] == ["uv", "pip", "install"]),
            stdout="",
            stderr="missing dependency",
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="missing dependency"):
        module["main"](["--dist", str(tmp_path)])
    assert len(calls) == 2
