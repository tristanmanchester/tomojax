"""Exercise the worktree bootstrap without a GPU or dependency installation."""

from pathlib import Path
import shlex
import subprocess

import pytest

pytestmark = pytest.mark.surface

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "worktree_setup.sh"


@pytest.fixture
def setup_home(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    commands = {
        "uname": 'printf "%s\\n" "$TEST_PLATFORM"',
        "nvidia-smi": 'touch "$HOME/gpu-probed"\nexit "$TEST_NVIDIA_STATUS"',
        "uv": 'touch "$HOME/uv-called"\nprintf "%s\\n" "$TMPDIR" "$PWD" "$*"',
    }
    for name, body in commands.items():
        command = bin_dir / name
        command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        command.chmod(0o755)

    monkeypatch.setenv("PATH", str(bin_dir), prepend=":")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TMPDIR", raising=False)
    return home


@pytest.mark.parametrize(
    ("platform", "nvidia_status", "flags", "extra", "probes_gpu"),
    [
        ("Linux", "0", (), "cuda12", True),
        ("Linux", "1", (), "cpu", True),
        ("Darwin", "0", (), "cpu", False),
        ("Linux", "0", ("--cpu",), "cpu", False),
        ("Linux", "1", ("--cuda",), "cuda12", False),
    ],
)
def test_dry_run(setup_home, monkeypatch, platform, nvidia_status, flags, extra, probes_gpu):
    monkeypatch.setenv("TEST_PLATFORM", platform)
    monkeypatch.setenv("TEST_NVIDIA_STATUS", nvidia_status)
    result = subprocess.run(
        [SCRIPT, "--dry-run", *flags], cwd=setup_home, capture_output=True, text=True, check=True
    )
    actions = [shlex.split(line) for line in result.stdout.splitlines()[1:]]
    tmp_cache = setup_home / ".cache" / "tomojax-tmp"
    assert ["uv", "sync", "--locked", "--extra", extra, "--dev"] in actions
    assert ["export", f"TMPDIR={tmp_cache}"] in actions
    assert ["mkdir", "-p", "--", str(tmp_cache)] in actions
    assert ("tomojax-studio-target" in result.stdout) == (platform == "Linux")
    assert (setup_home / "gpu-probed").exists() == probes_gpu
    assert not (setup_home / ".cache").exists()
    assert not (setup_home / "uv-called").exists()


@pytest.mark.parametrize("override_tmpdir", [False, True])
def test_setup_creates_caches_and_runs_sync(setup_home, monkeypatch, override_tmpdir):
    monkeypatch.setenv("TEST_PLATFORM", "Linux")
    tmp_cache = setup_home / ".cache" / "tomojax-tmp"
    effective_tmpdir = tmp_cache
    if override_tmpdir:
        effective_tmpdir = setup_home / "custom tmp"
        effective_tmpdir.mkdir()
        monkeypatch.setenv("TMPDIR", str(effective_tmpdir))

    result = subprocess.run(
        [SCRIPT, "--cpu"], cwd=setup_home, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == [
        str(effective_tmpdir),
        str(ROOT),
        "sync --locked --extra cpu --dev",
    ]
    assert tmp_cache.is_dir()
    assert (setup_home / ".cache" / "tomojax-studio-target").is_dir()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--invalid",), "Unknown option: --invalid"),
        (("--cpu", "--cuda"), "Choose one of --cpu or --cuda."),
    ],
)
def test_invalid_flags(setup_home, flags, message):
    result = subprocess.run(
        [SCRIPT, *flags], cwd=setup_home, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not list(setup_home.iterdir())
