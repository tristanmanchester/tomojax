from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tomojax.cli.align_auto as align_auto_cli

if TYPE_CHECKING:
    from pathlib import Path


def test_align_auto_smoke_help_documents_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = align_auto_cli.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--out-dir" in captured.out
    assert "verification artifacts" in captured.out
    assert "smoke32" in captured.out


def test_align_auto_smoke_command_writes_core_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "auto-smoke"

    exit_code = align_auto_cli.main(["--out-dir", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "final_volume.npy").exists()
    assert (out_dir / "geometry_final.json").exists()
    assert (out_dir / "verification.json").exists()
    captured = capsys.readouterr()
    assert "verification:" in captured.out
