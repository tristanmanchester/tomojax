from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _normalize_command(cmd: Sequence[str]) -> list[str]:
    args = [str(part) for part in cmd]
    if not args or not args[0]:
        raise ValueError("command must include an executable")
    executable = args[0]
    if not Path(executable).is_absolute():
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"Unable to resolve executable {executable!r}")
        executable = resolved
    args[0] = str(Path(executable).resolve())
    return args


def run_command(cmd: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Run a resolved command with shell disabled."""
    normalized = _normalize_command(cmd)
    return subprocess.run(normalized, shell=False, **kwargs)  # noqa: PLW1510  # nosec B603


def check_output_command(
    cmd: Sequence[str],
    *,
    stderr: int | None = None,
    text: bool = False,
) -> bytes | str:
    normalized = _normalize_command(cmd)
    return subprocess.check_output(  # nosec B603
        normalized,
        shell=False,
        stderr=stderr,
        text=text,
    )
