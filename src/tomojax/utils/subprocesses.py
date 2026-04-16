from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


def _normalize_command(cmd: Sequence[str]) -> list[str]:
    args = [str(part) for part in cmd]
    if not args or not args[0]:
        raise ValueError("command must include an executable")
    executable = args[0]
    if not os.path.isabs(executable):
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"Unable to resolve executable {executable!r}")
        executable = resolved
    args[0] = str(Path(executable).resolve())
    return args


def run_command(cmd: Sequence[str], **kwargs):
    normalized = _normalize_command(cmd)
    return subprocess.run(normalized, shell=False, **kwargs)  # nosec B603


def check_output_command(cmd: Sequence[str], **kwargs):
    normalized = _normalize_command(cmd)
    return subprocess.check_output(normalized, shell=False, **kwargs)  # nosec B603
