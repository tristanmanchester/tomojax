"""Start the worker built from this checkout with the active test interpreter."""

from functools import cache
import json
from pathlib import Path
import subprocess
import sys


@cache
def binary() -> str:
    studio = Path(__file__).resolve().parents[1] / "studio"
    build = subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "tomojax-worker",
            "--message-format=json",
        ],
        cwd=studio,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return next(
        row["executable"]
        for line in build.stdout.splitlines()
        if (row := json.loads(line)).get("executable") and row["target"]["name"] == "tomojax-worker"
    )


def command(*args: str) -> list[str]:
    return [binary(), "serve", "--python", sys.executable, *args]
