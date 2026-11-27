"""An additional format in an isolated Python package for real front-end tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def example_format():
    from tomojax.io import _export_formats as formats  # check-public-imports: allow-private

    def write(path, blocks, record):
        with path.open("wb") as stream:
            stream.write(json.dumps(record).encode() + b"\n")
            for _, slab in blocks:
                stream.write(slab.tobytes())

    def read(path):
        with path.open("rb") as stream:
            return json.loads(stream.readline())

    return formats.Format(
        "Test volume", (".example",), "file", ("crop",), "Sample values", write, read
    )


def fixture_environment(directory):
    root = Path(__file__).resolve().parents[1]
    package = directory / "tomojax"
    shutil.copytree(root / "src/tomojax", package, ignore=shutil.ignore_patterns("__pycache__"))
    with (package / "io/_export_formats.py").open("a") as module:
        module.write("\nfrom tests._export_fixture import example_format\n")
        module.write("FORMATS = (*FORMATS, example_format())\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (directory, root)))}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from pathlib import Path; from tools.build_wire_options import description; "
            "Path('tomojax/cli/parameters.json').write_text(json.dumps(description()))",
        ],
        cwd=directory,
        env=env,
        check=True,
    )
    return env


if __name__ == "__main__":
    print(fixture_environment(Path(sys.argv[1]))["PYTHONPATH"])
