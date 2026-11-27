"""Interrupt actual publication system calls and inspect the visible generation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

import tomojax as tj


def publish(path: Path, generation: int) -> None:
    grid = tj.Grid(3, 4, 5, 1.0, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, tj.Detector(3, 4, 1, 1), np.arange(3.0))
    recon = tj.Reconstruction(
        np.full(grid.shape, generation, np.float32), geometry, "fbp", {"generation": generation}
    )
    tj.export(path, recon)


def interrupted(path: Path, kill: int) -> None:
    from contextlib import ExitStack
    from unittest.mock import patch

    # Count changes to a public name, excluding backups inside the staging tree.
    count = 0

    def intercept(name):
        real = getattr(os, name)

        def call(*args, **kwargs):
            nonlocal count
            result = real(*args, **kwargs)
            dest = Path(args[-1]) if isinstance(args[-1], str | Path) else None
            if dest is not None and dest.parent == path.parent:
                count += 1
                if count == kill:
                    os._exit(77)
            return result

        return call

    with ExitStack() as scope:
        for name in ("link", "rename", "replace", "unlink"):
            scope.enter_context(patch.object(os, name, intercept(name)))
        publish(path, 2)


@pytest.mark.parametrize("name", ["volume.tif", "slices", "volume.mhd"])
@pytest.mark.parametrize("previous", [False, True])
def test_interruption_never_pairs_one_generations_record_with_anothers_pixels(
    tmp_path, name, previous
):
    for k in range(1, 6):
        root = tmp_path / str(k)
        root.mkdir()
        path = root / name
        if previous:
            publish(path, 1)
        result = subprocess.run([sys.executable, __file__, str(path), str(k)], check=False)
        assert result.returncode in (0, 77)
        if (
            path.suffix == ".mhd"
            and path.with_suffix(".raw").exists()
            and not path.with_suffix(".mhd.json").exists()
        ):
            with pytest.raises(ValueError, match="incomplete"):
                tj.inspect(path)
        elif not path.exists() and not previous:
            assert not (root / "volume.mhd.json").exists()
        else:
            record = tj.inspect(path)["record"]
            generation = record["geometry"]["meta"]["reconstruction_info"]["generation"]
            assert generation in (1, 2)
            if path.is_dir():
                planes = sorted(path.glob("*.tif"))
                assert len(planes) == 5
                data = np.stack([tifffile.imread(p) for p in planes])
            elif path.suffix == ".mhd":
                data = np.fromfile(path.with_suffix(".raw"), "<f4")
            else:
                data = tifffile.imread(path)
            assert data.size == 60 and np.all(data == generation)
        if result.returncode == 0:
            break


if __name__ == "__main__":
    interrupted(Path(sys.argv[1]), int(sys.argv[2]))
