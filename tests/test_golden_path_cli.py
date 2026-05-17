from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from tomojax.cli import main as main_cli
from tomojax.io import load_dataset


def test_golden_path_tiff_ingest_validate_inspect_recon(tmp_path: Path) -> None:
    stack = tmp_path / "stack"
    stack.mkdir()
    iio.imwrite(stack / "0001.tif", np.ones((2, 4), dtype=np.float32))
    iio.imwrite(stack / "0002.tif", np.full((2, 4), 2.0, dtype=np.float32))
    angles = tmp_path / "angles.csv"
    angles.write_text("angle\n0\n90\n", encoding="utf-8")
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    quicklook = tmp_path / "recon.png"

    commands = [
        [
            "ingest",
            str(stack),
            "--angles",
            str(angles),
            "--out",
            str(scan),
            "--du",
            "1",
            "--dv",
            "1",
            "--grid",
            "4",
            "4",
            "2",
        ],
        ["validate", str(scan)],
        ["inspect", str(scan)],
        [
            "recon",
            str(scan),
            "--out",
            str(recon),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--grid",
            "4",
            "4",
            "2",
            "--views-per-batch",
            "1",
            "--quicklook",
            str(quicklook),
        ],
        ["validate", str(recon)],
    ]

    for command in commands:
        assert main_cli.main(command) == 0

    reconstructed = load_dataset(recon)
    assert reconstructed.volume is not None
    assert reconstructed.volume.shape == (4, 4, 2)
    assert quicklook.stat().st_size > 0
