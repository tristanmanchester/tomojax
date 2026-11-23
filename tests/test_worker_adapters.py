"""Drive the production executor through the JSONL client interface."""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

import tomojax as tj

pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def worker():
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc:
        try:
            ready = json.loads(proc.stdout.readline())
            assert {"validate", "plan", "run", "plane", "inspect", "open"} <= set(
                ready["capabilities"]
            )

            def call(command, **parameters):
                nonlocal request
                request += 1
                rid = str(request)
                proc.stdin.write(
                    json.dumps({"id": rid, "command": {"type": command, **parameters}}) + "\n"
                )
                proc.stdin.flush()
                while line := proc.stdout.readline():
                    row = json.loads(line)
                    assert row["request"] == rid
                    if row["type"] != "progress":
                        return row
                pytest.fail(f"unexpected EOF from worker, status {proc.poll()}")

            request = 0
            yield call
            assert call("shutdown")["type"] == "result"
            assert proc.wait(10) == 0
        finally:
            if proc.poll() is None:
                proc.kill()


def output(reply):
    assert reply["type"] == "result", reply
    return reply["result"]


@pytest.fixture(scope="module")
def simulation(worker):
    return output(
        worker("run", operation={"type": "simulate", "phantom": "sphere", "size": 8, "n_views": 12})
    )


def test_matched_operators_and_ordered_handle_lineage(worker, simulation, tmp_path):
    sim = simulation["value"]
    projected = output(
        worker(
            "run",
            operation=dict(type="project", source=sim["scan"], volume=sim["phantom"]["storage"]),
        )
    )
    assert projected["manifest"]["inputs"] == [simulation["manifest"]] * 2
    reference = tj.simulate("sphere", size=8, n_views=12)
    actual = np.load(projected["value"]["projections"]["storage"]["path"])
    np.testing.assert_allclose(
        actual, tj.project(reference.scan.geometry, reference.phantom), atol=1e-6
    )
    file = tmp_path / "input.h5"
    with h5py.File(file, "w") as f:
        f["samples"] = actual
    back = output(
        worker(
            "run",
            operation=dict(
                type="backproject",
                source=sim["scan"],
                projections=dict(type="hdf5", path=str(file), dataset="samples"),
            ),
        )
    )
    np.testing.assert_allclose(
        np.load(back["value"]["volume"]["storage"]["path"]),
        tj.backproject(reference.scan.geometry, actual),
        atol=1e-5,
    )


def test_save_reopen_plan_edit_and_export(worker, simulation, tmp_path):
    scan = simulation["value"]["scan"]
    for operation in [
        dict(type="select", views=[0, 2, 4, 7, 11]),
        dict(type="crop", rows=[1, 7], cols=[0, 6]),
        dict(type="bin", binning=2),
        dict(type="poses", poses=None),
        dict(type="correct"),
    ]:
        operation["source"] = scan
        assert output(worker("validate", operation=operation))["value"]["type"] == "validated"
        result = output(worker("run", operation=operation))
        scan = dict(type="member", handle=result["value"]["handle"], member="self")
    assert result["value"]["projections"]["shape"] == [5, 3, 3]
    recipe = dict(
        type="reconstruct", source=scan, method="fbp", config=dict(type="FBPConfig", backend="jax")
    )
    planned = output(worker("plan", operation=recipe))["value"]
    assert planned["method"] == "fbp" and planned["placement"]["snapshot"]
    recon = output(worker("run", operation=recipe))
    root = dict(type="member", handle=recon["value"]["handle"], member="self")
    file = tmp_path / "volume.nxs"
    saved = output(worker("run", operation=dict(type="save", source=root, path=str(file))))
    assert json.loads(Path(saved["value"]["paths"][1]).read_text()) == saved["manifest"]
    assert output(worker("inspect", path=str(file)))["value"]["report"]["valid"]
    reopened = output(worker("open", kind="reconstruction", path=str(file)))["value"]
    assert reopened["scan"] is None and reopened["volume"]["storage"]["type"] == "hdf5"
    ref = dict(type="member", handle=reopened["handle"], member="volume")
    sliced = output(worker("plane", source=ref, axis="y", index=1))["value"]["plane"]
    with h5py.File(file) as f:
        np.testing.assert_array_equal(
            np.load(sliced["storage"]["path"]), f[reopened["volume"]["storage"]["dataset"]][:, 1, :]
        )
    raw = tmp_path / "result.raw"
    exported = output(worker("run", operation=dict(type="export", source=root, path=str(raw))))[
        "value"
    ]
    assert set(map(Path, exported["paths"])) == {
        raw,
        raw.with_suffix(".json"),
        raw.with_name("result.raw.manifest.json"),
    }
    assert raw.stat().st_size == np.prod(recon["value"]["volume"]["shape"]) * 4


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        (
            dict(type="reconstruct", method="fbp", config=dict(type="FBPConfig", filter="banana")),
            "config.filter",
        ),
        (dict(type="reconstruct", method="fbp", iterations=2), "iterations"),
        (dict(type="reconstruct", method="auto", warm_start=True), "warm_start"),
        (
            dict(
                type="reconstruct",
                method="fbp",
                config=dict(type="FBPConfig", devices=["nonexistent"]),
            ),
            "config.devices",
        ),
        (dict(type="align", fit="translations", method="opposite-views"), "fit"),
        (
            dict(
                type="correct",
                corrections=[dict(type="Stripes"), dict(type="Zingers", threshold=-1)],
            ),
            "corrections[1].threshold",
        ),
        (dict(type="select", views=[2, 1]), "views"),
        (dict(type="crop", rows=[0, 0], cols=[0, 2]), "rows"),
        (dict(type="bin", binning=1000), "binning"),
    ],
)
def test_validate_and_run_return_same_field(worker, simulation, operation, path):
    operation = dict(operation, source=simulation["value"]["scan"])
    for command in ("validate", "run"):
        reply = worker(command, operation=operation)
        assert reply["type"] == "error", reply
        assert (reply["error"]["kind"], reply["error"]["path"]) == ("invalid_parameter", path)


def test_raw_frames_stream_into_reopenable_slab_volume(worker, tmp_path):
    raw = tmp_path / "raw.h5"
    counts = np.random.default_rng(7).integers(430, 980, size=(17, 5, 13), dtype=np.uint16)
    with h5py.File(raw, "w") as f:
        f["exchange/data"] = counts
        f["exchange/theta"] = np.linspace(0, 180, 17, endpoint=False)
    opening = dict(
        kind="frames",
        path=str(raw),
        flats=1000,
        darks=10,
        detector=dict(nu=13, nv=5, du=1, dv=1),
        geometry=dict(kind="parallel"),
    )
    assert (
        output(worker("validate", operation=dict(type="open", **opening)))["value"]["type"]
        == "validated"
    )
    opened = output(worker("open", **opening))["value"]
    assert opened["counts"]["domain"] == "counts"
    root = dict(type="member", handle=opened["handle"], member="self")
    recipe = dict(
        type="reconstruct_slabs",
        source=root,
        method="fbp",
        config=dict(type="FBPConfig", backend="jax"),
    )
    result = output(worker("run", operation=recipe))["value"]["volume"]
    actual = np.load(result["storage"]["path"], mmap_mode="r")
    frames = tj.load_frames(raw, detector=tj.Detector(13, 5, 1, 1), flats=1000, darks=10)
    from tomojax.recon import FBPConfig

    expected = tj.reconstruct(frames.corrected(), "fbp", config=FBPConfig(backend="jax"))
    np.testing.assert_allclose(actual, expected.volume, rtol=2e-5, atol=3e-6)
    for command in ("validate", "run"):
        failed = worker(command, operation={**recipe, "corrections": [dict(type="RejectViews")]})
        assert failed["error"]["path"] == "corrections[0]"


def test_combined_scan_keeps_repeated_parents_and_alignment_exposes_scan(worker):
    recipe = dict(type="simulate", phantom="sphere", size=12, geometry="cone", n_views=24)
    cone = output(worker("run", operation=recipe))
    scan = cone["value"]["scan"]
    result = output(worker("run", operation=dict(type="combine", sources=[scan, scan])))
    assert result["value"]["projections"]["shape"][0] == 48
    assert result["manifest"]["inputs"] == [cone["manifest"], cone["manifest"]]
    # An opposite-view centre fit is defined for axial parallel scans.
    parallel = output(worker("run", operation={**recipe, "geometry": "parallel", "rotation": 360}))
    fit = dict(
        type="align", source=parallel["value"]["scan"], fit="center", method="opposite-views"
    )
    assert output(worker("plan", operation=fit))["value"]["pairs"] > 0
    alignment = output(worker("run", operation=fit))
    poses = alignment["value"]["poses"]
    assert poses["shape"] == [24, 6]
    assert [c["unit"] for c in poses["columns"]][:3] == ["degree"] * 3
    recon = output(
        worker(
            "run",
            operation=dict(type="reconstruct", source=alignment["value"]["scan"], method="fbp"),
        )
    )
    assert recon["manifest"]["inputs"] == [alignment["manifest"]]
