"""Use real stdio for both MCP eras, including concurrent subscriptions and tasks."""

import json
from pathlib import Path
import time

import h5py
import pytest

from tests.mcp_client import connection
import tomojax as tj

pytestmark = pytest.mark.surface
ACK = {
    "resultType": "complete",
    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "tomojax", "version": tj.__version__}},
}


def test_worker_tasks_use_production_tools_and_ttl(tmp_path):
    with connection(tmp_path) as c:
        task = c.call("simulate", tasks=True, size=16, n_views=32, rotation=360)
        assert task["resultType"] == "task" and task["ttlMs"] == 86_400_000
        assert task["pollIntervalMs"] == 2_000
        simulation = c.terminal(task["taskId"])["result"]["structuredContent"]
        args = dict(source=simulation["members"]["scan"], method="opposite-views")
        assert c.call("align", tasks=True, dry_run=True, **args)["resultType"] == "complete"
        aligned = c.terminal(c.call("align", tasks=True, **args)["taskId"])
        assert not aligned["result"]["isError"], aligned["result"]["content"][0]
        assert (
            aligned["status"] == "completed"
            and aligned["result"]["structuredContent"]["members"]["scan"]
        )
        bad = c.terminal(c.call("python", tasks=True, code="1 / 0")["taskId"])
        assert bad["status"] == "completed" and bad["result"]["isError"]
        job = c.call(
            "reconstruct", tasks=True, source=args["source"], method="sirt", iterations=10000
        )
        assert c.task("cancel", job["taskId"]) == ACK
        assert c.terminal(job["taskId"])["status"] == "cancelled"


def test_correction_history_matches_python_on_correct_and_open(tmp_path):
    import numpy as np

    from tests._helpers import write_raw_nxtomo

    raw = tmp_path / "raw.nxs"
    write_raw_nxtomo(raw)
    expected = [record.to_dict() for record in tj.load(raw).corrections]
    assert expected[0]["name"] == "flat_dark"
    with connection(tmp_path) as c:
        frames = c.call("open", kind="frames", path="raw.nxs")["structuredContent"]["handle"]
        corrected = c.call("correct", source=frames, detail="full")["structuredContent"]
        assert corrected["info"]["summary"]["corrections"] == expected
        assert corrected["corrections"] == expected
        c.call("save", source=corrected["handle"], path="scan.nxs")
        reopened = c.call("open", kind="scan", path="scan.nxs")["structuredContent"]
        assert reopened["corrections"] == expected

        kept = np.arange(1, 70)
        selected = tj.simulate("sphere", size=8, n_views=70).scan.selected(kept)
        tj.save(tmp_path / "selected.nxs", selected)
        reply = c.call("open", path="selected.nxs")["structuredContent"]
        assert reply["corrections"][-1]["settings"]["views"] is None
        ref = reply["info"]["arrays"]["/corrections/0/settings/views"]
        np.testing.assert_array_equal(np.load(ref["storage"]["path"]), kept)


def test_directory_inspection_and_unrecognized_hdf5(tmp_path):
    with h5py.File(tmp_path / "unknown.h5", "w") as file:
        file["a/b::c"] = [1, 2, 3]
    with connection(tmp_path) as c:
        tree = c.call("inspect", path="unknown.h5")["structuredContent"]
        assert tree["datasets"]["/a/b::c"]["shape"] == [3]
        root = tmp_path / "files"
        root.mkdir()
        tj.save(root / "scan.nxs", tj.simulate("sphere", size=8, n_views=12).scan)
        (root / "text").write_text("unrelated")
        scanned = c.call("inspect", path=str(root))["structuredContent"]
        assert list(scanned["datasets"]) == [str(root / "scan.nxs")]


def test_joint_alignment_reports_disturbances_and_explicit_exclusions(tmp_path):
    import numpy as np

    n, rows, cols = 100, 24, 32
    angles = np.linspace(0, 180, n, endpoint=False)
    geometry = tj.ParallelGeometry(
        tj.Grid(24, 24, 16, 1, 1, 1), tj.Detector(cols, rows, 1, 1), angles
    )
    u = np.arange(cols)
    v = np.arange(rows)[:, None]
    data = np.exp(
        -((v - 12) ** 2 / 20 + (u - 16 - 3 * np.cos(np.deg2rad(angles))[:, None, None]) ** 2 / 35)
    )
    data += np.random.default_rng(19).normal(0, 0.003, data.shape)
    shadow = 0.35 * np.exp(-((u - np.linspace(8, 20, 8)[:, None]) ** 2 / 10))
    data[61:69] += shadow[:, None, :]
    tj.save(tmp_path / "scan.nxs", tj.Scan(data.astype("f4"), geometry))
    with connection(tmp_path) as c:
        scan = c.call("open", kind="scan", path="scan.nxs")["structuredContent"]["handle"]
        config = dict(
            type="AlignConfig",
            exclude_views=[[3, 5]],
            detector_columns="transient",
            image_tv_weight="coverage",
            iterations=1,
            levels=[1],
            refine_iterations=0,
        )
        reply = c.call("align", source=scan, config=config)
        assert not reply["isError"], reply["content"][0]["text"]
        report = reply["structuredContent"]["disturbed_views"]
        assert report["message"] in reply["content"][0]["text"]
        assert report["ranges"] == [[61, 69]]
        # NeXus stores encoder angles in float32.
        np.testing.assert_allclose(
            report["angles"],
            [[angles[61], angles[68]]],
            rtol=0,
            atol=np.spacing(np.float32(180)),
        )
        scores = np.load(report["score"]["storage"]["path"])
        assert scores.shape == (n,) and np.isnan(scores[3:5]).all()
        np.testing.assert_array_equal(np.flatnonzero(scores > 0.5), np.arange(61, 69))


@pytest.mark.parametrize("size", [32, pytest.param(128, marks=pytest.mark.gpu)])
@pytest.mark.slow
def test_reconstruction_chain_matches_direct_api(tmp_path, size):
    import numpy as np

    def data(name, **params):
        reply = c.call(name, **params)
        assert not reply["isError"], reply["content"][0]["text"]
        if name == "align" and not params.get("dry_run"):
            assert [block["type"] for block in reply["content"]] == ["text"]
            assert "pose RMS" in reply["content"][0]["text"]
        if name == "reconstruct" and not params.get("dry_run"):
            assert any(block["type"] == "image" for block in reply["content"])
        return reply["structuredContent"]

    with connection(tmp_path) as c:
        acquisition = dict(phantom="shepp", size=size, n_views=size * 2, rotation=360)
        simulated = data("simulate", **acquisition, dry_run=True)
        assert simulated["views"] == size * 2 and simulated["phantom_shape"] == [size] * 3
        sim = data("simulate", **acquisition)
        scan = sim["members"]["scan"]
        settings = dict(source=scan, fit="center", method="opposite-views")
        assert data("align", **settings, dry_run=True)["pairs"] > 0
        fitted = data("align", **settings)
        recon = dict(source=fitted["members"]["scan"], method="fbp")
        assert data("reconstruct", **recon, dry_run=True)["grid"]["nx"] == size
        volume = data("reconstruct", **recon)
        candidates, table = check_chain_panels_and_evaluation(data, volume, recon, scan, size)
        reply = c.call("save", source=volume["handle"], path="result.nxs")
        assert not reply["isError"], reply["content"][0]["text"]
        assert [x["name"] for x in reply["content"] if x["type"] == "resource_link"] == [
            "result.nxs",
            "result.nxs.manifest.json",
        ]
        if size == 32:
            check_sweep_and_difference(c, data, scan, volume)
            corrected = data("correct", source=scan)
            edited = data(
                "edit_scan", operation=dict(type="select", source=corrected["handle"], views=[1, 3])
            )
            assert edited["shape"][0] == 2
            for source in (volume["handle"], "result.nxs"):
                array = data("project", source=scan, volume=source)
                back = data("backproject", source=scan, projections=array["handle"])
                assert back["shape"] == [size] * 3
                data("control", action="close", target=array["handle"])
            plan = data("export", source=volume["handle"], path="stack", dry_run=True)
            assert plan["shape"][0] == size and plan["destination_exists"] is False
            written = data("export", source=volume["handle"], path="stack")
            assert (
                Path(written["files"]["first"]).is_file()
                and Path(written["files"]["last"]).is_file()
            )
            sidecar = json.loads((tmp_path / "stack/stack.json").read_text())
            assert sidecar["provenance"]["lineage"]["operation"] == "export"
            assert data("export", source=volume["handle"], path="export.raw")["paths"]
            assert (
                data("open", path="result.nxs", kind="reconstruction", dry_run=True)["type"]
                == "validated"
            )

    phantom = tj.simulate(**acquisition)
    check_scores(phantom, candidates, table, size)
    aligned = tj.align(phantom.scan, fit="center", method="opposite-views")
    assert fitted["calibration"]["center"]["value"] == pytest.approx(
        aligned.scan.detector.center[0], abs=1e-5
    )
    actual = tj.load_reconstruction(tmp_path / "result.nxs")
    expected = tj.reconstruct(aligned.scan, "fbp")
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=1e-4, atol=3e-5)
    assert np.linalg.norm(actual.volume - phantom.phantom) < np.linalg.norm(phantom.phantom)


def check_chain_panels_and_evaluation(data, volume, recon, scan, size):
    index = size // 2
    window = dict(axis="y", index=index, roi=[[1, 7], [index, index + 1], [5, 10]])
    view = data("view", source=volume["handle"], **window)["panels"][0]
    preview = data("preview", operation=dict(type="slice", **window, **recon))["panels"][0]
    assert preview["labels"] == view["labels"]
    assert preview["intensity_limits"] == pytest.approx(view["intensity_limits"], abs=1e-5)
    handle = volume["handle"]
    compact = data("status")["handles"][handle]
    assert compact == {
        **{k: volume[k] for k in ("kind", "shape", "unit")},
        "bytes": compact["bytes"],
    }
    assert compact["bytes"] > 0
    assert "manifest" in data("status", detail="full")["handles"][handle]
    candidates = [
        dict(type="reconstruct", method="cgls", iterations=2),
        dict(type="reconstruct", method="cgls", iterations=3, config=dict(interpolation="cubic")),
    ]
    candidates.append(dict(type="align", fit="center", method="opposite-views"))
    table = data(
        "evaluate",
        source=scan,
        candidates=candidates,
        reference=dict(type="reconstruct", method="cgls", iterations=3),
    )
    assert len(table["scores"]) == 3
    assert data("python", code=f"float(handles[{volume['handle']!r}].volume.max())")["value"] > 0
    return candidates, table


def check_scores(phantom, candidates, table, size):
    import numpy as np

    from tomojax.evaluation import Split
    from tomojax.forward import project_joseph
    from tomojax.geometry import stack_view_poses
    from tomojax.recon import CGLSConfig

    split = Split.interleaved(phantom.scan.projections.shape, stride=5, offset=2)
    assert table["split_fingerprint"] == split.fingerprint
    training = phantom.scan.selected(split.train.ravel())
    for row, setting in zip(table["scores"], candidates, strict=True):
        assert row["count"] == int(split.test.sum()) * size**2 and np.isfinite(row["mse"])
        if size > 32:
            continue
        fit_scan = training
        detector = phantom.scan.detector
        if setting["type"] == "align":
            alignment = tj.align(training, method="opposite-views", fit="center")
            fit_scan = alignment.scan
            detector = fit_scan.detector
            assert alignment.image is None
            assert row["diagnostic"]["calibration"]["center"]["value"] == pytest.approx(
                detector.center[0], abs=1e-5
            )
        cfg = CGLSConfig(
            iterations=setting.get("iterations", 3),
            interpolation=setting.get("config", {}).get("interpolation", "linear"),
        )
        fitted_volume = tj.reconstruct(fit_scan, "cgls", config=cfg)
        predicted = project_joseph(
            fitted_volume.volume,
            stack_view_poses(phantom.scan.geometry, size * 2),
            phantom.scan.grid,
            detector,
            interpolation=cfg.interpolation,
        )
        expected_score = split.score(predicted, phantom.scan.projections)
        assert row["count"] == expected_score.count
        assert row["relative_l2"] == pytest.approx(expected_score.relative_l2, rel=1e-6)
        assert row["mse"] == pytest.approx(expected_score.mse, rel=2e-6)
        assert np.linalg.norm(fitted_volume.volume - phantom.phantom) < np.linalg.norm(
            phantom.phantom
        )


def test_read_directory_result_parts(tmp_path):
    with h5py.File(tmp_path / "custom.h5", "w") as f:
        for n in range(600):
            f[f"diagnostics/{n}"] = n
    with connection(tmp_path) as c:
        clipped = c.call("inspect", path="custom.h5")
        uri = clipped["structuredContent"]["continuation"]
        chunks = []
        while uri:
            reply = c.rpc("resources/read", {"uri": uri})
            content = reply["result"]["contents"][0]["text"]
            assert len(content.encode()) < 20_000
            part = json.loads(content)
            chunks.append(part["data"])
            uri = part["next"]
        full = json.loads("".join(chunks))
        assert len(full["structuredContent"]["datasets"]) == 600


def test_client_cancellation_and_task_progress_scope(tmp_path):
    with connection(tmp_path) as c:
        ref = c.call("simulate", size=16)["structuredContent"]["members"]["scan"]
        params = {
            "name": "reconstruct",
            "arguments": {"source": ref, "method": "sirt", "iterations": 10000},
            "_meta": {"progressToken": "reconstructing"},
        }
        rid = c.send("tools/call", params)
        progress = c.receive("method", "notifications/progress")
        assert progress["params"]["progressToken"] == "reconstructing"
        deadline = time.monotonic() + 10
        while not c.call("status")["structuredContent"]["jobs"]:
            assert time.monotonic() < deadline
        c.send("notifications/cancelled", {"requestId": rid}, notify=True)
        while (state := c.call("status")["structuredContent"])["jobs"]:
            assert time.monotonic() < deadline
        assert len(state["handles"]) == 1
        task = c.rpc(
            "tools/call",
            {
                "name": "reconstruct",
                "arguments": params["arguments"],
                "_meta": {"progressToken": "handed-off"},
            },
            tasks=True,
        )["result"]
        assert c.task("get", task["taskId"])["status"] == "working"
        c.task("cancel", task["taskId"])
        assert c.terminal(task["taskId"])["status"] == "cancelled"
        assert all(row.get("params", {}).get("progressToken") != "handed-off" for row in c.inbox)


@pytest.mark.parametrize("contained", [False, True])
@pytest.mark.parametrize("method", ["fbp", "sirt"])
def test_slab_destination_over_stdio(tmp_path, contained, method):
    import numpy as np

    with h5py.File(tmp_path / "raw.h5", "w") as f:
        counts = np.random.default_rng(7).integers(350, 850, size=(12, 5, 8))
        if contained:
            counts[..., :2] = counts[..., -2:] = 1000
        f["exchange/data"] = counts
        f["exchange/theta"] = np.arange(12) * 15
    with connection(tmp_path) as c:
        raw = c.call(
            "open",
            path="raw.h5",
            kind="frames",
            flats=1000,
            darks=0,
            detector=dict(nu=8, nv=5, du=1, dv=1),
            geometry=dict(kind="parallel"),
        )
        ref = raw["structuredContent"]["handle"]
        args = dict(
            source=ref,
            path="slabs.nxs",
            method=method,
            config={
                "backend": "jax",
                "exterior": "zero",
                **({"iterations": 2} if method == "sirt" else {}),
            },
        )
        valid = c.call("reconstruct_slabs", dry_run=True, detail="full", **args)
        assert valid["structuredContent"]["type"] == "validated"
        assert not (tmp_path / "slabs.nxs").exists()
        saved = c.call("reconstruct_slabs", **args)
        assert not saved["isError"], saved
        assert saved["structuredContent"]["paths"] == [
            str(tmp_path / "slabs.nxs"),
            str(tmp_path / "slabs.nxs.manifest.json"),
        ]
        info = tj.load_reconstruction(tmp_path / "slabs.nxs").info
        assert info["config"]["exterior"] == "zero"
        fraction = info["truncation"]["outside_fraction"]
        assert saved["structuredContent"]["truncation"] == info["truncation"]
        assert (fraction == 0) == contained
        text = saved["content"][0]["text"]
        assert ("beyond the detector" in text) == (not contained)
        if not contained:
            assert f"{fraction:.0%} of attenuation beyond the detector" in text
        assert list(c.call("status")["structuredContent"]["handles"]) == [ref]
        scan = c.call("correct", source=ref)["structuredContent"]["handle"]
        projected = c.call("project", source=scan, volume="slabs.nxs")
        assert projected["structuredContent"]["shape"] == [12, 5, 8]
        error = c.call("reconstruct_slabs", **args)
        assert error["isError"]


def check_sweep_and_difference(c, data, scan, volume):
    sweep = c.call(
        "preview",
        operation=dict(
            type="center_sweep",
            axis="z",
            index=8,
            centers=[-1, 0, 1],
            roi=[[8, 9], [1, 7], [5, 10]],
            source=scan,
            method="fbp",
        ),
    )
    assert [b["type"] for b in sweep["content"]] == ["text", "image"]
    assert len(sweep["structuredContent"]["panels"]) == 3
    cropped = data(
        "view", source=volume["handle"], axis="z", index=8, roi=[[8, 9], [1, 7], [5, 10]]
    )
    for panel in sweep["structuredContent"]["panels"]:
        assert panel["labels"][-2:] == cropped["panels"][0]["labels"][-2:]
    same = data("view", source=volume["handle"], other="result.nxs", mode="difference")
    assert same["intensity_limits"] == [0, 0]
    smaller = data("simulate", size=16, n_views=16)
    mismatch = c.call("view", source=volume["handle"], mode="difference", other=smaller["handle"])
    assert mismatch["isError"] and all(
        ref in mismatch["content"][0]["text"] for ref in (volume["handle"], smaller["handle"])
    )


def test_correction_comparison_and_frame_profile(tmp_path):
    import h5py
    import numpy as np

    counts = np.arange(6 * 8 * 12, dtype=np.float32).reshape(6, 8, 12) / 600 + 0.1
    with h5py.File(tmp_path / "raw.h5", "w") as file:
        file["exchange/data"] = counts
        file["exchange/theta"] = np.arange(6) * 10.0
    with connection(tmp_path) as c:
        opened = c.call(
            "open",
            path="raw.h5",
            kind="frames",
            flats=2,
            darks=0,
            pixel_size=1,
            geometry={"kind": "parallel"},
        )
        frames = opened["structuredContent"]["handle"]
        reply = c.call(
            "preview",
            operation=dict(type="correction_preview", source=frames, rows=[0, 4], binning=2),
        )
        assert [b["type"] for b in reply["content"]] == ["text", "image"]
        data = reply["structuredContent"]
        before, after = data["panels"]
        assert "native" in before["array"] and "corrected" in after["array"]
        assert before["labels"][1:] != after["labels"][1:]
        np.testing.assert_allclose(before["intensity_limits"], np.percentile(counts[3], [1, 99]))
        assert before["intensity_limits"] != after["intensity_limits"]
        assert "intensity_limits" not in data
        profile = c.call(
            "view", source=frames, mode="profile", roi=[[3, 4], [0, 1], [2, 12]], stride=[1, 1, 2]
        )["structuredContent"]
        np.testing.assert_allclose(profile["profile"], counts[3, 0, 2::2])
        assert len(profile["coordinates"]) == 5


def test_project_and_evaluate_saved_npz(tmp_path):
    def call(name, **params):
        reply = c.call(name, **params)
        assert not reply["isError"], reply
        return reply["structuredContent"]

    with connection(tmp_path) as c:
        sim = call("simulate", size=8, n_views=12)
        scan = sim["members"]["scan"]
        call("save", source=sim["handle"], path="sim.npz")
        actual = call("project", source=scan, volume="sim.npz")["handle"]
        expected = call("project", source=scan, volume=sim["members"]["phantom"])["handle"]
        assert call("python", code=f"np.array_equal(handles[{actual!r}], handles[{expected!r}])")[
            "value"
        ]
        for source in (sim["members"]["phantom"], "sim.npz"):
            stats = call("view", source=source, mode="statistics")
            assert stats["count"] == 8**3
        candidate = {"type": "reconstruct", "method": "cgls", "iterations": 2}
        settings = {"reference": candidate, "candidates": [candidate]}
        expected_score = call("evaluate", source=scan, **settings)["scores"][0]["mse"]
        assert call("evaluate", source="sim.npz", **settings)["scores"][0]["mse"] == pytest.approx(
            expected_score
        )


def test_backproject_reads_explicit_datasets(tmp_path):
    import numpy as np

    path = tmp_path / "simulation.nxs"
    sim = tj.simulate(size=8, n_views=12)
    tj.save(path, sim)
    with h5py.File(path, "r+") as file:
        record = json.loads(file["/entry/record"].asstr()[()])
        projections = record["arrays"]["projections"]
        file.move(projections["path"], "/relocated")
        projections["path"] = "/relocated"
        file["/entry/record"][()] = json.dumps(record)
        file["other"] = np.zeros_like(sim.scan.projections)
    plain = tmp_path / "arrays.h5"
    with h5py.File(plain, "w") as file:
        file["/entry/projections"] = sim.scan.projections
    with connection(tmp_path) as c:
        scan = c.call("open", kind="scan", path=str(path))["structuredContent"]["handle"]
        for src, expected in (
            (f"{scan}.projections", True),
            (f"{plain}::/entry/projections", True),
            (f"{path}::/other", False),
        ):
            reply = c.call("backproject", source=scan, projections=src)
            assert not reply["isError"], reply
            array = reply["structuredContent"]["handle"]
            value = c.call("python", code=f"bool(np.any(handles[{array!r}]))")
            assert value["structuredContent"]["value"] is expected


def test_projection_preview_keeps_requested_stride(tmp_path):
    import numpy as np

    with h5py.File(tmp_path / "stripes.h5", "w") as file:
        file["data"] = np.tile(np.arange(1536) % 2, (1, 3, 1))
    with connection(tmp_path) as c:
        selected = dict(source="stripes.h5::/data", stride=[1, 1, 2])
        stats = c.call("view", mode="statistics", **selected)["structuredContent"]
        assert stats["mean"] == 0
        (panel,) = c.call("view", mode="projection", **selected)["structuredContent"]["panels"]
        assert panel["intensity_limits"] == [0, 0]
        assert "stride (1, 1, 2), display subsampling (1, 1, 2)" in panel["labels"]


def test_view_acquisition_requires_opened_member(tmp_path):
    import numpy as np

    from tests.test_io_workflows import _labeled_dataset
    from tomojax.io import save_dataset

    path = tmp_path / "raw.nxs"
    save_dataset(path, _labeled_dataset([5, 11, 9, 1], [0, 1, 0, 2]))
    with connection(tmp_path) as c:
        scan = c.call("open", kind="scan", path=path.name)["structuredContent"]["handle"]
        for name, args in (
            ("view", {"source": path.name}),
            ("project", {"source": scan, "volume": path.name, "dry_run": True}),
            ("backproject", {"source": scan, "projections": path.name, "dry_run": True}),
        ):
            reply = c.call(name, **args)
            error = reply["structuredContent"]
            assert reply["isError"] and error["kind"] == "invalid_parameter"
            for hint in (
                "open",
                ".projections",
                "kind=frames",
                ".counts",
                "correction_preview",
                "inspect",
                "preview=true",
            ):
                assert hint in error["message"]
        frames = c.call("open", kind="frames", path=path.name)["structuredContent"]["handle"]
        for ref, expected in (
            (f"{scan}.projections", -np.log([0.4, 0.8])),
            (f"{frames}.counts", [5, 9]),
        ):
            profile = c.call("view", source=ref, mode="profile", roi=[[0, 2], [0, 1], [1, 2]])
            np.testing.assert_allclose(profile["structuredContent"]["profile"], expected, rtol=1e-6)


def _preview(reply):
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(base64.b64decode(reply["content"][1]["data"]))) as image:
        return np.asarray(image)


def _shows(image, pixels):
    """The gray image is ``pixels`` on a linear scale (clipped at its extremes)."""
    import numpy as np

    assert image.shape == pixels.shape
    assert np.corrcoef(image.ravel(), np.asarray(pixels, float).ravel())[0, 1] > 0.99


def test_inspect_saved_npz_with_and_without_preview(tmp_path):
    import numpy as np

    sim = tj.simulate(size=8, n_views=12)
    tj.save(tmp_path / "scan.npz", sim.scan)
    tj.save(
        tmp_path / "recon.npz",
        tj.Reconstruction(sim.phantom, sim.scan.geometry, "example", {}, sim.scan),
    )
    with connection(tmp_path) as c:
        for name in ("scan", "recon"):
            for preview in (False, True):
                reply = c.call("inspect", path=f"{name}.npz", preview=preview)
                assert not reply["isError"], reply
                report = reply["structuredContent"]
                assert report["valid"] and report["angles"]["count"] == 12
                assert report["projection"]["found"] == (name == "scan")
                assert report["volume"]["found"] == (name == "recon")
                assert [x["type"] for x in reply["content"]] == (
                    ["text", "image"] if preview else ["text"]
                )
                if preview:
                    pixels = sim.scan.projections[6] if name == "scan" else sim.phantom[4]
                    _shows(_preview(reply), np.asarray(pixels))


@pytest.mark.parametrize("suffix", [".nxs", ".mat"])
def test_inspect_preview_shows_the_central_sample_plane(tmp_path, suffix):
    import h5py
    import numpy as np

    from tests.test_psi_reader import _write_v73

    path = tmp_path / ("scan" + suffix)
    rows = 1 if suffix == ".nxs" else 3  # a single-row sinogram, and a PSI phase stack
    planes = np.arange(5 * rows * 7, dtype=np.float32).reshape(5, rows, 7) ** 2
    if suffix == ".mat":
        _write_v73(path, planes, np.arange(5), 1e-6)
    else:
        with h5py.File(path, "w") as file:
            file["entry/instrument/detector/data"] = planes[:, 0]
            file["entry/sample/rotation_angle"] = np.arange(5)
    with connection(tmp_path) as c:
        reply = c.call("inspect", path=path.name, preview=True)
        assert not reply["isError"], reply
        _shows(_preview(reply), planes[2])
