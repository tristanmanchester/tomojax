"""Independent client fixtures, schema drift and the adapter/session authority boundary."""

import json
from pathlib import Path
import runpy

import h5py
from jsonschema import Draft202012Validator
import msgspec
import numpy as np
import pytest
from referencing import Registry, Resource

import tomojax as tj
from tomojax import progress
from tomojax.cli import envelope, execution, operations, products
from tomojax.wire import arrays, progress as wire

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/protocol"
pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def schemas():
    documents = {
        p.stem.split(".")[0]: json.loads(p.read_text()) for p in DOCS.glob("*.schema.json")
    }
    for schema in documents.values():
        Draft202012Validator.check_schema(schema)
    registry = Registry().with_resources(
        (doc["$id"], Resource.from_contents(doc)) for doc in documents.values()
    )
    return {key: Draft202012Validator(doc, registry=registry) for key, doc in documents.items()}


def test_generated_contract_is_current():
    options = runpy.run_path(str(ROOT / "tools/build_wire_options.py"))
    assert options["render"]() == options["TARGET"].read_text()
    schemas = runpy.run_path(str(ROOT / "tools/build_wire_schemas.py"))["documents"]()
    assert schemas == {p.name: json.loads(p.read_text()) for p in DOCS.glob("*.schema.json")}


@pytest.mark.parametrize(
    "scenario",
    ["saved-volume", "alignment-scan", "simulation", "edit-combine", "raw-slabs", "session"],
)
def test_fixed_client_transcripts(schemas, scenario):
    pending = {}
    results = {}
    for line in (DOCS / "transcripts" / f"{scenario}.jsonl").read_text().splitlines():
        row = json.loads(line)
        schemas["worker"].validate(row)
        if "id" in row:
            request = msgspec.json.decode(line, type=envelope.Request)
            pending.setdefault(request.id, request.channel)
            continue
        msgspec.json.decode(line, type=envelope.Response)
        if row["type"] in ("protocol_error", "ready"):
            continue
        assert pending[row["request"]] == (row["channel"] and row["channel"]["name"])
        if row["type"] != "progress":
            del pending[row["request"]]
            results[row["request"]] = row.get("result")
    assert not pending
    if scenario == "edit-combine":
        left, right = results["m"]["manifest"]["inputs"]
        assert right["parameters"]["path"] == "/data/b.h5"
        for name in ("poses", "bin", "crop", "select"):
            assert left["operation"] == name
            (left,) = left["inputs"]
        assert left["parameters"]["path"] == "/data/a.h5"
    if scenario == "simulation":
        parents = results["b"]["manifest"]["inputs"]
        assert parents == [results["s"]["manifest"], results["p"]["manifest"]]
    if scenario == "raw-slabs":
        assert results["v"]["value"]["volume"]["storage"]["type"] == "npy"
    if scenario == "saved-volume":
        assert results["open"]["value"]["scan"] is None
        assert results["browse"]["value"]["plane"]["fixed"][0]["coordinate"] == 9


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("started", dict(parent=None, name="prepare", label="compile", total=None, unit="")),
        ("advanced", dict(done=3, total=8, metrics={"residual": 0.75}, eta=4.5)),
        ("finished", dict(status="done")),
        ("note", dict(level="warning", message="saturated")),
        ("placed", dict(devices=["cpu:0"])),
        ("paused", {}),
        ("resumed", {}),
        ("preview", dict(arrays={})),
    ],
)
def test_progress_events_are_nonterminal(schemas, name, fields):
    event = dict(schema=2, event=name, task=1, time=0.25, **fields)
    schemas["progress"].validate(event)
    schemas["worker"].validate(dict(type="progress", request="a", channel=None, record=event))
    assert not schemas["progress"].is_valid({**event, "schema": 1})
    assert not schemas["progress"].is_valid({**event, "surprise": True})


def test_terminals_and_invalid_admission_have_distinct_wire_shapes(schemas):
    result = dict(
        schema=2,
        event="result",
        outputs=[],
        method=None,
        arrays={},
        info=dict(summary={}, arrays={}),
    )
    error = dict(
        schema=2,
        event="error",
        kind="invalid_parameter",
        path="corrections[2].alpha",
        message="invalid",
    )
    for terminal in (result, error):
        schemas["progress"].validate(terminal)
        assert not schemas["worker"].is_valid(
            dict(type="progress", request="r", channel=None, record=terminal)
        )
    assert not schemas["progress"].is_valid({**error, "kind": "invalid_request"})
    assert not schemas["worker"].is_valid(
        dict(id="r", command=dict(type="release", path="/input.h5"))
    )
    for command in (
        dict(type="run", operation=dict(type="combine", sources=[{"path": "a.h5"}])),
        dict(
            type="run",
            operation=dict(
                type="bin", source={"type": "member", "handle": "h", "member": "self"}, binning="2"
            ),
        ),
        dict(
            type="run",
            operation=dict(type="project", source={"type": "path", "path": "a"}, volume=[1, 2]),
        ),
    ):
        assert not schemas["worker"].is_valid(dict(id="q", command=command))


class Stage:
    def __init__(self, root):
        self.private = root / "staged"
        self.public = root / "published.npy"

    def array(self):
        return execution.ArrayStage(self.private, arrays.Npy(str(self.public), "issued-token"))


class Control:
    def __init__(self, stop):
        self.stop = stop
        self.checks = 0

    def checkpoint(self):
        self.checks += 1
        if self.checks == self.stop:
            raise progress.Cancelled


def stub(operation, context):
    context.control.checkpoint()
    context.event(wire.Note(2, 1, 0.0, "info", "stub"))
    context.snapshot(products.Snapshot(0.0, "cpu:0", "host", None, None, None))
    stage = context.staging.array()
    plane = np.array([[12, 13, 14, 15], [24, 25, 26, 27]], dtype=np.float32)
    with stage.path.open("wb") as stream:
        np.save(stream, plane)
    context.control.checkpoint()
    metadata = arrays.Array(
        stage.reference,
        (2, 4),
        "<f4",
        (
            arrays.Axis("z", "volume", "mm", arrays.Uniform(7, 3)),
            arrays.Axis("x", "volume", "mm", arrays.Uniform(11, 0.5)),
        ),
        "1/mm",
        "attenuation",
        (arrays.Fixed("y", "volume", "mm", 2, -1),),
    )
    return execution.Success(
        products.PlaneResult(metadata),
        wire.Info({}, {}),
        products.Provenance("plane", {"axis": "y", "index": 2}, context.lineage(operation)),
    )


@pytest.mark.parametrize("adapter", [stub, execution.browse_plane], ids=["stub", "cpu"])
@pytest.mark.parametrize(
    "stop", [None, 1, 2], ids=["success", "cancel-before-work", "cancel-after-stage"]
)
def test_adapter_contract(tmp_path, adapter, stop):
    grid = tj.Grid(4, 3, 2, 0.5, 2, 3, vol_origin=(11, -5, 7))
    ref = arrays.Member("closed-after-admission", "volume")
    descriptor = execution.volume_metadata(grid, storage=ref, dtype="<f4", length_unit="mm")
    data = np.arange(4, 28, dtype=np.float32).reshape(2, 3, 4)
    provenance = products.Provenance("open", {"path": "source.h5"}, ())
    lease = execution.Lease(execution.RegisteredArray(data, descriptor), provenance)
    stage = Stage(tmp_path)
    events, snapshots = [], []
    registry = {ref: lease}
    context = execution.ExecutionContext(
        registry, Control(stop), events.append, snapshots.append, stage
    )
    registry.clear()
    op = operations.Plane(ref, "y", 2)
    if stop:
        with pytest.raises(progress.Cancelled):
            adapter(op, context)
        assert stage.private.exists() == (stop == 2)
    else:
        success = adapter(op, context)
        assert isinstance(success, execution.Success)
        assert success.manifest.inputs == (provenance,)
        plane = success.value.plane
        assert plane.fixed == (arrays.Fixed("y", "volume", "mm", 2, -1),)
        assert plane.axes == (
            arrays.Axis("z", "volume", "mm", arrays.Uniform(7, 3)),
            arrays.Axis("x", "volume", "mm", arrays.Uniform(11, 0.5)),
        )
        assert plane.storage == arrays.Npy(str(stage.public), "issued-token")
        assert plane.shape == (2, 4)
        np.testing.assert_array_equal(np.load(stage.private), [[12, 13, 14, 15], [24, 25, 26, 27]])
    assert not stage.public.exists()
    assert all(isinstance(e, wire.Note) for e in events)
    assert all(isinstance(s, products.Snapshot) for s in snapshots)


def test_summary_externalises_before_json_conversion():
    numbers = np.arange(1000000)
    seen = []

    def stage(pointer, value):
        seen.append((pointer, value))
        return arrays.Array(
            arrays.Npy("history.npy", "a"),
            (1000000,),
            "<i8",
            (arrays.Axis("iteration", "solver", "index", arrays.Uniform(0, 1)),),
            "1",
            "history",
        )

    info = execution.summarise(
        {"automatic": {"method": "fbp"}, "a/b": [{"~": numbers}]}, externalise=stage
    )
    assert info.summary == {"automatic": {"method": "fbp"}, "a/b": [{"~": None}]}
    assert list(info.arrays) == ["/a~1b/0/~0"]
    assert seen[0][1] is numbers
    with pytest.raises(ValueError, match="65536"):
        wire.Info({"message": "a" * 65536}, {})


def test_browse_reopened_volume_without_scan(tmp_path):
    path = tmp_path / "volume.h5"
    grid = tj.Grid(4, 3, 2, 0.5, 2, 3, vol_center=(3, 10, 30))
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 0.5, 3), [0.0, 90.0])
    tj.save(
        path,
        tj.Reconstruction(np.arange(24, dtype=np.float32).reshape(2, 3, 4), geometry, "fbp", {}),
    )
    with pytest.warns(UserWarning, match="without its scan"):
        result = tj.load_reconstruction(path)
    assert result.scan is None
    terminal = json.loads((DOCS / "transcripts/saved-volume.jsonl").read_text().splitlines()[1])
    dataset = terminal["result"]["value"]["volume"]["storage"]["dataset"]
    with h5py.File(path) as saved:
        np.testing.assert_array_equal(saved[dataset][:], result.volume)
    ref = arrays.Member("r", "volume")
    metadata = execution.volume_metadata(result.grid, storage=ref, dtype="<f4", length_unit="mm")
    source = execution.RegisteredArray(result.volume, metadata)
    provenance = products.Provenance("open", {"path": str(path)}, ())
    staging = Stage(tmp_path)
    context = execution.ExecutionContext(
        {ref: execution.Lease(source, provenance)}, Control(None), pytest.fail, pytest.fail, staging
    )
    sliced = execution.browse_plane(operations.Plane(ref, "x", 1), context)
    assert sliced.value.plane.fixed[0].coordinate == 2.75
    np.testing.assert_array_equal(np.load(staging.private), [[1, 5, 9], [13, 17, 21]])


def test_pose_registration_requires_column_order_and_units(schemas):
    raw = json.loads((DOCS / "transcripts/alignment-scan.jsonl").read_text().splitlines()[1])
    encoded = raw["result"]["value"]["poses"]
    pose = msgspec.convert(encoded, type=arrays.Array)
    assert [c.unit for c in pose.columns] == ["degree"] * 3 + ["mm"] * 3
    for broken in (
        [],
        list(reversed(encoded["columns"])),
        [dict(name=c.name, unit="mm") for c in pose.columns],
    ):
        assert not schemas["arrays"].is_valid({**encoded, "columns": broken})
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert({**encoded, "columns": broken}, type=arrays.Array)


def test_admission_finds_all_nested_inputs_before_queueing():
    scan = dict(type="member", handle="s", member="scan")
    mask = dict(type="npy", path="/session/mask.npy", artifact="mask")
    request = msgspec.convert(
        dict(
            id="r",
            command=dict(
                type="plan",
                operation=dict(
                    type="reconstruct", source=scan, config=dict(type="FISTAConfig", support=mask)
                ),
            ),
        ),
        type=envelope.Request,
    )
    assert operations.input_references(request.command) == (
        arrays.Member("s", "scan"),
        arrays.Npy("/session/mask.npy", "mask"),
    )
    request = msgspec.convert(
        dict(
            id="r",
            command=dict(
                type="run",
                operation=dict(
                    type="combine", sources=[scan, dict(type="path", path="a.nxs"), scan]
                ),
            ),
        ),
        type=envelope.Request,
    )
    assert operations.input_references(request.command) == (arrays.Member("s", "scan"),) * 2


def test_adapter_lineage_uses_admitted_handles_and_artifacts_in_command_order(tmp_path):
    scan = arrays.Member("s", "scan")
    artifact = arrays.Npy(str(tmp_path / "never-opened.npy"), "projection")
    simulation = products.Provenance("simulate", {"seed": 17}, ())
    projection = products.Provenance("project", {}, (simulation, simulation))
    context = execution.ExecutionContext(
        {
            artifact: execution.Lease(artifact, projection),
            scan: execution.Lease(
                tj.Scan(
                    np.ones((1, 1, 1)),
                    tj.ParallelGeometry(tj.Grid(1, 1, 1, 1, 1, 1), tj.Detector(1, 1, 1, 1), [0.0]),
                ),
                simulation,
            ),
        },
        Control(None),
        pytest.fail,
        pytest.fail,
        Stage(tmp_path),
    )
    assert context.lineage(operations.Run(operations.Backproject(scan, artifact))) == (
        simulation,
        projection,
    )
    combine = operations.Combine((scan, operations.PathInput("input.nxs"), scan))
    assert context.lineage(operations.Validate(combine)) == (simulation, simulation)
    unpinned = arrays.Npy(artifact.path, "unknown")
    with pytest.raises(KeyError):
        context.lineage(operations.Run(operations.Backproject(scan, unpinned)))


def test_cli_streams_one_terminal_per_command(tmp_path):
    import subprocess
    import sys

    import h5py

    def invoke(*args, status=0):
        call = subprocess.run(
            [sys.executable, "-m", "tomojax.cli.main", *map(str, args), "--progress=json"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert call.returncode == status, call.stderr
        records = [
            msgspec.json.decode(line, type=wire.ProgressRecord) for line in call.stdout.splitlines()
        ]
        assert sum(isinstance(r, wire.Result | wire.Error) for r in records) == 1
        assert isinstance(records[-1], wire.Error if status else wire.Result)
        return records[-1]

    source = tmp_path / "scan.nxs"
    output = tmp_path / "recon.nxs"
    aligned = tmp_path / "aligned.nxs"
    raw = tmp_path / "export.raw"
    result = invoke("simulate", "--size", 12, "--n-views", 24, "--rotation", 360, "-o", source)
    with h5py.File(source) as f:
        for key in ("projections", "phantom"):
            desc = result.arrays[key]
            assert desc.shape == f[desc.storage.dataset].shape
            assert desc.dtype == f[desc.storage.dataset].dtype.str
    assert invoke("inspect", source).info.summary["valid"]
    assert invoke("correct", source, "-o", tmp_path / "corrected.nxs").arrays[
        "projections"
    ].shape == (24, 12, 12)
    result = invoke("reconstruct", source, "-o", output)
    assert result.info.summary["automatic"]["method"] == result.method
    assert result.arrays["volume"].shape == (12, 12, 12)
    result = invoke("align", source, "--method", "opposite-views", "--fit", "center", "-o", aligned)
    assert result.method == "opposite-views" and result.arrays["poses"].shape == (24, 6)
    result = invoke("export", output, "-o", raw)
    assert result.outputs == (str(raw), str(raw.with_suffix(".json")))
    with h5py.File(output) as f:
        np.testing.assert_array_equal(
            np.fromfile(raw, "<f4").reshape(12, 12, 12),
            f[result.arrays["source_volume"].storage.dataset][:],
        )

    for command in ("correct", "reconstruct", "align", "inspect", "export"):
        target = ["-o", tmp_path / "new.nxs"] if command != "inspect" else []
        assert invoke(command, tmp_path / "missing", *target, status=1).kind == "input"
    assert (
        invoke("simulate", "--size", 0, "-o", tmp_path / "bad.nxs", status=2).kind
        == "invalid_parameter"
    )
    assert invoke("inspect", source, "--json", status=2).kind == "invalid_parameter"


def test_result_summary_keeps_selection_when_solver_history_exceeds_budget():
    record = execution.summarise(
        {"history": [{"detail": "x" * 500} for _ in range(1000)], "automatic": {"method": "cgls"}},
        externalise=lambda *_: pytest.fail("structured records are not numeric arrays"),
    )
    assert len(msgspec.json.encode(record)) <= 65536
    assert record.summary == {"automatic": {"method": "cgls"}, "_omitted_fields": 1}
