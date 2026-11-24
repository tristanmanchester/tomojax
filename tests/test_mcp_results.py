"""Check result budgets, request-scoped progress and wire coverage independently."""

import asyncio
import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import get_args

from jsonschema import Draft202012Validator
from mcp_types import ImageContent
import msgspec
import numpy as np
from PIL import Image
import pytest

from tomojax.cli import operations as op
from tomojax.cli.mcp import tasks, tools
from tomojax.cli.mcp.parts import Parts
from tomojax.cli.mcp.supervisor import Activity
from tomojax.cli.mcp.tools import budget, seeing

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("case", ["traceback", "poses", "image"])
def test_budget_and_lossless_reassembly(case):
    data = {"handle": "@align-12-abcd", "kind": "alignment"}
    if case == "traceback":
        data.update(message="invalid \n" + "\\\u2603\n" * 18000, next="Inspect the input")
    if case == "poses":
        data.update(poses=np.random.default_rng(4).normal(size=(1600, 6)).tolist())
    full = tools.result(data, error=case == "traceback")
    if case == "image":
        pixels = np.random.default_rng(3).integers(0, 256, size=(1024, 1024), dtype=np.uint8)
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        full.content.append(
            ImageContent(
                type="image",
                mime_type="image/png",
                data=base64.b64encode(stream.getvalue()).decode(),
            )
        )

    async def page():
        parts = Parts()
        clipped = budget.bounded(full, parts)
        assert len(clipped.model_dump_json().encode()) < 20_000
        assert clipped.is_error == full.is_error
        assert clipped.structured_content["handle"] == data["handle"]
        Draft202012Validator(tools.workflow_output()).validate(clipped.structured_content)
        uri = clipped.structured_content["continuation"]
        chunks = []
        while uri:
            call, n = uri.split("/")[-2:]
            part = await parts.read(call, int(n))
            chunks.append(part["data"])
            uri = part["next"]
        return chunks

    chunks = asyncio.run(page())
    for chunk in chunks:
        assert (
            len(
                json.dumps(
                    {"data": chunk, "next": "tomojax://result/" + "0" * 32 + "/1000"}
                ).encode()
            )
            < 20_000
        )
    assert json.loads("".join(chunks)) == full.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def test_paging_evicts_oldest_reply_by_total_bytes():
    async def run():
        parts = Parts(limit=12)
        oldest, second = (parts.add(s).split("/")[-2] for s in ("first", "second"))
        assert (await parts.read(oldest, 0))["data"] == "first"
        third = parts.add("third!").split("/")[-2]
        assert parts.add("too large for the store") is None
        with pytest.raises(ValueError, match="Repeat the call"):
            await parts.read(oldest, 0)
        assert (await parts.read(second, 0))["data"] == "second"
        assert (await parts.read(third, 0))["data"] == "third!"

    asyncio.run(run())


def test_synchronous_call_pages_expire_without_another_request():
    async def run():
        parts = Parts(ttl=0)
        reply = budget.bounded(tools.result({"profile": "x" * 24000}), parts)
        call = reply.structured_content["continuation"].split("/")[-2]
        elapsed = asyncio.Event()
        asyncio.get_running_loop().call_later(0, elapsed.set)
        await elapsed.wait()
        assert parts.nbytes == 0
        with pytest.raises(ValueError, match="Repeat the call"):
            await parts.read(call, 0)

    asyncio.run(run())


def test_each_wire_field_has_a_tool_owner():
    records = tools.registry(SimpleNamespace(supervisor=SimpleNamespace(ledger=None), parts={}))
    schemas = {row["name"]: row["schema"] for row in records}
    interactive = {
        op.Plane: "view",
        op.CentreSweep: "M3 preview",
        op.Slice: "M3 preview",
        op.CorrectionPreview: "M3 preview",
        op.RegionPreview: "M3 preview",
    }
    for cls in (op.Slice, op.CentreSweep, op.CorrectionPreview, op.RegionPreview):
        assert (
            set(cls.__struct_fields__)
            <= schemas["preview"]["$defs"][cls.__name__]["properties"].keys()
        )
    for cls in (op.Read, op.Statistics):
        assert (
            set(cls.__struct_fields__) - {"metadata_only"} <= schemas["view"]["properties"].keys()
        )
    assert set(interactive) == set(get_args(op.Interactive.__value__))
    assert set(get_args(op.ExecutionInput.__value__)) == {
        op.Open,
        op.Python,
        op.Inspect,
        op.Plan,
        op.Validate,
        op.Run,
        op.Interactive,
        op.Read,
        op.Statistics,
    }
    assert (
        set(op.Python.__struct_fields__) - {"aliases", "ledger"}
        == schemas["python"]["properties"].keys()
    )
    for candidate in (op.ReconstructSettings, op.AlignSettings):
        fields = schemas["evaluate"]["$defs"][candidate.__name__]["properties"]
        assert set(candidate.__struct_fields__) <= fields.keys()
    assert op.Run.__struct_fields__ == ("operation", "previews")  # previews owned by M3 panels
    for wrapper in (op.Plan, op.Validate):
        assert wrapper.__struct_fields__ == ("operation",)  # dry_run
    for cls in (op.Inspect, op.Open, *get_args(op.Workflow.__value__)):
        name = cls.__struct_config__.tag
        schema = schemas.get(name) or schemas["edit_scan"]["$defs"][cls.__name__]
        assert set(cls.__struct_fields__) <= schema["properties"].keys()
        if name != "inspect":
            assert "dry_run" in schemas.get(name, schemas["edit_scan"])["properties"]
        for key in ("config", "corrections"):
            if key in cls.__struct_fields__:
                wire = msgspec.json.schema(cls)["$defs"]
                for nested in wire:
                    if nested.endswith("Config") and cls is not op.Evaluate:
                        assert '"type": "object"' in json.dumps(schema["properties"]["config"])
                        assert nested not in schema.get("$defs", {})  # keys belong to describe
                    elif nested in {step.__name__ for step in get_args(op.Step.__value__)}:
                        assert (
                            wire[nested]["properties"].keys()
                            == schema["$defs"][nested]["properties"].keys()
                        )


def test_progress_monotonicity_stage_labels_and_heartbeat(monkeypatch):
    monkeypatch.setattr(tasks, "HEARTBEAT", 0.04)
    monkeypatch.setattr(tasks, "PROGRESS_INTERVAL", 0.01)

    async def run():
        received = asyncio.Queue()

        async def notify(value, total, message):
            received.put_nowait((value, total, message))

        job = Activity()
        ctx = SimpleNamespace(session=SimpleNamespace(report_progress=notify))
        job.record(dict(event="started", task=1, name="prepare", label="compiling"))
        monitor = asyncio.create_task(tasks.notify_progress(job, ctx))
        try:
            first = await asyncio.wait_for(received.get(), 1)
            assert "prepare compiling" in first[2]
            idle = await asyncio.wait_for(received.get(), 1)
            job.record(dict(event="started", task=2, name="refine", label="poses"))
            job.record(
                dict(event="advanced", task=2, done=3, total=10, metrics={"loss": 0.002}, eta=40)
            )
            step = await asyncio.wait_for(received.get(), 1)
            assert first[0] < idle[0] < step[0]
            assert step[1] == step[0] + 40
            assert "refine poses 3/10" in step[2] and "loss" in step[2]
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    asyncio.run(run())


def test_comparison_panels_use_one_pixel_mapping_and_label_each_plane():
    from tomojax.cli.images import Panel, limits, montage

    panels = [
        Panel("Before", np.array([[0.0, 1.0], [0.0, 1.0]]), ("z[1] = 2 mm",), "count"),
        Panel("After", np.array([[1.0, 2.0], [1.0, 2.0]]), ("z[1] = 2 mm",), "1"),
    ]
    lo, hi = limits(panels)
    assert (lo, hi) == (0.0, 2.0)
    picture = np.asarray(Image.open(BytesIO(montage(panels, [(lo, hi)] * len(panels)))))
    assert max(picture.shape) <= 1024 and picture.dtype == np.uint8
    assert picture[80, 220] == picture[80, 328 + 100] == 127
    assert picture[80, 100] == 0 and picture[80, 328 + 220] == 255
    assert (picture[240:, :328] > 128).any() and (picture[240:, 328:] > 128).any()


def test_roundoff_in_panel_labels():
    from tomojax.cli.images import label_number

    assert label_number(-3.5527e-15) == "0"
    assert label_number(-0.000025) == "-2.5e-05"


@pytest.mark.parametrize(
    ("fit", "modes"),
    [
        (("center", "roll"), ["planes"]),
        ("translations", ["pose_plot", "planes"]),
        (("center", "rotations"), ["pose_plot", "planes"]),
    ],
)
def test_alignment_default_panels(monkeypatch, fit, modes):
    requested = []

    async def view(_, args):
        requested.append(args.mode)
        return tools.result({})

    monkeypatch.setattr(seeing, "view", view)
    alignment = {"members": {"poses": "@align.poses", "image": "@align.image"}}
    asyncio.run(seeing.default_panels(None, alignment, tools.result({}), fit=fit))
    assert requested == modes
