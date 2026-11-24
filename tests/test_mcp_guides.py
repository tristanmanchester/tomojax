"""Keep the worked Python routes reachable in an installed server."""

from importlib import resources
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface


def guide(c, topic):
    row = c.rpc("resources/read", {"uri": "tomojax://guide/" + topic})
    return row["result"]["contents"][0]["text"]


def example(text, heading):
    return text.split("## " + heading + "\n", 1)[1].split("```python\n", 1)[1].split("\n```", 1)[0]


def test_worked_python_routes(tmp_path):
    with connection(tmp_path) as c:
        text = guide(c, "python")
        for topic in ("solvers", "geometry", "calibration"):
            assert guide(c, topic) in text
        simulated = c.call("simulate", size=12, n_views=20)["structuredContent"]
        python(c, f"sim=handles[{simulated['handle']!r}]\nscan=sim.scan")
        direct = python(c, example(text, "Direct solver"))["value"]
        custom = python(c, example(text, "Custom geometry"))["value"]
        projected = c.call("project", source=custom, volume=direct)["structuredContent"]["handle"]
        assert python(c, f"np.allclose(handles[{projected!r}],tj.project(geometry,volume))")[
            "value"
        ]
        assert not python(
            c, f"np.allclose(handles[{projected!r}],tj.project(scan.geometry,volume))"
        )["value"]
        assert python(c, example(text, "Memory-mapped reconstruction"))["value"] > 0
        assert np.load(tmp_path / "reconstruction.npy").shape == (12, 12, 12)
        assert not c.call("save", source=simulated["members"]["scan"], path="scan.nxs")["isError"]
        metadata = python(c, example(text, "Acquisition metadata"))["value"]
        assert metadata["views"] == 20 and metadata["shape"] == [12, 12, 12]


@pytest.mark.gpu
def test_torch_guide(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("tomojax.torch requires CUDA")
    with connection(tmp_path) as c:
        python(c, "scan=tj.simulate(size=8,n_views=11).scan")
        assert python(c, example(guide(c, "python"), "Torch"))["value"]


def test_public_surface_has_a_tool_or_worked_example():
    # Spell out the entry points exercised by typed workflows. New public names
    # must acquire a route even when their module already has a guide.
    typed = {
        "align": "Alignment align alignment.AlignConfig",
        "simulate": "Simulation simulate datasets.SimulationArtefacts datasets.SimConfig datasets.SimulatedData",
        "describe": "InvalidParameter",
        "open": "Scan Frames load load_frames load_alignment load_reconstruction load_simulation ConeBeam ConeGeometry Grid Detector ParallelGeometry LaminographyGeometry geometry.RotationAxisGeometry geometry.Geometry geometry.ScanGeometry",
        "reconstruct": "Reconstruction reconstruct recon.FBPConfig recon.CGLSConfig recon.MLEMConfig recon.SIRTConfig recon.FISTAConfig recon.SPDHGConfig recon.FourierConfig",
        "correct": "corrections.Step corrections.Corrected corrections.Correction corrections.BeamHardening corrections.Paganin corrections.RadialDistortion corrections.RejectViews corrections.Rings corrections.Stripes corrections.Zingers corrections.correct_frames corrections.correct_projections",
        "preview": "RegionEstimate estimate_region interactive.region_preview interactive.RegionOverlay interactive.correction_preview interactive.CorrectionBand interactive.centre_sweep interactive.reconstruct_slice",
        "inspect": "inspect io.Inspection io.ValidationReport io.validate_dataset",
        "control": "progress.Control progress.Cancelled progress.Advanced progress.Finished progress.Note progress.Paused progress.Placed progress.Preview progress.Resumed progress.Started progress.to_json progress.watch",
        "evaluate": "evaluation.Score evaluation.Split",
        "project": "project",
        "backproject": "backproject",
        "save": "save",
        "export": "export",
        "reconstruct_slabs": "reconstruct_slabs",
    }
    typed_members = {
        "edit_scan": {
            "Scan": "cropped selected binned combine with_poses",
            "Frames": "cropped selected",
        },
        "correct": {
            "Scan": "corrected",
            "Frames": "corrected",
            "corrections.Step": "apply record retained select statistic workspace_pixels",
            "corrections.BeamHardening": "apply",
            "corrections.Paganin": "apply workspace_pixels",
            "corrections.RadialDistortion": "apply workspace_pixels",
            "corrections.RejectViews": "select statistic",
            "corrections.Rings": "apply mirror_axis record retained",
            "corrections.Stripes": "apply",
            "corrections.Zingers": "apply",
        },
        "open": {"Scan": "grid poses detector angles", "Frames": "views"},
        "reconstruct": {"Reconstruction": "grid"},
        "align": {"Alignment": "poses"},
        "preview": {"RegionEstimate": "grid"},
        "evaluate": {"evaluation.Split": "angular interleaved fingerprint"},
        "control": {"progress.Control": "cancel cancelled checkpoint pause resume state"},
        "simulate": {"datasets.SimulationArtefacts": "has_enabled to_dict"},
    }
    from tomojax.cli.mcp import tools

    registry = {
        r["name"]
        for r in tools.registry(SimpleNamespace(supervisor=SimpleNamespace(ledger=None), parts={}))
    }
    assert typed.keys() | typed_members.keys() <= registry
    documented = {n for names in typed.values() for n in names.split()}
    documented.update(
        f"{name}.{method}"
        for classes in typed_members.values()
        for name, methods in classes.items()
        for method in methods.split()
    )
    guides = "\n".join(
        f.read_text() for f in resources.files("tomojax.cli.mcp").joinpath("guides").iterdir()
    )
    code = "\n".join(
        a or b for a, b in re.findall(r"```python\n(.*?)\n```|`([^`\n]+)`", guides, re.S)
    )
    ray_methods = "pose_matrix rays_for_view"
    guide_methods = {
        "geometry": {
            "ConeBeam": "detector_frame from_dict magnification source to_dict",
            "ConeGeometry": ray_methods + " geometry_metadata pose_matrices rotation_axis",
            "Detector": "from_dict to_dict",
            "Grid": "from_dict resampled shape to_dict",
            "LaminographyGeometry": ray_methods + " rotation_axis",
            "ParallelGeometry": ray_methods,
            "Scan": "from_astra to_astra",
            "geometry.ConeSegments": ray_methods + " angles detector grid stack_poses",
            "geometry.Geometry": "pose_matrix",
            "geometry.RotationAxisGeometry": ray_methods + " rotation_axis",
            "geometry.ScanGeometry": "angles detector grid rays_for_view",
        },
        "python": {
            "io.ProjectionDataset": "from_nxtomo geometry_inputs image_key to_nxtomo_metadata with_parallel_detector",
            "torch.Projector": "adjoint forward",
        },
        "calibration": {
            "corrections.BeamHardening": "curve from_polynomial from_spectrum inverse",
            "corrections.Correction": "from_dict to_dict",
            "datasets.DetectorModel": "from_flats simulate",
        },
        "solvers": {
            "evaluation.Score": "mse relative_l2",
            "evaluation.Split": "detector score",
            "Projector": "__call__ adjoint",
        },
    }
    routes = {
        (name, method): resources.files("tomojax.cli.mcp")
        .joinpath("guides", topic + ".md")
        .read_text()
        for topic, classes in guide_methods.items()
        for name, methods in classes.items()
        for method in methods.split()
    }
    surface = Path(__file__).with_name("guardrails").joinpath("api_surface.txt").read_text()
    for line in surface.splitlines():
        if definition := re.match(r"(?:def|class) tomojax\.([\w.]+)", line):
            name = definition[1]
            canonical = (
                name.removeprefix("geometry.")
                if name.startswith("geometry.") and "." not in name[9:] and name[9:] in documented
                else name
            )
            if canonical not in documented:
                assert (
                    re.search(r"(?<![\w.])(?:tj\.)?" + re.escape(canonical) + r"\b", code)
                    or f"from tomojax.{canonical.rpartition('.')[0]} import {canonical.rpartition('.')[2]}"
                    in code
                ), name
        elif member := re.match(r"    \.(\w+)", line):
            method = member[1]
            if f"{canonical}.{method}" in documented:
                continue
            guide_text = routes[canonical, method]
            expression = r"operator\(x\)" if method == "__call__" else r"\." + method + r"\b"
            assert re.search(expression, guide_text), f"{name}.{method}"
