"""Check the installed client's form data against the numerical API."""

from dataclasses import MISSING, fields
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import get_args, get_type_hints

from jsonschema import Draft202012Validator
import msgspec
import pytest

import tomojax as tj
from tomojax import alignment, corrections, datasets, recon
from tomojax.cli import wire_options
from tomojax.recon.api import MethodConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def description():
    requests = (
        b'{"id":"d","command":{"type":"describe"}}\n{"id":"q","command":{"type":"shutdown"}}\n'
    )
    process = subprocess.run(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        input=requests,
        capture_output=True,
        check=True,
        timeout=20,
    )
    ready, described, shutdown = map(json.loads, process.stdout.splitlines())
    assert "describe" in ready["capabilities"]
    assert described["request"] == "d"
    assert shutdown["request"] == "q"
    schema = described["result"]["parameters"]
    Draft202012Validator.check_schema(schema)
    build = runpy.run_path(str(ROOT / "tools/build_wire_options.py"))
    assert schema == json.loads(msgspec.json.encode(build["description"]()))
    return schema["$defs"]


def test_every_public_workflow_parameter_has_a_wire_field_or_owner(description):
    owners = {
        "Open": [
            tj.load,
            tj.load_frames,
            tj.load_alignment,
            tj.load_simulation,
            tj.load_reconstruction,
        ],
        "Inspect": [tj.inspect],
        "Save": [tj.save],
        "Export": [tj.export],
        "Correct": [tj.Frames.corrected, tj.Scan.corrected],
        "Simulate": [tj.simulate],
        "Reconstruct": [tj.reconstruct],
        "ReconstructSlabs": [tj.reconstruct_slabs],
        "Align": [tj.align],
        "Project": [tj.project],
        "Backproject": [tj.backproject],
        "Select": [tj.Frames.selected, tj.Scan.selected],
        "Crop": [tj.Frames.cropped, tj.Scan.cropped],
        "Bin": [tj.Scan.binned],
        "Poses": [tj.Scan.with_poses],
        "Combine": [tj.Scan.combine],
        "RegionPreview": [tj.estimate_region],
    }
    for tag, funcs in owners.items():
        properties = description[tag]["properties"]
        for func in funcs:
            mapping = description[tag]["x-python"][func.__qualname__]
            names = set(inspect.signature(func).parameters) - {"self", "cls"}
            assert set(mapping) == names
            assert set(get_type_hints(func)) >= names
            for name, target in mapping.items():
                if (func, name) == (tj.reconstruct, "out"):
                    assert target["owner"] == "worker"
                elif name == "checkpoint":
                    assert target["owner"] == "python"
                else:
                    assert properties[target]["description"]


def test_parameter_defaults_reconstruct_the_owning_objects(description):
    owners = [
        tj.Grid,
        tj.Detector,
        datasets.SimulationArtifacts,
        alignment.AlignConfig,
        *get_args(MethodConfig.__value__),
        *(
            cls
            for name in corrections.__all__
            if isinstance(cls := getattr(corrections, name), type)
            and issubclass(cls, corrections.Step)
            and cls is not corrections.Step
        ),
    ]
    required = {
        tj.Grid: dict(nx=9, ny=10, nz=11, vx=0.3, vy=0.4, vz=0.5),
        tj.Detector: dict(nu=4, nv=2, du=0.7, dv=0.3),
        corrections.Paganin: dict(delta_beta=2, distance=0.2, energy_kev=18, pixel_size=1e-5),
        corrections.RadialDistortion: dict(center=(1, 2), coefficients=(1.0,)),
    }
    for cls in owners:
        document = description[cls.__name__]
        properties = document["properties"]
        kwargs = required.get(cls, {})
        field_names = {f.name for f in fields(cls) if f.init}
        assert properties.keys() - {"type"} == field_names
        assert set(get_type_hints(cls)) >= field_names
        for f in fields(cls):
            if f.init and f.default is MISSING and f.default_factory is MISSING:
                assert f.name in document["required"]
        data = dict(kwargs, **{k: v["default"] for k, v in properties.items() if "default" in v})
        wire = getattr(wire_options, cls.__name__)
        if "type" in properties:
            data["type"] = cls.__name__
        values = msgspec.json.decode(msgspec.json.encode(data), type=wire)
        assert cls(**{k: getattr(values, k) for k in field_names}) == cls(**kwargs)


def test_canonical_choices_and_physically_distinct_units(description):
    def p(owner, field):
        return description[owner]["properties"][field]

    assert p("Simulate", "translation_sigma")["x-unit"] == "geometry_length"
    assert p("Simulate", "translation_sigma")["default"] == 0.0
    assert p("Paganin", "energy_kev")["x-unit"] == "keV"
    assert p("Paganin", "pixel_size")["x-unit"] == "m"
    assert p("Grid", "vz")["x-unit"] == "geometry_length"
    assert p("Open", "angles")["x-unit"] == "degree"
    assert p("Open", "flat_exposure_times")["x-unit"] == "s"
    assert p("AlignConfig", "tolerance")["x-unit"] == "working_pixel"
    assert p("AlignConfig", "slip_scale")["x-unit"] == "native_pixel"
    assert set(p("Align", "fit")["x-choices"]) == {
        "center",
        "tilt",
        "roll",
        "rotations",
        "translations",
    }
    assert p("FBPConfig", "filter")["x-choices"] == ["ramp", "hann", "shepp-logan"]
    assert recon.FBPConfig(filter="HANN").filter == "hann"
    assert p("Slice", "method") == p("Reconstruct", "method")
    assert p("Align", "tolerance")["x-unit"] == "working_pixel"


def test_describable_signatures_preserve_the_native_host_import_path():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tomojax as tj; from tomojax.recon import CGLSConfig; "
            "CGLSConfig(); funcs=(tj.Scan, tj.reconstruct, tj.inspect, tj.load); "
            "assert 'jax' not in sys.modules; from typing import get_type_hints; "
            "assert 'config' in get_type_hints(tj.reconstruct)",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_each_method_config_names_its_method():
    from tomojax._parameters import Method, ReconMethod
    from tomojax.recon.api import method_config

    names = get_args(ReconMethod.__value__)
    assert get_args(Method.__value__) == ("auto", *names)
    assert sorted(kind.method for kind in get_args(MethodConfig.__value__)) == sorted(names)
    for name in names:
        assert method_config(name).method == name


def test_parameter_help_is_plain_text(description):
    props = description["Simulate"]["properties"]
    assert "tomojax.datasets.SimulationArtifacts" in props["artifacts"]["description"]
    assert "grid" in props["size"]["description"]
    for model in description.values():
        for prop in model.get("properties", {}).values():
            assert "`" not in prop.get("description", "")
