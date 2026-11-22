"""The public surface speaks one vocabulary.

The surface is what a user types or reads back: every public name, parameter, field,
property and method; the recon configs and ``tomojax.datasets``; every CLI command, flag and choice; the ``info``
keys and string values of a small ``tj.align`` and ``tj.reconstruct``; the keys
``tj.save`` writes and ``tj.inspect`` reports; and the code blocks of the docs. Each rule
names the replacement when it fails. ``ALLOWED`` holds sites allowed to break a rule, with
the reason (empty since the v1 renames); a row that no longer occurs fails too, so the
list only shrinks.
"""

import argparse
from collections.abc import Iterator
import dataclasses
from fnmatch import fnmatch
import inspect
import json
from pathlib import Path
import re

import h5py
import numpy as np

import tomojax as tj
from tomojax import datasets, recon
from tomojax.cli.api import PRODUCT_COMMANDS
from tomojax.cli.main import main as cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "README.md",
    ROOT / "src/tomojax/cli/README.md",
]

# Names retired at a site, with what replaces them. A site's place is matched with fnmatch.
RETIRED = {
    ("*", "batch_views"): "views_per_batch",
    ("*", "innovation_scale"): 'motion="independent" (the default) or slip_scale',
    ("*", "select_views"): "views",
    ("*", "stream_projections"): "stream",
    ("*", "joseph_interpolation"): "interpolation",
    ("*", "recon_rel_tol"): "tolerance",
    ("*", "recon_patience"): "patience",
    ("*", "initial_epochs"): "start_epochs",
    ("*", "initial_subsets"): "start_subsets",
    ("*", "pose_for_view"): "pose_matrix",
    ("*", "pose_dofs"): "pose_columns",
    ("*", "estimate_center"): 'tj.align(fit="center", method=...)',
    ("*", "ray_integrator"): "(removed: one projector model)",
    ("*", "projector_model"): "(removed: one projector model)",
    ("*", "gather_dtype"): "(removed: one projector model)",
    ("*", "projector_unroll"): "(removed: one projector model)",
    ("*", "checkpoint_projector"): "(removed: one projector model)",
    ("*", "grad_mode"): "(removed: one projector model)",
    ("tj.Scan", "positions"): "view_positions",
    ("tj.Scan.binned()", "factor"): "binning",
    ("tj.ConeGeometry", "poses"): "pose_matrices",
    ("tj.Alignment", "volume"): "image",
    ("tj.load()", "poses"): "apply_poses",
    ("cli *", "poses"): "apply_poses",
    ("recon.CGLSConfig", "rtol"): "tolerance",
    ("recon.CGLSConfig", "atol"): "absolute_tolerance",
    ("*align*info*", "budget"): "iterations, image_iterations",
    ("*align*info*", "outer"): "iterations",
    ("*align*info*", "inner"): "image_iterations",
    ("*align*info*", "init_iterations"): "initial_image_iterations",
    ("*align*info*", "lm_initial"): "(internal)",
    ("*align*info*", "corrections"): "calibration (fitted globals) and poses (per view)",
    ("cli", "recon"): "reconstruct",
    ("cli", "import"): "correct",
    ("cli", "load"): "correct",
    ("cli", "preprocess"): "correct",
    ("docs *", "tomojax recon"): "tomojax reconstruct",
    ("docs *", "tomojax import"): "tomojax correct",
    ("docs *", "tomojax load"): "tomojax correct",
    ("docs *", "tomojax preprocess"): "tomojax correct",
}
BRITISH = {
    "centre": "center",
    "regularis": "regulariz",
    "colour": "color",
    "normalis": "normaliz",
    "optimis": "optimiz",
    "initialis": "initializ",
    "analys": "analyz",
}
# Unit words, anywhere in a name: angles are in degrees and lengths in the geometry's unit.
UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians", "px"}
SYNONYMS = {
    "rtol": "tolerance",
    "atol": "absolute_tolerance",
    "rel_tol": "tolerance",
    "tol": "tolerance",
    "n_iter": "iterations",
    "max_iter": "iterations",
    "roi": "region",
}
# CLI commands and the API function each mirrors; parameters with no flag by design.
MIRRORS = {
    "align": ("align", {"scan", "config"}),
    "reconstruct": ("reconstruct", {"scan", "out", "config", "devices"}),
    "correct": ("load", {"path", "detector", "grid", "exposure_times", "flat_exposure_times"}),
    "export": ("export", {"path", "reconstruction"}),
    "inspect": ("inspect", {"path"}),
    "simulate": ("simulate", {"artefacts"}),
}
# Sites allowed to break a rule, by surface: (place, name) -> the reason. Empty: the renames
# of v1 are done.
ALLOWED: dict[str, dict[tuple[str, str], str]] = {
    "api": {},
    "results": {},
    "docs": {},
    "mirror": {},
}


@dataclasses.dataclass(frozen=True)
class Site:
    place: str
    name: str


def _parameters(place, function):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    for parameter in signature.parameters.values():
        if parameter.name not in ("self", "cls"):
            yield Site(f"{place}()", parameter.name)
            if isinstance(parameter.default, str):
                yield Site(f"{place}({parameter.name}=)", parameter.default)


def _class(place, cls):
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            yield Site(place, field.name)
            if isinstance(field.default, str):
                yield Site(f"{place}.{field.name}", field.default)
    yield from _parameters(place, cls)
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue
        yield Site(place, name)
        function = getattr(value, "__func__", value)
        if callable(function) and not isinstance(value, type):
            yield from _parameters(f"{place}.{name}", function)


def api_sites() -> Iterator[Site]:
    for name in tj.__all__:
        yield Site("tomojax", name)
        value = getattr(tj, name)
        if inspect.isclass(value):
            yield from _class(f"tj.{name}", value)
        elif callable(value):
            yield from _parameters(f"tj.{name}", value)
    for name in recon.__all__:
        value = getattr(recon, name)
        if inspect.isclass(value) and name.endswith("Config"):
            yield from _class(f"recon.{name}", value)
    for name in datasets.__all__:
        yield Site("tomojax.datasets", name)
        value = getattr(datasets, name)
        if inspect.isclass(value):
            yield from _class(f"datasets.{name}", value)
        elif callable(value):
            yield from _parameters(f"datasets.{name}", value)


class _Parsed(Exception):
    def __init__(self, parser):
        self.parser = parser


def parser(command, monkeypatch) -> argparse.ArgumentParser:
    """The command's parser, as ``tomojax <command>`` builds it."""

    parse = argparse.ArgumentParser.parse_known_args

    def stop(self, *args, **kwargs):
        if not self.add_help:  # a pre-parser (--config) reading ahead
            return parse(self, *args, **kwargs)
        raise _Parsed(self)

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_known_args", stop)
        try:
            cli([command])
        except _Parsed as parsed:
            return parsed.parser
    raise AssertionError(f"tomojax {command} built no parser")


def _flag(option):
    return option.lstrip("-").replace("-", "_")


def cli_sites(monkeypatch) -> Iterator[Site]:
    for command in PRODUCT_COMMANDS:
        yield Site("cli", command.name)
        for action in parser(command.name, monkeypatch)._actions:
            for option in action.option_strings:
                yield Site(f"cli {command.name}", _flag(option))
            for choice in action.choices or ():
                if isinstance(choice, str):
                    yield Site(f"cli {command.name} {action.option_strings[0]}", choice)


def _info(place, value) -> Iterator[Site]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield Site(place, str(key))
            yield from _info(f"{place}[{key}]", item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _info(place, item)
    elif isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value):
        yield Site(place, value)


def _small_scan():
    grid, detector = tj.Grid(12, 12, 8, 1, 1, 1), tj.Detector(16, 10, 1, 1)
    geometry = tj.LaminographyGeometry(grid, detector, np.linspace(0, 360, 16, endpoint=False), 30)
    z, y, x = np.mgrid[:8, :12, :12]
    volume = (((x - 5.5) ** 2 + (y - 6) ** 2 + (z - 3.5) ** 2) < 9).astype(np.float32)
    return tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)


def _json(text) -> object:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _saved(path) -> Iterator[Site]:
    """Every group, dataset and attribute name, and the keys of JSON held in them."""
    place = f"saved {path.stem}"
    with h5py.File(path) as f:
        names = []
        f.visit(names.append)
        for name in ["", *names]:
            item = f[name] if name else f
            yield Site(place, name.rsplit("/", 1)[-1])
            values = list(item.attrs.items())
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "OS" and item.size == 1:
                values.append((None, item[()]))
            for key, value in values:
                if key is not None:
                    yield Site(place, key)
                text = value.decode() if isinstance(value, bytes) else value
                if isinstance(parsed := _json(text), dict | list):
                    yield from _info(place, parsed)


def result_sites(tmp_path) -> Iterator[Site]:
    scan = _small_scan()
    alignment = tj.align(scan, fit=("translations", "center"))
    reconstruction = tj.reconstruct(scan, method="cgls", iterations=2)
    yield from _info("align info", alignment.info)
    yield from _info("reconstruct info", reconstruction.info)
    for name, item in (
        ("scan", scan),
        ("reconstruction", reconstruction),
        ("alignment", alignment),
    ):
        tj.save(tmp_path / f"{name}.nxs", item)
        yield from _saved(tmp_path / f"{name}.nxs")
        yield from _info(f"inspect {name}", tj.inspect(tmp_path / f"{name}.nxs"))


def docs_sites() -> Iterator[Site]:
    for path in DOCS:
        blocks = re.findall(r"```[^\n]*\n(.*?)```", path.read_text(), re.S)
        code = "\n".join(blocks)
        place = f"docs {path.relative_to(ROOT)}"
        for word in sorted(set(re.findall(r"[A-Za-z_-][A-Za-z0-9_-]*", code))):
            yield Site(place, word.lstrip("-").replace("-", "_"))
        for command in sorted(set(re.findall(r"tomojax ([a-z]+)", code))):
            yield Site(place, f"tomojax {command}")


def violations(sites) -> dict[tuple[str, str], str]:
    found = {}
    for site in sites:
        name, key = site.name, (site.place, site.name)
        for (place, old), new in RETIRED.items():
            if name == old and fnmatch(site.place, place):
                found[key] = f"retired: use {new}"
        for british, american in BRITISH.items():
            if british in name.lower():
                found[key] = f"American spelling: {american}"
        # A ``units`` value names the unit a file records; it is meant to be one.
        if UNITS & set(name.lower().split("_")) and not site.place.endswith("[units]"):
            found[key] = "no unit in a name: angles are in degrees, said once in the docs"
        if name in SYNONYMS:
            found[key] = f"one concept, one word: {SYNONYMS[name]}"
    return found


def mirror_violations(monkeypatch) -> dict[tuple[str, str], str]:
    found = {}
    commands = {command.name for command in PRODUCT_COMMANDS}
    for command, (function, by_design) in MIRRORS.items():
        if command not in commands:
            found["cli", command] = f"no `tomojax {command}` command for tj.{function}"
            continue
        if not hasattr(tj, function):
            found["tomojax", function] = f"no tj.{function} behind `tomojax {command}`"
            continue
        flags = {
            _flag(option)
            for action in parser(command, monkeypatch)._actions
            for option in action.option_strings
        }
        for parameter in inspect.signature(getattr(tj, function)).parameters:
            if parameter not in flags | by_design:
                found[f"cli {command}", parameter] = f"tj.{function}({parameter}=) has no flag"
    return found


def _check(surface: str, found: dict[tuple[str, str], str]) -> None:
    allowed = ALLOWED[surface]
    new = sorted(f"{p}: {n!r} ({why})" for (p, n), why in found.items() if (p, n) not in allowed)
    stale = sorted(f"{key} ({why})" for key, why in allowed.items() if key not in found)
    assert not new, "\n".join(new)
    assert not stale, "no longer occurs, so drop from ALLOWED:\n" + "\n".join(stale)


def test_the_api_and_cli_use_one_vocabulary(monkeypatch):
    _check("api", violations([*api_sites(), *cli_sites(monkeypatch)]))


def test_results_and_saved_files_use_one_vocabulary(tmp_path):
    _check("results", violations(result_sites(tmp_path)))


def test_the_docs_code_uses_one_vocabulary():
    _check("docs", violations(docs_sites()))


def test_each_cli_command_mirrors_its_api_function(monkeypatch):
    _check("mirror", mirror_violations(monkeypatch))
