"""Hold the code to its recorded shape: public API, CLI vocabulary and debt ratchets.

When a test here fails after an intended change, run
``python tools/guardrails.py --write`` and commit the updated records with the
change, so the review shows what moved.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import inspect
import json
from pathlib import Path
import tomllib
from types import ModuleType

import pytest

import tomojax

pytestmark = pytest.mark.surface

ROOT = Path(__file__).resolve().parents[1]
UPDATE = "run `python tools/guardrails.py --write` and commit the result"


def _guardrails() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guardrails", ROOT / "tools" / "guardrails.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_api_matches_the_recorded_surface() -> None:
    guardrails = _guardrails()
    recorded = guardrails.SURFACE_FILE.read_text(encoding="utf-8").splitlines()
    current = guardrails.api_surface().splitlines()
    diff = list(difflib.unified_diff(recorded, current, "recorded", "current", lineterm=""))
    assert not diff, "the public API changed; if intended, " + UPDATE + ":\n" + "\n".join(diff[:80])


def _compare(name: str, recorded: object, current: object, problems: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(set(recorded) | set(current)):
            _compare(f"{name}.{key}", recorded.get(key, 0), current.get(key, 0), problems)
    elif isinstance(recorded, list) and isinstance(current, list):
        added = sorted(set(current) - set(recorded))
        removed = sorted(set(recorded) - set(current))
        if added:
            problems.append(f"{name} gained {added}: fix these rather than raise the ratchet")
        if removed:
            problems.append(f"{name} lost {removed}: well done; {UPDATE} to lock it in")
    elif current != recorded:
        verb = "rose" if current > recorded else "fell"  # pyright: ignore[reportOperatorIssue]
        advice = "bring it back down" if verb == "rose" else f"well done; {UPDATE} to lock it in"
        problems.append(f"{name} {verb} from {recorded} to {current}: {advice}")


def test_code_health_ratchets_only_tighten() -> None:
    guardrails = _guardrails()
    recorded = json.loads(guardrails.RATCHET_FILE.read_text(encoding="utf-8"))
    problems: list[str] = []
    _compare("ratchets", recorded, guardrails.ratchets(), problems)
    assert not problems, "\n".join(problems)


def test_no_subpackage_shares_a_root_name() -> None:
    # A subpackage named like a root function replaces it once imported.
    package = ROOT / "src" / "tomojax"
    for name in tomojax.__all__:
        assert not (package / name).is_dir(), f"tomojax.{name} is also a subpackage"
        assert not (package / f"{name}.py").is_file(), f"tomojax.{name} is also a module"


_STANDARD_OPTIONS = {"-h", "--help", "-o", "--output", "--force", "--config", "--config-keys"}
# Public CLI options and the Python keyword each one is, by command. An option
# with no Python equivalent says why it is command-line only.
_CLI_VOCABULARY: dict[str, dict[str, tuple[str, str] | str]] = {
    "recon": {
        "--method": ("reconstruct", "method"),
        "--filter": ("reconstruct", "filter"),
        "--iterations": ("reconstruct", "iterations"),
        "--tv-weight": ("reconstruct", "tv_weight"),
        "--nonnegative": ("reconstruct", "nonnegative"),
        "--warm-start": ("reconstruct", "warm_start"),
        "--seed": ("reconstruct", "seed"),
        "--grid": ("reconstruct", "grid"),
        "--poses": ("load", "poses"),
        "--roi": "crops the grid to the field of view; Python passes the grid",
        "--preview": "file output",
        "--manifest": "file output",
        "--progress": "terminal display",
    },
    "align": {
        "--mode": ("align", "mode"),
        "--quality": ("align", "quality"),
        "--levels": ("align", "levels"),
        "--freeze": ("align", "freeze"),
        "--grid": ("align", "grid"),
        "--checkpoint": ("align", "checkpoint"),
        "--poses": ("load", "poses"),
        "--roi": "crops the grid to the field of view; Python passes the grid",
        "--manifest": "file output",
        "--dry-run": "prints the plan; Python calls alignment_plan",
        "--progress": "terminal display",
    },
    "preprocess": {
        "--flats": ("load_frames", "flats"),
        "--darks": ("load_frames", "darks"),
        "--angles": ("load_frames", "angles"),
        "--select-views": "index ranges; Python calls Frames.selected",
        "--reject-views": "index ranges; Python calls Frames.selected",
        "--crop": "Y0:Y1,X0:X1; Python calls Frames.cropped",
        "--reject-outliers": "the RejectViews step",
        "--zingers": "the Zingers step",
        "--remove-stripes": "the Stripes step",
        "--beam-hardening": "the BeamHardening step",
        "--preview": "file output",
    },
}
_HELP_BUDGET = 16


def _actions(command: str) -> list[argparse.Action]:
    from tomojax.cli._options import options  # check-public-imports: allow-private

    module = {"import": "tomojax.cli.import_"}.get(command, f"tomojax.cli.{command}")
    parser_module = importlib.import_module(module)
    build = getattr(parser_module, "build_parser", None) or parser_module._build_parser
    return options(build())


def _public_options(command: str) -> set[str]:
    return {
        option
        for action in _actions(command)
        if action.help != argparse.SUPPRESS
        for option in action.option_strings
        if option.startswith("--")
        and option not in _STANDARD_OPTIONS
        # --no-x is --x's other half, not an option of its own.
        and not (isinstance(action, argparse.BooleanOptionalAction) and option.startswith("--no-"))
    }


@pytest.mark.parametrize("command", sorted(_CLI_VOCABULARY))
def test_cli_options_use_the_python_names(command: str) -> None:
    vocabulary = _CLI_VOCABULARY[command]
    assert _public_options(command) == set(vocabulary), (
        "every public option needs an entry in _CLI_VOCABULARY: its Python keyword, "
        "or why it is command-line only"
    )
    for option, target in vocabulary.items():
        if isinstance(target, tuple):
            function, keyword = target
            parameters = inspect.signature(getattr(tomojax, function)).parameters
            assert keyword in parameters, f"{option} names tomojax.{function}({keyword}=...)"
            assert option.removeprefix("--").replace("-", "_") == keyword


def _configuration_fields(command: str) -> set[str]:
    from dataclasses import fields

    from tomojax.alignment import AlignConfig
    from tomojax.recon import CGLSConfig, FBPConfig, FistaConfig, SPDHGConfig

    if command == "align":
        return {item.name for item in fields(AlignConfig) if item.init}
    configs = (FBPConfig, CGLSConfig, FistaConfig, SPDHGConfig)
    # Arrays and devices have no TOML form.
    return {item.name for c in configs for item in fields(c) if item.init} - {"support", "devices"}


@pytest.mark.parametrize("command", ["align", "recon"])
def test_options_are_stored_under_their_own_names(command: str) -> None:
    # A --config key is the option's dest, so it must be the option's name; the
    # file's other keys, its expert settings, are the configuration's fields.
    from tomojax.cli._options import config_settings  # check-public-imports: allow-private

    module = importlib.import_module(f"tomojax.cli.{command}")
    for action in _actions(command):
        for option in action.option_strings:
            if option.startswith("--") and option not in _STANDARD_OPTIONS:
                name = option.removeprefix("--").removeprefix("no-").replace("-", "_")
                assert name == action.dest, f"{option} stores {action.dest!r}"
    assert set(config_settings(module.build_parser())) == _configuration_fields(command)


@pytest.mark.parametrize("command", _guardrails().CLI_COMMANDS)
def test_cli_help_lists_only_the_options_most_runs_need(command: str) -> None:
    count = len(_public_options(command))
    assert count <= _HELP_BUDGET, (
        f"tomojax {command} --help lists {count} options; move expert ones to --config "
        "with tomojax.cli._options.hide_expert"
    )


def test_version_matches_the_package_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert tomojax.__version__ == project["project"]["version"]
