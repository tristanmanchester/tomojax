"""Config-file parsing helpers for TomoJAX CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib
from typing import TYPE_CHECKING, TypedDict, cast, overload

from tomojax._typed_arrays import object_list
from tomojax.cli import _console
from tomojax.cli._options import config_settings, options
from tomojax.cli._progress import ProgressParser

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


ConfigValue = object


class ConfigMetadata(TypedDict):
    """Config precedence and effective arguments for a command invocation."""

    config_path: str | None
    config_file_values: dict[str, ConfigValue]
    explicit_cli_keys: list[str]
    effective_options: dict[str, ConfigValue]
    settings: dict[str, ConfigValue]


@overload
def parse_args_with_config(
    parser: ProgressParser,
    argv: Sequence[str] | None = None,
    *,
    required: Sequence[str] = (),
    namespace: None = None,
) -> tuple[argparse.Namespace, ConfigMetadata]: ...


@overload
def parse_args_with_config[N: argparse.Namespace](
    parser: ProgressParser,
    argv: Sequence[str] | None = None,
    *,
    required: Sequence[str] = (),
    namespace: N,
) -> tuple[N, ConfigMetadata]: ...


def parse_args_with_config(
    parser: ProgressParser,
    argv: Sequence[str] | None = None,
    *,
    required: Sequence[str] = (),
    namespace: argparse.Namespace | None = None,
) -> tuple[argparse.Namespace, ConfigMetadata]:
    """Parse CLI args with optional TOML defaults.

    Precedence is:
        argparse defaults < config file < explicit CLI flags

    A config file's expert settings (``add_config(parser, settings=...)``)
    have no option; they are returned as written, under ``"settings"``.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.set_invocation(raw_argv)
    config_path = _discover_config_path(parser, raw_argv)
    config_values: dict[str, ConfigValue] = {}
    config_defaults: dict[str, ConfigValue] = {}
    settings: dict[str, ConfigValue] = {}
    explicit_dests = _explicit_cli_dests(parser, raw_argv)

    if config_path is not None:
        config_values = _load_config_file(parser, config_path)
        _validate_config_keys(parser, config_path, config_values)
        known = _config_actions_by_dest(parser)
        settings = {k: v for k, v in config_values.items() if k not in known}
        config_values = {k: v for k, v in config_values.items() if k in known}
        config_defaults = _coerce_config_defaults(parser, config_path, config_values)
        defaults = {
            dest: value for dest, value in config_defaults.items() if dest not in explicit_dests
        }
        parser.set_defaults(**defaults)

    args = parser.parse_args(raw_argv, namespace=namespace or argparse.Namespace())
    _validate_required(parser, args, required)

    effective_options: dict[str, object] = vars(args)
    metadata: ConfigMetadata = {
        "config_path": str(config_path) if config_path is not None else None,
        "config_file_values": config_defaults,
        "explicit_cli_keys": sorted(explicit_dests),
        "effective_options": effective_options.copy(),
        "settings": settings,
    }
    return args, metadata


def _discover_config_path(
    parser: ProgressParser,
    argv: Sequence[str],
) -> Path | None:
    config_parser = ProgressParser(add_help=False, prog=parser.prog, allow_abbrev=False)
    config_parser.set_invocation(list(argv))
    _ = config_parser.add_argument("--config", default=None)

    class ConfigArgs(argparse.Namespace):
        config: str | None

    args, _ = config_parser.parse_known_args(argv, namespace=ConfigArgs())
    return None if args.config is None else Path(args.config)


def _load_config_file(
    parser: ProgressParser,
    path: Path,
) -> dict[str, ConfigValue]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        parser.error(
            "YAML config files are not supported by this runtime; use TOML (.toml) instead"
        )
    if suffix != ".toml":
        parser.error(f"unsupported config file extension for {path}; expected .toml")

    try:
        with path.open("rb") as fh:
            payload = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        parser.error(f"invalid TOML config {path}: {exc}")
    except OSError as exc:
        parser.error(f"could not read config file {path}: {exc}")

    return payload


# Config keys that were renamed, and their new names. An old key fails with
# the new name rather than being accepted as a second spelling.
_RETIRED_KEYS = {
    "algo": "method",
    "iters": "iterations",
    "lambda_tv": "tv_weight",
    "positivity": "nonnegative",
    "spdhg_seed": "seed",
    "tv_prox_iters": "tv_prox_iterations",
    "L": "lipschitz",
    "apply_saved_alignment": "poses",
    "spdhg_tau": "tau",
    "spdhg_sigma_data": "sigma_data",
    "spdhg_sigma_tv": "sigma_tv",
}


def _validate_config_keys(
    parser: ProgressParser,
    path: Path,
    values: Mapping[str, ConfigValue],
) -> None:
    valid_keys = _config_actions_by_dest(parser).keys() | config_settings(parser).keys()
    for key in values:
        new = _RETIRED_KEYS.get(str(key))
        if new is not None and new in valid_keys:
            parser.error(f"config key '{key}' in {path} was renamed '{new}'")
    unknown = sorted(str(key) for key in values if str(key) not in valid_keys)
    if unknown:
        message = f"unknown config key(s) in {path}: {', '.join(unknown)}"
        listing = f"{parser.prog} --config-keys lists every key"
        close = _console.suggest(unknown[0], valid_keys)
        hint = f"did you mean {close}? {listing}" if close else listing
        full = f"{message}. Valid keys: {', '.join(sorted(valid_keys))}"
        parser.fail(full, hint=hint, display=message)


def _coerce_config_defaults(
    parser: ProgressParser,
    path: Path,
    values: Mapping[str, ConfigValue],
) -> dict[str, ConfigValue]:
    actions = _config_actions_by_dest(parser)
    coerced: dict[str, ConfigValue] = {}
    for raw_key, raw_value in values.items():
        dest = str(raw_key)
        action = actions[dest]
        try:
            coerced[dest] = _coerce_action_value(action, raw_value)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"invalid value for config key '{dest}' in {path}: {exc}")
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid value for config key '{dest}' in {path}: {exc}")
        _validate_choices(parser, path, dest, action, coerced[dest])
    return coerced


def _config_actions_by_dest(
    parser: ProgressParser,
) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in options(parser):
        if action.dest in (argparse.SUPPRESS, "help", "config", "out", "force"):
            continue
        if not action.option_strings:
            continue
        _ = actions.setdefault(action.dest, action)
    return actions


def _explicit_cli_dests(
    parser: ProgressParser,
    argv: Sequence[str],
) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in options(parser):
        if action.dest in (argparse.SUPPRESS, "help"):
            continue
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit


def _coerce_action_value(action: argparse.Action, value: ConfigValue) -> ConfigValue:
    action_const: object = action.const
    flag = action.nargs == 0 and isinstance(action_const, bool)
    if flag or isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise TypeError("expected a boolean")
        return bool(value)

    list_value = object_list(value)
    if action.__class__.__name__ == "_AppendAction":
        values = list_value if isinstance(value, list) else [value]
        return [_coerce_scalar(action, item) for item in values]

    nargs = action.nargs
    if nargs is None or nargs == "?":
        if isinstance(value, list):
            raise TypeError("expected a scalar value")
        return _coerce_scalar(action, value)

    if not isinstance(value, list):
        raise TypeError("expected a list value")
    if isinstance(nargs, int) and len(list_value) != nargs:
        raise ValueError(f"expected {nargs} values, got {len(list_value)}")
    if nargs == "+" and len(list_value) == 0:
        raise ValueError("expected at least one value")
    return [_coerce_scalar(action, item) for item in list_value]


def _coerce_scalar(action: argparse.Action, value: ConfigValue) -> ConfigValue:
    if action.type is None:
        return value
    if action.type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected an integer, got {type(value).__name__}")
        return int(value)
    if action.type is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"expected a number, got {type(value).__name__}")
        return float(value)
    coercer = cast("Callable[[ConfigValue], ConfigValue]", action.type)
    return coercer(value)


def _validate_choices(
    parser: ProgressParser,
    path: Path,
    dest: str,
    action: argparse.Action,
    value: ConfigValue,
) -> None:
    if action.choices is None:
        return

    choices = cast("Iterable[ConfigValue]", action.choices)
    list_value = object_list(value)
    values = list_value if isinstance(value, list) else [value]
    invalid: list[ConfigValue] = [item for item in values if item not in choices]
    if invalid:
        choices_text = ", ".join(str(choice) for choice in choices)
        parser.error(
            f"invalid choice for config key '{dest}' in {path}: "
            f"{invalid[0]!r} (choose from {choices_text})"
        )


def _validate_required(
    parser: ProgressParser,
    args: argparse.Namespace,
    required: Sequence[str],
) -> None:
    missing: list[str] = []
    for dest in required:
        value = getattr(args, dest, None)
        if value is None or value == "":
            missing.append(f"--{dest.replace('_', '-')}")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
