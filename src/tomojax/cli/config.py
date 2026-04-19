from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import tomllib
from typing import Any


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    required: Sequence[str] = (),
) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse CLI args with optional TOML defaults.

    Precedence is:
        argparse defaults < config file < explicit CLI flags
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _discover_config_path(parser, raw_argv)
    config_values: dict[str, Any] = {}
    config_defaults: dict[str, Any] = {}
    explicit_dests = _explicit_cli_dests(parser, raw_argv)

    if config_path is not None:
        config_values = _load_config_file(parser, config_path)
        _validate_config_keys(parser, config_path, config_values)
        config_defaults = _coerce_config_defaults(parser, config_path, config_values)
        parser.set_defaults(
            **{
                dest: value
                for dest, value in config_defaults.items()
                if dest not in explicit_dests
            }
        )

    args = parser.parse_args(raw_argv)
    _validate_required(parser, args, required)

    effective_options = dict(vars(args))
    metadata = {
        "config_path": str(config_path) if config_path is not None else None,
        "config_file_values": config_defaults,
        "explicit_cli_keys": sorted(explicit_dests),
        "effective_options": effective_options,
    }
    return args, metadata


def _discover_config_path(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> Path | None:
    config_parser = argparse.ArgumentParser(add_help=False, prog=parser.prog)
    config_parser.add_argument("--config", default=None)
    namespace, _ = config_parser.parse_known_args(list(argv))
    if namespace.config is None:
        return None
    return Path(namespace.config)


def _load_config_file(
    parser: argparse.ArgumentParser,
    path: Path,
) -> dict[str, Any]:
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

    if not isinstance(payload, Mapping):
        parser.error(f"config file {path} must contain top-level key/value pairs")
    return dict(payload)


def _validate_config_keys(
    parser: argparse.ArgumentParser,
    path: Path,
    values: Mapping[str, Any],
) -> None:
    valid_keys = _config_actions_by_dest(parser).keys()
    unknown = sorted(str(key) for key in values if str(key) not in valid_keys)
    if unknown:
        parser.error(
            f"unknown config key(s) in {path}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(valid_keys))}"
        )


def _coerce_config_defaults(
    parser: argparse.ArgumentParser,
    path: Path,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    actions = _config_actions_by_dest(parser)
    coerced: dict[str, Any] = {}
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
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest in (argparse.SUPPRESS, "help", "config"):
            continue
        if not action.option_strings:
            continue
        actions.setdefault(action.dest, action)
    return actions


def _explicit_cli_dests(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
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


def _coerce_action_value(action: argparse.Action, value: Any) -> Any:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise TypeError("expected a boolean")
        return bool(value)

    if isinstance(action, argparse._AppendAction):
        values = value if isinstance(value, list) else [value]
        return [_coerce_scalar(action, item) for item in values]

    nargs = action.nargs
    if nargs is None or nargs == "?":
        if isinstance(value, list):
            raise TypeError("expected a scalar value")
        return _coerce_scalar(action, value)

    if not isinstance(value, list):
        raise TypeError("expected a list value")
    if isinstance(nargs, int) and len(value) != nargs:
        raise ValueError(f"expected {nargs} values, got {len(value)}")
    if nargs == "+" and len(value) == 0:
        raise ValueError("expected at least one value")
    return [_coerce_scalar(action, item) for item in value]


def _coerce_scalar(action: argparse.Action, value: Any) -> Any:
    if action.type is None:
        return value
    return action.type(value)


def _validate_choices(
    parser: argparse.ArgumentParser,
    path: Path,
    dest: str,
    action: argparse.Action,
    value: Any,
) -> None:
    if action.choices is None:
        return

    values = value if isinstance(value, list) else [value]
    invalid = [item for item in values if item not in action.choices]
    if invalid:
        choices = ", ".join(str(choice) for choice in action.choices)
        parser.error(
            f"invalid choice for config key '{dest}' in {path}: "
            f"{invalid[0]!r} (choose from {choices})"
        )


def _validate_required(
    parser: argparse.ArgumentParser,
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
