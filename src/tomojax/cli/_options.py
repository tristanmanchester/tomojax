"""Conventions every `tomojax` command shares.

Commands read ``INPUT`` (positional) and write ``-o/--output``, which must not
exist unless ``--force`` is given. ``--help`` lists the options most runs
need; expert settings stay out of it and go in the TOML file passed with
``--config``, whose keys ``--config-keys`` prints. Exit status is 0 on
success, 1 when the run fails and 2 for a usage error (including a missing
input or an existing output).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

EXPERT_EPILOG = (
    "Expert settings are TOML keys for --config; --config-keys lists them with their "
    "defaults. Exit status: 0 success, 1 failure, 2 usage error."
)


def add_output(parser: argparse.ArgumentParser, description: str) -> None:
    """Add ``-o/--output`` (stored as ``out``) and ``--force``."""
    _ = parser.add_argument(
        "-o", "--output", dest="out", required=True, metavar="OUTPUT", help=description
    )
    add_force(parser)


def add_force(parser: argparse.ArgumentParser) -> None:
    """Add ``--force``, which allows replacing existing outputs."""
    _ = parser.add_argument(
        "--force", action="store_true", help="Replace the output if it already exists"
    )


def options(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """The parser's arguments (argparse has no public way to list them)."""
    return parser._actions  # noqa: SLF001


def hide_expert(parser: argparse.ArgumentParser, public: Iterable[str]) -> None:
    """Keep only ``public`` options (and positionals) in ``--help``; record the others' help."""
    keep = set(public) | {"-h", "--help", "--config", "--config-keys", "-o", "--output", "--force"}
    for action in options(parser):
        if action.option_strings and set(action.option_strings).isdisjoint(keep):
            if action.help != argparse.SUPPRESS:
                action.expert_help = action.help  # type: ignore[attr-defined]
            action.help = argparse.SUPPRESS
    parser.epilog = (
        EXPERT_EPILOG if parser.epilog is None else f"{parser.epilog}\n\n{EXPERT_EPILOG}"
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter


class _ConfigKeys(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        settings: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)  # type: ignore[arg-type]
        self.settings: Mapping[str, object] = settings or {}

    def __call__(self, parser: argparse.ArgumentParser, *_args: object) -> NoReturn:
        entries: dict[str, str] = {}
        for action in options(parser):
            if not action.option_strings or action.dest in {"help", "config", "out", "force"}:
                continue
            if isinstance(action, _ConfigKeys):
                continue
            text = getattr(action, "expert_help", None) or action.help
            text = "" if text in (None, argparse.SUPPRESS) else str(text)
            # A flag that stores False documents what false means.
            negated = action.nargs == 0 and cast("object", action.const) is False
            if negated:
                text = f"false: {text}" if text else ""
            default = cast("object", action.default)
            value = "..." if default is None else _toml(default)
            line = f"{action.dest} = {value}" + (f"  # {text}" if text else "")
            if action.dest not in entries or not negated:
                _ = entries.setdefault(action.dest, line)
        for key, value in self.settings.items():
            _ = entries.setdefault(key, f"{key} = {'...' if value is None else _toml(value)}")
        print("\n".join(entries[key] for key in sorted(entries)))
        parser.exit(0)


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(v) for v in cast("Iterable[object]", value)) + "]"
    return str(value)


def add_config(
    parser: argparse.ArgumentParser, *, settings: Mapping[str, object] | None = None
) -> None:
    """Add ``--config FILE`` and ``--config-keys``.

    A config file's keys are the parser's options, and ``settings``: expert
    settings with no option, by name and default, which
    :func:`tomojax.cli.config.parse_args_with_config` returns as given.
    """
    _ = parser.add_argument(
        "--config", metavar="FILE", help="Read option defaults (and expert settings) from TOML"
    )
    _ = parser.add_argument(
        "--config-keys",
        action=_ConfigKeys,
        settings=settings,
        help="List the keys a --config file may set, and exit",
    )


def config_settings(parser: argparse.ArgumentParser) -> Mapping[str, object]:
    """The expert settings ``parser``'s config file may hold besides its options."""
    for action in options(parser):
        if isinstance(action, _ConfigKeys):
            return action.settings
    return {}


def check_paths(
    parser: argparse.ArgumentParser,
    *,
    inputs: Iterable[str | Path | None] = (),
    outputs: Iterable[str | Path | None] = (),
    force: bool = False,
) -> None:
    """Exit with a usage error for a missing input or an existing output (without ``--force``)."""
    for path in inputs:
        if path is not None and not Path(path).exists():
            parser.error(f"input not found: {path}")
    for path in outputs:
        if path is not None and Path(path).exists() and not force:
            parser.error(f"output already exists: {path} (pass --force to replace it)")


def fail(command: str, message: str) -> int:
    """Report a run failure on stderr and return exit status 1."""
    print(f"tomojax {command}: error: {message}", file=sys.stderr)
    return 1


__all__ = [
    "EXPERT_EPILOG",
    "add_config",
    "add_force",
    "add_output",
    "check_paths",
    "config_settings",
    "fail",
    "hide_expert",
    "options",
]
