"""Conventions every `tomojax` command shares.

Commands read ``INPUT`` (positional) and write ``-o/--output``, which must not
exist unless ``--force`` is given. ``--help`` lists the options most runs
need; expert settings stay out of it. A command with ``--config`` takes them in
that TOML file, whose keys ``--config-keys`` prints; the others take them as
options that ``--help`` does not list. Exit status is 0 on
success, 1 on operational failure, 2 on usage errors and 130 on interruption.
"""

from __future__ import annotations

import argparse
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from tomojax.cli._progress import ProgressParser
    from tomojax.cli.products import Provenance

EXPERT_EPILOG = (
    "Expert settings are TOML keys for --config; --config-keys lists them with their defaults."
)
HIDDEN_EPILOG = "Expert options are accepted but not listed here; the documentation describes them."
EXIT_STATUS = "Exit status: 0 success, 1 failure, 2 usage error, 130 interrupted."


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


def hide_expert(
    parser: argparse.ArgumentParser, public: Iterable[str], *, settings: str | None = None
) -> None:
    """Keep only ``public`` options (and positionals) in ``--help``; record the others' help.

    The help ends with ``settings``, what the expert settings are (a command with
    its own kind of settings says so; by default ``--config`` where the command has
    it, else the hidden options), and the exit statuses.
    """
    if settings is None:
        has_config = any("--config" in action.option_strings for action in options(parser))
        settings = EXPERT_EPILOG if has_config else HIDDEN_EPILOG
    keep = set(public) | {"-h", "--help", "--config", "--config-keys", "-o", "--output", "--force"}
    for action in options(parser):
        if action.option_strings and set(action.option_strings).isdisjoint(keep):
            if action.help != argparse.SUPPRESS:
                action.expert_help = action.help  # type: ignore[attr-defined]
            action.help = argparse.SUPPRESS
    ending = f"{settings.rstrip()}\n{EXIT_STATUS}"
    parser.epilog = ending if parser.epilog is None else f"{parser.epilog}\n\n{ending}"


class _ConfigKeys(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        settings: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)  # type: ignore[arg-type]
        self._settings = settings

    @cached_property
    def settings(self) -> Mapping[str, object]:
        """Resolve method defaults only for config validation or listing."""
        return (self._settings() if callable(self._settings) else self._settings) or {}

    def _setting_entries(self, namespace: argparse.Namespace) -> dict[str, str]:
        selected = self.settings
        if callable(self._settings):
            import inspect

            if "method" in inspect.signature(self._settings).parameters:
                method = getattr(namespace, "method", None)
                selected = cast("Mapping[str, object]", cast("Any", self._settings)(method=method))
        formatted: dict[str, str] = {}
        for key, value in selected.items():
            if value is None:
                formatted[key] = f"# {key} = ..."
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
                default_val, note = value
                rhs = "..." if default_val is None else _toml(default_val)
                prefix = "# " if default_val is None else ""
                formatted[key] = f"{prefix}{key} = {rhs}  # {note}"
            else:
                formatted[key] = f"{key} = {_toml(value)}"
        return formatted

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        *_args: object,
    ) -> NoReturn:
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
            suffix = f"  # {text}" if text else ""
            prefix = "# " if default is None else ""
            rhs = "..." if default is None else _toml(default)
            line = f"{prefix}{action.dest} = {rhs}{suffix}"
            if action.dest not in entries or not negated:
                _ = entries.setdefault(action.dest, line)
        for key, line in self._setting_entries(namespace).items():
            _ = entries.setdefault(key, line)
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
    parser: argparse.ArgumentParser,
    *,
    settings: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
) -> None:
    """Add ``--config FILE`` and ``--config-keys``.

    A config file's keys are the parser's options, and ``settings``: expert
    settings with no option, by name and default, which
    :func:`tomojax.cli.config.parse_args_with_config` returns as given. A callable
    defers loading method defaults until config validation or listing needs them.
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
    parser: ProgressParser,
    *,
    inputs: Iterable[str | Path | None] = (),
    outputs: Iterable[str | Path | None] = (),
    force: bool = False,
) -> None:
    """Exit with a usage error for missing inputs, colliding destinations, or existing outputs."""
    resolved_inputs: set[Path] = set()
    for path in inputs:
        if path is not None:
            candidate = Path(path)
            if not candidate.exists():
                if parser.json_output:
                    raise FileNotFoundError(f"input not found: {path}")
                parser.error(f"input not found: {path}")
            resolved_inputs.add(candidate.resolve())
    try:
        check_destinations(
            resolved_inputs, (Path(p) for p in outputs if p is not None), force=force
        )
    except FileExistsError as exc:
        parser.fail(str(exc), hint="pass --force to replace it")
    except ValueError as exc:
        parser.error(str(exc))


def ancestor_paths(ancestors: tuple[Provenance, ...]) -> Iterator[Path]:
    """Identify caller-owned source files in a durable output's input history."""
    for node in ancestors:
        if node.operation == "open":
            yield Path(node.parameters["path"]).resolve()
            for field in ("flats", "darks"):
                if isinstance(path := node.parameters.get(field), str):
                    yield Path(path).resolve()
        yield from storage_paths(node.parameters)
        yield from ancestor_paths(node.inputs)


def storage_paths(value: object) -> Iterator[Path]:
    if isinstance(value, dict):
        if value.get("type") in {"path", "npy", "hdf5"}:
            yield Path(value["path"]).resolve()
        else:
            for option in value.values():
                yield from storage_paths(option)
    elif isinstance(value, tuple | list):
        for option in value:
            yield from storage_paths(option)


def check_destinations(inputs: set[Path], outputs: Iterable[Path], *, force: bool) -> None:
    """Check all caller-owned destinations against their inputs and each other."""
    from tomojax.errors import InvalidParameter

    seen: set[Path] = set()
    for path in outputs:
        resolved = path.resolve()
        if resolved in seen:
            raise InvalidParameter("path", f"duplicate output destination: {path}")
        seen.add(resolved)
        if any(src.is_relative_to(resolved) for src in inputs):
            raise InvalidParameter("path", f"output path must differ from input path: {path}")
        if not force and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"output already exists: {path}")


__all__ = [
    "EXPERT_EPILOG",
    "add_config",
    "add_force",
    "add_output",
    "check_paths",
    "config_settings",
    "hide_expert",
    "options",
]
