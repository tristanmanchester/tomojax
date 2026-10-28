"""Typed alignment loss specifications and parsers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class L2LossSpec:
    """Plain squared-error projection loss."""


@dataclass(frozen=True, slots=True)
class L2OtsuLossSpec:
    """Squared-error loss weighted by a soft Otsu foreground mask.

    Parameters
    ----------
    temp : float, optional
        Soft-mask sigmoid temperature; smaller values sharpen the foreground weighting.
        Default is 0.5.
    """

    temp: float = 0.5


@dataclass(frozen=True, slots=True)
class HuberLossSpec:
    """Convex robust projection loss for fixed-volume Gauss-Newton alignment.

    Parameters
    ----------
    delta : float, optional
        Positive residual threshold in projection units. Residuals above this
        threshold have bounded influence. Default is 1.0; choose it for the
        noise scale of the measured projections. Joint GN is not supported.
    """

    delta: float = 1.0


type AlignmentLossSpec = L2LossSpec | L2OtsuLossSpec | HuberLossSpec


@dataclass(frozen=True, slots=True)
class LossScheduleEntry:
    """One level-specific loss override.

    Parameters
    ----------
    level_factor : int
        Current coarsening factor relative to the native grid and detector.
    spec : AlignmentLossSpec
        Loss specification for this level.
    """

    level_factor: int
    spec: AlignmentLossSpec


@dataclass(frozen=True, slots=True)
class AlignmentLossSchedule:
    """Default loss plus optional per-pyramid-level overrides.

    Parameters
    ----------
    default : AlignmentLossSpec
        Loss used at levels without an explicit override.
    by_level : tuple[LossScheduleEntry, ...]
        Level-specific loss overrides.
    """

    default: AlignmentLossSpec
    by_level: tuple[LossScheduleEntry, ...]


type AlignmentLossConfig = AlignmentLossSpec | AlignmentLossSchedule


def canonicalize_loss_kind(kind: str) -> str:
    """Normalize a user-facing loss kind."""
    return str(kind).strip().lower()


def loss_spec_name(spec: AlignmentLossSpec) -> str:
    """Return the canonical loss name for a loss spec.

    Parameters
    ----------
    spec : AlignmentLossSpec
        Typed projection-domain loss specification.

    Returns
    -------
    str
        Canonical registered name of the loss.

    Raises
    ------
    TypeError
        If the object is not a supported loss specification.
    """
    if isinstance(spec, L2LossSpec):
        return "l2"
    if isinstance(spec, L2OtsuLossSpec):
        return "l2_otsu"
    if isinstance(spec, HuberLossSpec):
        return "huber"
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_spec_params(spec: AlignmentLossSpec) -> dict[str, float]:
    """Return JSON-compatible scalar parameters for a loss spec.

    Parameters
    ----------
    spec : AlignmentLossSpec
        Typed projection-domain loss specification.

    Returns
    -------
    dict
        Scalar parameters in JSON-compatible form.

    Raises
    ------
    TypeError
        If the object is not a supported loss specification.
    """
    if isinstance(spec, L2LossSpec):
        return {}
    if isinstance(spec, L2OtsuLossSpec):
        return {"temp": float(spec.temp)}
    if isinstance(spec, HuberLossSpec):
        return {"delta": float(spec.delta)}
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_spec_supports_setup_validation_lm(spec: AlignmentLossSpec) -> bool:
    """Return whether a loss can provide validation-LM residual weights."""
    return isinstance(spec, L2LossSpec | L2OtsuLossSpec)


def parse_loss_spec(
    kind: str,
    params: Mapping[str, float] | None = None,
) -> AlignmentLossSpec:
    """Parse a named loss and scalar parameters into a typed spec.

    Parameters
    ----------
    kind : str
        Registered loss name, such as "l2", "l2_otsu" or "huber".
    params : mapping or None, optional
        Scalar parameters for the named loss; None uses its defaults.

    Returns
    -------
    AlignmentLossSpec
        Typed specification for the requested loss.

    Raises
    ------
    ValueError
        If the loss name or one of its parameter names is unknown.
    """
    canonical = canonicalize_loss_kind(kind)
    raw = {} if params is None else {str(k): float(v) for k, v in params.items()}
    allowed = {"l2": (), "l2_otsu": ("temp",), "huber": ("delta",)}
    if canonical not in allowed:
        raise ValueError(f"Unknown or removed loss kind: {kind}; use l2, l2_otsu or huber")
    extras = sorted(set(raw) - set(allowed[canonical]))
    if extras:
        raise ValueError(f"Unsupported parameters for {canonical}: {', '.join(extras)}")
    if canonical == "l2":
        return L2LossSpec()
    if canonical == "l2_otsu":
        return L2OtsuLossSpec(temp=raw.get("temp", 0.5))
    return HuberLossSpec(delta=raw.get("delta", 1.0))


def _parse_loss_schedule_level(raw_level: object) -> int:
    level_text = str(raw_level).strip()
    if not level_text:
        raise ValueError("Loss schedule level must not be empty")
    try:
        level = int(level_text)
    except ValueError as exc:
        raise ValueError(f"Loss schedule level must be a positive integer: {raw_level!r}") from exc
    if level < 1:
        raise ValueError(f"Loss schedule level must be a positive integer: {raw_level!r}")
    return level


def _parse_loss_schedule_loss(raw_loss: object, *, level: int) -> AlignmentLossSpec:
    loss_name = str(raw_loss).strip()
    if not loss_name:
        raise ValueError(f"Loss schedule entry for level {level} must name a loss")
    try:
        return parse_loss_spec(loss_name)
    except ValueError as exc:
        raise ValueError(f"Invalid loss schedule entry for level {level}: {exc}") from exc


def parse_loss_schedule(
    value: str | Mapping[object, object],
    default: AlignmentLossSpec,
) -> AlignmentLossSchedule:
    """Parse a pyramid-level loss schedule.

    String values use ``LEVEL:LOSS`` comma-separated entries, for example
    ``"4:huber,2:l2,1:l2_otsu"``. Mapping values are intended for TOML
    config files, for example ``{"4": "huber", "2": "l2"}``.

    Parameters
    ----------
    value : str or mapping
        Comma-separated LEVEL:LOSS entries or a mapping from levels to loss
        specifications.
    default : AlignmentLossSpec
        Loss for pyramid levels without an explicit override.

    Returns
    -------
    AlignmentLossSchedule
        Default and level-specific loss specifications.

    Raises
    ------
    ValueError
        If the schedule syntax, factors or loss specifications are invalid.
    TypeError
        If value is not a string or mapping.
    """
    raw_entries: list[tuple[object, object]] = []
    if isinstance(value, Mapping):
        raw_entries = list(value.items())
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Loss schedule must not be empty")
        for entry in text.split(","):
            item = entry.strip()
            if not item:
                raise ValueError("Loss schedule contains an empty entry")
            if ":" not in item:
                raise ValueError(f"Loss schedule entry {item!r} must use LEVEL:LOSS format")
            raw_level, raw_loss = item.split(":", 1)
            raw_entries.append((raw_level, raw_loss))
    else:
        raise TypeError("Loss schedule must be a string or mapping")

    if not raw_entries:
        raise ValueError("Loss schedule must contain at least one entry")

    entries: list[LossScheduleEntry] = []
    seen: set[int] = set()
    for raw_level, raw_loss in raw_entries:
        level = _parse_loss_schedule_level(raw_level)
        if level in seen:
            raise ValueError(f"Duplicate loss schedule level: {level}")
        seen.add(level)
        entries.append(
            LossScheduleEntry(
                level_factor=level,
                spec=_parse_loss_schedule_loss(raw_loss, level=level),
            )
        )

    return AlignmentLossSchedule(
        default=default,
        by_level=tuple(sorted(entries, key=lambda entry: entry.level_factor)),
    )


def resolve_loss_for_level(
    loss_config: AlignmentLossConfig,
    level_factor: int,
) -> AlignmentLossSpec:
    """Resolve the loss active at a pyramid level.

    Parameters
    ----------
    loss_config : AlignmentLossConfig
        Loss specification or pyramid-level loss schedule.
    level_factor : int
        Current pyramid coarsening factor.

    Returns
    -------
    AlignmentLossSpec
        Level override when present, otherwise the default loss.
    """
    if not isinstance(loss_config, AlignmentLossSchedule):
        return loss_config
    level = int(level_factor)
    for entry in loss_config.by_level:
        if int(entry.level_factor) == level:
            return entry.spec
    return loss_config.default


def validate_loss_schedule_levels(
    loss_config: AlignmentLossConfig,
    factors: Iterable[int],
) -> None:
    """Validate that scheduled loss levels exist in the configured pyramid.

    Parameters
    ----------
    loss_config : AlignmentLossConfig
        Loss specification or pyramid-level loss schedule.
    factors : iterable of int
        Coarsening factors present in the configured pyramid.

    Raises
    ------
    ValueError
        If a loss override names a level outside the pyramid.
    """
    if not isinstance(loss_config, AlignmentLossSchedule):
        return
    factor_list = [int(factor) for factor in factors]
    valid = set(factor_list)
    missing = sorted(
        int(entry.level_factor)
        for entry in loss_config.by_level
        if int(entry.level_factor) not in valid
    )
    if not missing:
        return
    valid_text = ", ".join(str(factor) for factor in factor_list)
    missing_text = ", ".join(str(factor) for factor in missing)
    raise ValueError(
        f"Loss schedule level(s) {missing_text} do not match configured levels: {valid_text}"
    )


def loss_is_within_relative_tolerance(
    loss_before: float, loss_after: float, rel_tol: float
) -> bool:
    """Return True when ``loss_after`` stays within a relative tolerance of ``loss_before``."""
    before = float(loss_before)
    after = float(loss_after)
    tol = max(float(rel_tol), 0.0) * abs(before)
    return after < before + tol


__all__ = [
    "AlignmentLossConfig",
    "AlignmentLossSchedule",
    "AlignmentLossSpec",
    "HuberLossSpec",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "LossScheduleEntry",
    "canonicalize_loss_kind",
    "loss_is_within_relative_tolerance",
    "loss_spec_name",
    "loss_spec_params",
    "loss_spec_supports_setup_validation_lm",
    "parse_loss_schedule",
    "parse_loss_spec",
    "resolve_loss_for_level",
    "validate_loss_schedule_levels",
]
