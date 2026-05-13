"""Typed alignment loss specifications and parsers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

type RobustLossKind = Literal[
    "charbonnier",
    "huber",
    "cauchy",
    "welsch",
    "student_t",
    "barron",
    "correntropy",
]
type CorrelationLossKind = Literal["zncc", "phasecorr", "fft_mag"]
type GradientLossKind = Literal["grad_l1", "ngf", "grad_orient", "chamfer_edge"]


@dataclass(frozen=True, slots=True)
class L2LossSpec:
    """Plain squared-error projection loss."""


@dataclass(frozen=True, slots=True)
class L2OtsuLossSpec:
    """Squared-error loss weighted by a soft Otsu foreground mask."""

    temp: float = 0.5


@dataclass(frozen=True, slots=True)
class PWLSLossSpec:
    """Poisson-weighted least-squares projection loss."""

    a: float = 1.0
    b: float = 0.0


@dataclass(frozen=True, slots=True)
class EdgeL2LossSpec:
    """Squared-error loss with target-edge-aware weights."""


@dataclass(frozen=True, slots=True)
class RobustLossSpec:
    """Robust projection-domain loss family."""

    kind: RobustLossKind
    eps: float = 1e-3
    delta: float = 1.0
    c: float = 1.0
    nu: float = 4.0
    sigma: float = 1.0
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class CorrelationLossSpec:
    """Correlation-style projection loss family."""

    kind: CorrelationLossKind
    eps: float = 1e-5
    beta: float = 10.0


@dataclass(frozen=True, slots=True)
class SSIMLossSpec:
    """Structural-similarity projection loss."""

    multiscale: bool = False
    otsu_mask: bool = False
    K1: float = 0.01
    K2: float = 0.03
    window: int = 7
    levels: int = 3


@dataclass(frozen=True, slots=True)
class TverskyLossSpec:
    """Soft foreground-overlap loss based on the Tversky index."""

    temp: float = 0.5
    alpha: float = 0.7
    beta: float = 0.3
    gamma: float = 1.0


@dataclass(frozen=True, slots=True)
class GradientLossSpec:
    """Gradient-domain projection loss family."""

    kind: GradientLossKind
    eps: float = 1e-3


@dataclass(frozen=True, slots=True)
class InformationLossSpec:
    """Mutual-information projection loss family."""

    normalized: bool = False
    renyi_alpha: float | None = None
    bins: int = 32
    bw_x: float | None = None
    bw_y: float | None = None


@dataclass(frozen=True, slots=True)
class SWDLossSpec:
    """Sliced-Wasserstein projection loss."""

    n_samples: int = -1
    p: int = 1


@dataclass(frozen=True, slots=True)
class MindLossSpec:
    """MIND descriptor projection loss."""


@dataclass(frozen=True, slots=True)
class PoissonLossSpec:
    """Poisson negative-log-likelihood projection loss."""


type AlignmentLossSpec = (
    L2LossSpec
    | L2OtsuLossSpec
    | PWLSLossSpec
    | EdgeL2LossSpec
    | RobustLossSpec
    | CorrelationLossSpec
    | SSIMLossSpec
    | TverskyLossSpec
    | GradientLossSpec
    | InformationLossSpec
    | SWDLossSpec
    | MindLossSpec
    | PoissonLossSpec
)


@dataclass(frozen=True, slots=True)
class LossScheduleEntry:
    """One level-specific loss override."""

    level_factor: int
    spec: AlignmentLossSpec


@dataclass(frozen=True, slots=True)
class AlignmentLossSchedule:
    """Default loss plus optional per-pyramid-level overrides."""

    default: AlignmentLossSpec
    by_level: tuple[LossScheduleEntry, ...]


type AlignmentLossConfig = AlignmentLossSpec | AlignmentLossSchedule

type LossBuilder = Callable[[Mapping[str, float]], AlignmentLossSpec]
type LossMatcher = Callable[[AlignmentLossSpec], bool]
type LossNameEmitter = Callable[[AlignmentLossSpec], str]
type LossParamEmitter = Callable[[AlignmentLossSpec], dict[str, float]]


@dataclass(frozen=True, slots=True)
class LossKindEntry:
    build: LossBuilder
    allowed_params: frozenset[str]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LossSpecDescriptor:
    matches: LossMatcher
    name: LossNameEmitter
    params: LossParamEmitter


def _empty_params(_: AlignmentLossSpec) -> dict[str, float]:
    return {}


def _build_l2(_: Mapping[str, float]) -> AlignmentLossSpec:
    return L2LossSpec()


def _build_l2_otsu(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return L2OtsuLossSpec(temp=float(raw.get("temp", 0.5)))


def _build_pwls(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return PWLSLossSpec(a=float(raw.get("a", 1.0)), b=float(raw.get("b", 0.0)))


def _build_edge_l2(_: Mapping[str, float]) -> AlignmentLossSpec:
    return EdgeL2LossSpec()


def _robust_builder(kind: RobustLossKind) -> LossBuilder:
    def _build(raw: Mapping[str, float]) -> AlignmentLossSpec:
        return RobustLossSpec(
            kind=kind,
            eps=float(raw.get("eps", 1e-3)),
            delta=float(raw.get("delta", 1.0)),
            c=float(raw.get("c", 1.0)),
            nu=float(raw.get("nu", 4.0)),
            sigma=float(raw.get("sigma", 1.0)),
            alpha=float(raw.get("alpha", 1.0)),
        )

    return _build


def _correlation_builder(kind: CorrelationLossKind) -> LossBuilder:
    def _build(raw: Mapping[str, float]) -> AlignmentLossSpec:
        return CorrelationLossSpec(
            kind=kind,
            eps=float(raw.get("eps", 1e-5)),
            beta=float(raw.get("beta", 10.0)),
        )

    return _build


def _build_ssim(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return SSIMLossSpec(
        K1=float(raw.get("K1", 0.01)),
        K2=float(raw.get("K2", 0.03)),
        window=int(raw.get("window", 7)),
    )


def _build_ms_ssim(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return SSIMLossSpec(
        multiscale=True,
        K1=float(raw.get("K1", 0.01)),
        K2=float(raw.get("K2", 0.03)),
        window=int(raw.get("window", 7)),
        levels=int(raw.get("levels", 3)),
    )


def _build_ssim_otsu(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return SSIMLossSpec(
        otsu_mask=True,
        K1=float(raw.get("K1", 0.01)),
        K2=float(raw.get("K2", 0.03)),
        window=int(raw.get("window", 7)),
    )


def _build_tversky(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return TverskyLossSpec(
        temp=float(raw.get("temp", 0.5)),
        alpha=float(raw.get("alpha", 0.7)),
        beta=float(raw.get("beta", 0.3)),
        gamma=float(raw.get("gamma", 1.0)),
    )


def _gradient_builder(kind: GradientLossKind) -> LossBuilder:
    def _build(raw: Mapping[str, float]) -> AlignmentLossSpec:
        return GradientLossSpec(kind=kind, eps=float(raw.get("eps", 1e-3)))

    return _build


def _build_mi(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return InformationLossSpec(
        bins=int(raw.get("bins", 32)),
        bw_x=raw.get("bw_x"),
        bw_y=raw.get("bw_y"),
    )


def _build_nmi(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return InformationLossSpec(
        normalized=True,
        bins=int(raw.get("bins", 32)),
        bw_x=raw.get("bw_x"),
        bw_y=raw.get("bw_y"),
    )


def _build_renyi_mi(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return InformationLossSpec(
        renyi_alpha=float(raw.get("alpha", 1.5)),
        bins=int(raw.get("bins", 32)),
        bw_x=raw.get("bw_x"),
        bw_y=raw.get("bw_y"),
    )


def _build_swd(raw: Mapping[str, float]) -> AlignmentLossSpec:
    return SWDLossSpec(
        n_samples=int(raw.get("n_samples", -1)),
        p=int(raw.get("p", 1)),
    )


def _build_mind(_: Mapping[str, float]) -> AlignmentLossSpec:
    return MindLossSpec()


def _build_poisson(_: Mapping[str, float]) -> AlignmentLossSpec:
    return PoissonLossSpec()


_LOSS_KIND_REGISTRY: dict[str, LossKindEntry] = {
    "l2": LossKindEntry(_build_l2, frozenset()),
    "l2_otsu": LossKindEntry(_build_l2_otsu, frozenset({"temp"}), aliases=("l2-otsu", "otsu-l2")),
    "pwls": LossKindEntry(_build_pwls, frozenset({"a", "b"})),
    "edge_l2": LossKindEntry(_build_edge_l2, frozenset(), aliases=("edge_aware_l2",)),
    "charbonnier": LossKindEntry(
        _robust_builder("charbonnier"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("charb",),
    ),
    "huber": LossKindEntry(
        _robust_builder("huber"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
    ),
    "cauchy": LossKindEntry(
        _robust_builder("cauchy"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("lorentzian",),
    ),
    "welsch": LossKindEntry(
        _robust_builder("welsch"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("leclerc",),
    ),
    "student_t": LossKindEntry(
        _robust_builder("student_t"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("student-t",),
    ),
    "barron": LossKindEntry(
        _robust_builder("barron"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("robust_general",),
    ),
    "correntropy": LossKindEntry(
        _robust_builder("correntropy"),
        frozenset({"eps", "delta", "c", "nu", "sigma", "alpha"}),
        aliases=("mcc",),
    ),
    "zncc": LossKindEntry(
        _correlation_builder("zncc"), frozenset({"eps", "beta"}), aliases=("ncc",)
    ),
    "phasecorr": LossKindEntry(
        _correlation_builder("phasecorr"),
        frozenset({"eps", "beta"}),
        aliases=("phase_corr_soft",),
    ),
    "fft_mag": LossKindEntry(
        _correlation_builder("fft_mag"),
        frozenset({"eps", "beta"}),
        aliases=("fftmag",),
    ),
    "ssim": LossKindEntry(_build_ssim, frozenset({"K1", "K2", "window"})),
    "ms_ssim": LossKindEntry(
        _build_ms_ssim,
        frozenset({"K1", "K2", "window", "levels"}),
        aliases=("ms-ssim", "msssim"),
    ),
    "ssim_otsu": LossKindEntry(_build_ssim_otsu, frozenset({"K1", "K2", "window"})),
    "tversky": LossKindEntry(
        _build_tversky,
        frozenset({"temp", "alpha", "beta", "gamma"}),
        aliases=("focal_tversky",),
    ),
    "grad_l1": LossKindEntry(_gradient_builder("grad_l1"), frozenset({"eps"}), aliases=("gdl",)),
    "ngf": LossKindEntry(_gradient_builder("ngf"), frozenset({"eps"})),
    "grad_orient": LossKindEntry(
        _gradient_builder("grad_orient"), frozenset({"eps"}), aliases=("go",)
    ),
    "chamfer_edge": LossKindEntry(
        _gradient_builder("chamfer_edge"),
        frozenset({"eps"}),
        aliases=("chamfer",),
    ),
    "mi": LossKindEntry(
        _build_mi, frozenset({"bins", "bw_x", "bw_y", "alpha"}), aliases=("mi_kde",)
    ),
    "nmi": LossKindEntry(
        _build_nmi, frozenset({"bins", "bw_x", "bw_y", "alpha"}), aliases=("nmi_kde",)
    ),
    "renyi_mi": LossKindEntry(
        _build_renyi_mi,
        frozenset({"bins", "bw_x", "bw_y", "alpha"}),
        aliases=("tsallis_mi",),
    ),
    "swd": LossKindEntry(
        _build_swd, frozenset({"n_samples", "p"}), aliases=("sliced_wasserstein",)
    ),
    "mind": LossKindEntry(_build_mind, frozenset()),
    "poisson": LossKindEntry(_build_poisson, frozenset(), aliases=("poisson_nll",)),
}

_LOSS_ALIASES: dict[str, str] = {
    alias: canonical for canonical, entry in _LOSS_KIND_REGISTRY.items() for alias in entry.aliases
}
_SETUP_VALIDATION_LM_LOSSES = frozenset({"l2", "l2_otsu", "pwls", "edge_l2"})


def _robust_params(spec: RobustLossSpec) -> dict[str, float]:
    params: dict[str, float] = {}
    if spec.kind == "charbonnier":
        params["eps"] = float(spec.eps)
    elif spec.kind == "huber":
        params["delta"] = float(spec.delta)
    elif spec.kind in {"cauchy", "welsch"}:
        params["c"] = float(spec.c)
    elif spec.kind == "student_t":
        params["nu"] = float(spec.nu)
        params["sigma"] = float(spec.sigma)
    elif spec.kind == "barron":
        params["alpha"] = float(spec.alpha)
        params["c"] = float(spec.c)
    elif spec.kind == "correntropy":
        params["sigma"] = float(spec.sigma)
    return params


def _correlation_params(spec: CorrelationLossSpec) -> dict[str, float]:
    if spec.kind == "zncc":
        return {"eps": float(spec.eps)}
    if spec.kind == "phasecorr":
        return {"beta": float(spec.beta)}
    return {}


def _ssim_name(spec: SSIMLossSpec) -> str:
    if spec.otsu_mask:
        return "ssim_otsu"
    return "ms_ssim" if spec.multiscale else "ssim"


def _ssim_params(spec: SSIMLossSpec) -> dict[str, float]:
    params = {
        "K1": float(spec.K1),
        "K2": float(spec.K2),
        "window": float(spec.window),
    }
    if spec.multiscale:
        params["levels"] = float(spec.levels)
    return params


def _tversky_params(spec: TverskyLossSpec) -> dict[str, float]:
    return {
        "temp": float(spec.temp),
        "alpha": float(spec.alpha),
        "beta": float(spec.beta),
        "gamma": float(spec.gamma),
    }


def _gradient_params(spec: GradientLossSpec) -> dict[str, float]:
    return {"eps": float(spec.eps)} if spec.kind in {"ngf", "grad_orient"} else {}


def _information_name(spec: InformationLossSpec) -> str:
    if spec.renyi_alpha is not None:
        return "renyi_mi"
    return "nmi" if spec.normalized else "mi"


def _information_params(spec: InformationLossSpec) -> dict[str, float]:
    params = {"bins": float(spec.bins)}
    if spec.bw_x is not None:
        params["bw_x"] = float(spec.bw_x)
    if spec.bw_y is not None:
        params["bw_y"] = float(spec.bw_y)
    if spec.normalized:
        params["nmi"] = 1.0
    if spec.renyi_alpha is not None:
        params["alpha"] = float(spec.renyi_alpha)
    return params


_LOSS_SPEC_DESCRIPTORS: tuple[LossSpecDescriptor, ...] = (
    LossSpecDescriptor(lambda spec: isinstance(spec, L2LossSpec), lambda _: "l2", _empty_params),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, L2OtsuLossSpec),
        lambda _: "l2_otsu",
        lambda spec: {"temp": float(cast("L2OtsuLossSpec", spec).temp)},
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, PWLSLossSpec),
        lambda _: "pwls",
        lambda spec: {
            "a": float(cast("PWLSLossSpec", spec).a),
            "b": float(cast("PWLSLossSpec", spec).b),
        },
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, EdgeL2LossSpec), lambda _: "edge_l2", _empty_params
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, RobustLossSpec),
        lambda spec: cast("RobustLossSpec", spec).kind,
        lambda spec: _robust_params(cast("RobustLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, CorrelationLossSpec),
        lambda spec: cast("CorrelationLossSpec", spec).kind,
        lambda spec: _correlation_params(cast("CorrelationLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, SSIMLossSpec),
        lambda spec: _ssim_name(cast("SSIMLossSpec", spec)),
        lambda spec: _ssim_params(cast("SSIMLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, TverskyLossSpec),
        lambda _: "tversky",
        lambda spec: _tversky_params(cast("TverskyLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, GradientLossSpec),
        lambda spec: cast("GradientLossSpec", spec).kind,
        lambda spec: _gradient_params(cast("GradientLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, InformationLossSpec),
        lambda spec: _information_name(cast("InformationLossSpec", spec)),
        lambda spec: _information_params(cast("InformationLossSpec", spec)),
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, SWDLossSpec),
        lambda _: "swd",
        lambda spec: {
            "n_samples": float(cast("SWDLossSpec", spec).n_samples),
            "p": float(cast("SWDLossSpec", spec).p),
        },
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, MindLossSpec), lambda _: "mind", _empty_params
    ),
    LossSpecDescriptor(
        lambda spec: isinstance(spec, PoissonLossSpec), lambda _: "poisson", _empty_params
    ),
)


def canonicalize_loss_kind(kind: str) -> str:
    """Normalize a user-facing loss kind or alias."""
    normalized = str(kind).strip().lower()
    return _LOSS_ALIASES.get(normalized, normalized)


def loss_spec_name(spec: AlignmentLossSpec) -> str:
    """Return the canonical loss name for a loss spec."""
    for descriptor in _LOSS_SPEC_DESCRIPTORS:
        if descriptor.matches(spec):
            return descriptor.name(spec)
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_spec_params(spec: AlignmentLossSpec) -> dict[str, float]:
    """Return JSON-compatible scalar parameters for a loss spec."""
    for descriptor in _LOSS_SPEC_DESCRIPTORS:
        if descriptor.matches(spec):
            return descriptor.params(spec)
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_spec_supports_setup_validation_lm(spec: AlignmentLossSpec) -> bool:
    """Return whether a loss can provide validation-LM residual weights."""
    return loss_spec_name(spec) in _SETUP_VALIDATION_LM_LOSSES


def parse_loss_spec(
    kind: str,
    params: Mapping[str, float] | None = None,
) -> AlignmentLossSpec:
    """Parse a named loss and scalar parameters into a typed spec."""
    canonical = canonicalize_loss_kind(kind)
    raw = {} if params is None else {str(k): float(v) for k, v in params.items()}
    entry = _LOSS_KIND_REGISTRY.get(canonical)
    if entry is None:
        raise ValueError(f"Unknown loss kind: {kind}")
    extras = sorted(set(raw) - set(entry.allowed_params))
    if extras:
        raise ValueError(f"Unsupported parameters for {canonical}: {', '.join(extras)}")
    return entry.build(raw)


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
    ``"4:phasecorr,2:ssim,1:l2_otsu"``. Mapping values are intended for TOML
    config files, for example ``{"4": "phasecorr", "2": "ssim"}``.
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
    """Resolve the loss active at a pyramid level."""
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
    """Validate that scheduled loss levels exist in the configured pyramid."""
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
    "CorrelationLossSpec",
    "EdgeL2LossSpec",
    "GradientLossSpec",
    "InformationLossSpec",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "LossScheduleEntry",
    "MindLossSpec",
    "PWLSLossSpec",
    "PoissonLossSpec",
    "RobustLossSpec",
    "SSIMLossSpec",
    "SWDLossSpec",
    "TverskyLossSpec",
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
