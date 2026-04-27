from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Mapping, TypeAlias, cast


RobustLossKind: TypeAlias = Literal[
    "charbonnier",
    "huber",
    "cauchy",
    "welsch",
    "student_t",
    "barron",
    "correntropy",
]
CorrelationLossKind: TypeAlias = Literal["zncc", "phasecorr", "fft_mag"]
GradientLossKind: TypeAlias = Literal["grad_l1", "ngf", "grad_orient", "chamfer_edge"]


@dataclass(frozen=True, slots=True)
class L2LossSpec:
    pass


@dataclass(frozen=True, slots=True)
class L2OtsuLossSpec:
    temp: float = 0.5


@dataclass(frozen=True, slots=True)
class PWLSLossSpec:
    a: float = 1.0
    b: float = 0.0


@dataclass(frozen=True, slots=True)
class EdgeL2LossSpec:
    pass


@dataclass(frozen=True, slots=True)
class RobustLossSpec:
    kind: RobustLossKind
    eps: float = 1e-3
    delta: float = 1.0
    c: float = 1.0
    nu: float = 4.0
    sigma: float = 1.0
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class CorrelationLossSpec:
    kind: CorrelationLossKind
    eps: float = 1e-5
    beta: float = 10.0


@dataclass(frozen=True, slots=True)
class SSIMLossSpec:
    multiscale: bool = False
    otsu_mask: bool = False
    K1: float = 0.01
    K2: float = 0.03
    window: int = 7
    levels: int = 3


@dataclass(frozen=True, slots=True)
class TverskyLossSpec:
    temp: float = 0.5
    alpha: float = 0.7
    beta: float = 0.3
    gamma: float = 1.0


@dataclass(frozen=True, slots=True)
class GradientLossSpec:
    kind: GradientLossKind
    eps: float = 1e-3


@dataclass(frozen=True, slots=True)
class InformationLossSpec:
    normalized: bool = False
    renyi_alpha: float | None = None
    bins: int = 32
    bw_x: float | None = None
    bw_y: float | None = None


@dataclass(frozen=True, slots=True)
class SWDLossSpec:
    n_samples: int = -1
    p: int = 1


@dataclass(frozen=True, slots=True)
class MindLossSpec:
    pass


@dataclass(frozen=True, slots=True)
class PoissonLossSpec:
    pass


AlignmentLossSpec: TypeAlias = (
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
    level_factor: int
    spec: AlignmentLossSpec


@dataclass(frozen=True, slots=True)
class AlignmentLossSchedule:
    default: AlignmentLossSpec
    by_level: tuple[LossScheduleEntry, ...]


AlignmentLossConfig: TypeAlias = AlignmentLossSpec | AlignmentLossSchedule

_LOSS_ALIASES: dict[str, str] = {
    "charb": "charbonnier",
    "lorentzian": "cauchy",
    "leclerc": "welsch",
    "ncc": "zncc",
    "ms-ssim": "ms_ssim",
    "msssim": "ms_ssim",
    "ms_ssim": "ms_ssim",
    "go": "grad_orient",
    "phase_corr_soft": "phasecorr",
    "fftmag": "fft_mag",
    "chamfer": "chamfer_edge",
    "gdl": "grad_l1",
    "poisson_nll": "poisson",
    "student-t": "student_t",
    "robust_general": "barron",
    "mcc": "correntropy",
    "mi_kde": "mi",
    "nmi_kde": "nmi",
    "tsallis_mi": "renyi_mi",
    "l2-otsu": "l2_otsu",
    "otsu-l2": "l2_otsu",
    "edge_aware_l2": "edge_l2",
    "sliced_wasserstein": "swd",
    "focal_tversky": "tversky",
}


def canonicalize_loss_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    return _LOSS_ALIASES.get(normalized, normalized)


def loss_spec_name(spec: AlignmentLossSpec) -> str:
    if isinstance(spec, L2LossSpec):
        return "l2"
    if isinstance(spec, L2OtsuLossSpec):
        return "l2_otsu"
    if isinstance(spec, PWLSLossSpec):
        return "pwls"
    if isinstance(spec, EdgeL2LossSpec):
        return "edge_l2"
    if isinstance(spec, RobustLossSpec):
        return spec.kind
    if isinstance(spec, CorrelationLossSpec):
        return spec.kind
    if isinstance(spec, SSIMLossSpec):
        if spec.otsu_mask:
            return "ssim_otsu"
        return "ms_ssim" if spec.multiscale else "ssim"
    if isinstance(spec, TverskyLossSpec):
        return "tversky"
    if isinstance(spec, GradientLossSpec):
        return spec.kind
    if isinstance(spec, InformationLossSpec):
        if spec.renyi_alpha is not None:
            return "renyi_mi"
        return "nmi" if spec.normalized else "mi"
    if isinstance(spec, SWDLossSpec):
        return "swd"
    if isinstance(spec, MindLossSpec):
        return "mind"
    if isinstance(spec, PoissonLossSpec):
        return "poisson"
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_spec_params(spec: AlignmentLossSpec) -> Dict[str, float]:
    if isinstance(spec, L2LossSpec | EdgeL2LossSpec | MindLossSpec | PoissonLossSpec):
        return {}
    if isinstance(spec, L2OtsuLossSpec):
        return {"temp": float(spec.temp)}
    if isinstance(spec, PWLSLossSpec):
        return {"a": float(spec.a), "b": float(spec.b)}
    if isinstance(spec, RobustLossSpec):
        params: Dict[str, float] = {}
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
    if isinstance(spec, CorrelationLossSpec):
        if spec.kind == "zncc":
            return {"eps": float(spec.eps)}
        if spec.kind == "phasecorr":
            return {"beta": float(spec.beta)}
        return {}
    if isinstance(spec, SSIMLossSpec):
        params = {
            "K1": float(spec.K1),
            "K2": float(spec.K2),
            "window": float(spec.window),
        }
        if spec.multiscale:
            params["levels"] = float(spec.levels)
        return params
    if isinstance(spec, TverskyLossSpec):
        return {
            "temp": float(spec.temp),
            "alpha": float(spec.alpha),
            "beta": float(spec.beta),
            "gamma": float(spec.gamma),
        }
    if isinstance(spec, GradientLossSpec):
        return {"eps": float(spec.eps)} if spec.kind in {"ngf", "grad_orient"} else {}
    if isinstance(spec, InformationLossSpec):
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
    if isinstance(spec, SWDLossSpec):
        return {"n_samples": float(spec.n_samples), "p": float(spec.p)}
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def parse_loss_spec(
    kind: str,
    params: Mapping[str, float] | None = None,
) -> AlignmentLossSpec:
    canonical = canonicalize_loss_kind(kind)
    raw = {} if params is None else {str(k): float(v) for k, v in params.items()}

    def _reject_extra_params(consumed: set[str]) -> None:
        extras = sorted(set(raw) - consumed)
        if extras:
            raise ValueError(f"Unsupported parameters for {canonical}: {', '.join(extras)}")

    if canonical == "l2":
        _reject_extra_params(set())
        return L2LossSpec()
    if canonical == "l2_otsu":
        _reject_extra_params({"temp"})
        return L2OtsuLossSpec(temp=float(raw.get("temp", 0.5)))
    if canonical == "pwls":
        _reject_extra_params({"a", "b"})
        return PWLSLossSpec(a=float(raw.get("a", 1.0)), b=float(raw.get("b", 0.0)))
    if canonical == "edge_l2":
        _reject_extra_params(set())
        return EdgeL2LossSpec()
    if canonical in {"charbonnier", "huber", "cauchy", "welsch", "student_t", "barron", "correntropy"}:
        _reject_extra_params({"eps", "delta", "c", "nu", "sigma", "alpha"})
        return RobustLossSpec(
            kind=cast(RobustLossKind, canonical),
            eps=float(raw.get("eps", 1e-3)),
            delta=float(raw.get("delta", 1.0)),
            c=float(raw.get("c", 1.0)),
            nu=float(raw.get("nu", 4.0)),
            sigma=float(raw.get("sigma", 1.0)),
            alpha=float(raw.get("alpha", 1.0)),
        )
    if canonical in {"zncc", "phasecorr", "fft_mag"}:
        _reject_extra_params({"eps", "beta"})
        return CorrelationLossSpec(
            kind=cast(CorrelationLossKind, canonical),
            eps=float(raw.get("eps", 1e-5)),
            beta=float(raw.get("beta", 10.0)),
        )
    if canonical == "ssim":
        _reject_extra_params({"K1", "K2", "window"})
        return SSIMLossSpec(
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
        )
    if canonical == "ms_ssim":
        _reject_extra_params({"K1", "K2", "window", "levels"})
        return SSIMLossSpec(
            multiscale=True,
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
            levels=int(raw.get("levels", 3)),
        )
    if canonical == "ssim_otsu":
        _reject_extra_params({"K1", "K2", "window"})
        return SSIMLossSpec(
            otsu_mask=True,
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
        )
    if canonical == "tversky":
        _reject_extra_params({"temp", "alpha", "beta", "gamma"})
        return TverskyLossSpec(
            temp=float(raw.get("temp", 0.5)),
            alpha=float(raw.get("alpha", 0.7)),
            beta=float(raw.get("beta", 0.3)),
            gamma=float(raw.get("gamma", 1.0)),
        )
    if canonical in {"grad_l1", "ngf", "grad_orient", "chamfer_edge"}:
        _reject_extra_params({"eps"})
        return GradientLossSpec(
            kind=cast(GradientLossKind, canonical),
            eps=float(raw.get("eps", 1e-3)),
        )
    if canonical in {"mi", "nmi", "renyi_mi"}:
        _reject_extra_params({"bins", "bw_x", "bw_y", "alpha"})
        return InformationLossSpec(
            normalized=(canonical == "nmi"),
            renyi_alpha=(float(raw.get("alpha", 1.5)) if canonical == "renyi_mi" else None),
            bins=int(raw.get("bins", 32)),
            bw_x=raw.get("bw_x"),
            bw_y=raw.get("bw_y"),
        )
    if canonical == "swd":
        _reject_extra_params({"n_samples", "p"})
        return SWDLossSpec(
            n_samples=int(raw.get("n_samples", -1)),
            p=int(raw.get("p", 1)),
        )
    if canonical == "mind":
        _reject_extra_params(set())
        return MindLossSpec()
    if canonical == "poisson":
        _reject_extra_params(set())
        return PoissonLossSpec()
    raise ValueError(f"Unknown loss kind: {kind}")


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
        raise ValueError(
            f"Invalid loss schedule entry for level {level}: {exc}"
        ) from exc


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
                raise ValueError(
                    f"Loss schedule entry {item!r} must use LEVEL:LOSS format"
                )
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


def loss_is_within_relative_tolerance(loss_before: float, loss_after: float, rel_tol: float) -> bool:
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
    "parse_loss_schedule",
    "parse_loss_spec",
    "resolve_loss_for_level",
    "validate_loss_schedule_levels",
]
