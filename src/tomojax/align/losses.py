from __future__ import annotations

"""Differentiable alignment loss functions (per-view and some precomputed).

All losses operate on per-view projections `pred` and `tar` (nv, nu) and return a
scalar loss. Heavy precomputations that only depend on the targets happen once in
`build_loss` and are stored in `LossState` (kept on device where useful).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Mapping, Optional, TypeAlias, Tuple

import jax
import jax.numpy as jnp
import numpy as np
try:  # optional, used for Chamfer loss
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover
    distance_transform_edt = None  # type: ignore


@dataclass
class LossState:
    kind: str
    params: Dict[str, float]
    # Optional per-view mask (n, nv, nu) for masked/ROI losses
    mask: Optional[jnp.ndarray] = None
    # Optional per-view precomputes
    bins_x: Optional[jnp.ndarray] = None
    bins_y: Optional[jnp.ndarray] = None
    bw_x: Optional[float] = None
    bw_y: Optional[float] = None
    dt_edge: Optional[jnp.ndarray] = None
    thr: Optional[jnp.ndarray] = None  # per-view scalar thresholds broadcastable


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
    kind: Literal[
        "charbonnier",
        "huber",
        "cauchy",
        "welsch",
        "student_t",
        "barron",
        "correntropy",
    ]
    eps: float = 1e-3
    delta: float = 1.0
    c: float = 1.0
    nu: float = 4.0
    sigma: float = 1.0
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class CorrelationLossSpec:
    kind: Literal["zncc", "phasecorr", "fft_mag"]
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
    kind: Literal["grad_l1", "ngf", "grad_orient", "chamfer_edge"]
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

PerViewLossFn: TypeAlias = Callable[
    [jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]],
    jnp.ndarray,
]
GaussNewtonWeightFn: TypeAlias = Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]


@dataclass(frozen=True, slots=True)
class LossAdapter:
    spec: AlignmentLossSpec
    name: str
    state: LossState
    per_view_loss: PerViewLossFn
    supports_gauss_newton: bool
    gauss_newton_weights: GaussNewtonWeightFn


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

    def _unused(consumed: set[str]) -> None:
        extras = sorted(set(raw) - consumed)
        if extras:
            raise ValueError(f"Unsupported parameters for {canonical}: {', '.join(extras)}")

    if canonical == "l2":
        _unused(set())
        return L2LossSpec()
    if canonical == "l2_otsu":
        _unused({"temp"})
        return L2OtsuLossSpec(temp=float(raw.get("temp", 0.5)))
    if canonical == "pwls":
        _unused({"a", "b"})
        return PWLSLossSpec(a=float(raw.get("a", 1.0)), b=float(raw.get("b", 0.0)))
    if canonical == "edge_l2":
        _unused(set())
        return EdgeL2LossSpec()
    if canonical in {"charbonnier", "huber", "cauchy", "welsch", "student_t", "barron", "correntropy"}:
        _unused({"eps", "delta", "c", "nu", "sigma", "alpha"})
        return RobustLossSpec(
            kind=canonical,  # type: ignore[arg-type]
            eps=float(raw.get("eps", 1e-3)),
            delta=float(raw.get("delta", 1.0)),
            c=float(raw.get("c", 1.0)),
            nu=float(raw.get("nu", 4.0)),
            sigma=float(raw.get("sigma", 1.0)),
            alpha=float(raw.get("alpha", 1.0)),
        )
    if canonical in {"zncc", "phasecorr", "fft_mag"}:
        _unused({"eps", "beta"})
        return CorrelationLossSpec(
            kind=canonical,  # type: ignore[arg-type]
            eps=float(raw.get("eps", 1e-5)),
            beta=float(raw.get("beta", 10.0)),
        )
    if canonical == "ssim":
        _unused({"K1", "K2", "window"})
        return SSIMLossSpec(
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
        )
    if canonical == "ms_ssim":
        _unused({"K1", "K2", "window", "levels"})
        return SSIMLossSpec(
            multiscale=True,
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
            levels=int(raw.get("levels", 3)),
        )
    if canonical == "ssim_otsu":
        _unused({"K1", "K2", "window"})
        return SSIMLossSpec(
            otsu_mask=True,
            K1=float(raw.get("K1", 0.01)),
            K2=float(raw.get("K2", 0.03)),
            window=int(raw.get("window", 7)),
        )
    if canonical == "tversky":
        _unused({"temp", "alpha", "beta", "gamma"})
        return TverskyLossSpec(
            temp=float(raw.get("temp", 0.5)),
            alpha=float(raw.get("alpha", 0.7)),
            beta=float(raw.get("beta", 0.3)),
            gamma=float(raw.get("gamma", 1.0)),
        )
    if canonical in {"grad_l1", "ngf", "grad_orient", "chamfer_edge"}:
        _unused({"eps"})
        return GradientLossSpec(
            kind=canonical,  # type: ignore[arg-type]
            eps=float(raw.get("eps", 1e-3)),
        )
    if canonical in {"mi", "nmi", "renyi_mi"}:
        _unused({"bins", "bw_x", "bw_y", "alpha"})
        return InformationLossSpec(
            normalized=(canonical == "nmi"),
            renyi_alpha=(float(raw.get("alpha", 1.5)) if canonical == "renyi_mi" else None),
            bins=int(raw.get("bins", 32)),
            bw_x=raw.get("bw_x"),
            bw_y=raw.get("bw_y"),
        )
    if canonical == "swd":
        _unused({"n_samples", "p"})
        return SWDLossSpec(
            n_samples=int(raw.get("n_samples", -1)),
            p=int(raw.get("p", 1)),
        )
    if canonical == "mind":
        _unused(set())
        return MindLossSpec()
    if canonical == "poisson":
        _unused(set())
        return PoissonLossSpec()
    raise ValueError(f"Unknown loss kind: {kind}")


def _safe_epsilon(p: Dict[str, float], key: str, default: float) -> float:
    v = float(p.get(key, default))
    return max(v, 1e-12)


def _validated_renyi_alpha(params: Dict[str, float]) -> float:
    a = float(params.get("alpha", 1.5))
    a = min(max(a, 0.1), 3.0)
    if abs(a - 1.0) < 1e-6:
        raise ValueError(
            "alpha=1 is undefined for renyi_mi/tsallis_mi in this implementation; "
            "use mi_kde/nmi_kde for Shannon MI or choose alpha away from 1.0"
        )
    return a


def loss_is_within_relative_tolerance(loss_before: float, loss_after: float, rel_tol: float) -> bool:
    """Return True when ``loss_after`` stays within a relative tolerance of ``loss_before``."""
    before = float(loss_before)
    after = float(loss_after)
    tol = max(float(rel_tol), 0.0) * abs(before)
    return after < before + tol


def _loss_l2(pred: jnp.ndarray, tar: jnp.ndarray, _: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum(r * r)


def _loss_charbonnier(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    return jnp.sum(jnp.sqrt(r * r + eps * eps))


def _loss_huber(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    delta = float(st.params.get("delta", 1.0))
    absr = jnp.abs(r)
    quad = jnp.minimum(absr, delta)
    return jnp.sum(0.5 * quad * quad + delta * (absr - quad))


def _loss_cauchy(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    c = _safe_epsilon(st.params, "c", 1.0)
    z = (r / c) ** 2
    return jnp.sum(0.5 * (c * c) * jnp.log1p(z))


def _loss_welsch(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    c = _safe_epsilon(st.params, "c", 1.0)
    z = (r / c) ** 2
    return jnp.sum(0.5 * (c * c) * (1.0 - jnp.exp(-z)))


def _loss_zncc(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = _safe_epsilon(st.params, "eps", 1e-5)
    p = pred.astype(jnp.float32); y = tar.astype(jnp.float32)
    p0 = p - jnp.mean(p); y0 = y - jnp.mean(y)
    num = jnp.sum(p0 * y0)
    den = jnp.sqrt(jnp.sum(p0 * p0) * jnp.sum(y0 * y0) + eps)
    ncc = num / den
    return (1.0 - ncc) * jnp.float32(p.size)


def _avgpool2(x: jnp.ndarray, k: int) -> jnp.ndarray:
    x4 = x[None, :, :, None]
    kernel = jnp.ones((k, k, 1, 1), dtype=x4.dtype) / float(k * k)
    y4 = jax.lax.conv_general_dilated(x4, kernel, (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y4[0, :, :, 0]


def _loss_ssim(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    K1 = float(st.params.get("K1", 0.01)); K2 = float(st.params.get("K2", 0.03))
    win = max(3, int(st.params.get("window", 7)) | 1)
    C1 = (K1 * 1.0) ** 2; C2 = (K2 * 1.0) ** 2
    x = pred.astype(jnp.float32); y = tar.astype(jnp.float32)
    mu_x = _avgpool2(x, win); mu_y = _avgpool2(y, win)
    sigma_x2 = _avgpool2(x * x, win) - mu_x * mu_x
    sigma_y2 = _avgpool2(y * y, win) - mu_y * mu_y
    sigma_xy = _avgpool2(x * y, win) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2))
    return (1.0 - jnp.mean(ssim_map)) * jnp.float32(x.size)


def _loss_ms_ssim(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    levels = int(st.params.get("levels", 3))
    win = max(3, int(st.params.get("window", 7)) | 1)
    K1 = float(st.params.get("K1", 0.01)); K2 = float(st.params.get("K2", 0.03))
    x = pred.astype(jnp.float32); y = tar.astype(jnp.float32)
    mssim = 1.0
    for li in range(levels):
        C1 = (K1 * 1.0) ** 2; C2 = (K2 * 1.0) ** 2
        mu_x = _avgpool2(x, win); mu_y = _avgpool2(y, win)
        sigma_x2 = _avgpool2(x * x, win) - mu_x * mu_x
        sigma_y2 = _avgpool2(y * y, win) - mu_y * mu_y
        sigma_xy = _avgpool2(x * y, win) - mu_x * mu_y
        ssim = jnp.mean(((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)))
        mssim = mssim * jnp.clip(ssim, 0.0, 1.0)
        if li < levels - 1:
            x = _avgpool2(x, 2)[::2, ::2]
            y = _avgpool2(y, 2)[::2, ::2]
    return (1.0 - mssim) * jnp.float32(pred.size)


def _sobel(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], jnp.float32) / 8.0
    ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], jnp.float32) / 8.0
    x4 = x[None, :, :, None]
    gx = jax.lax.conv_general_dilated(x4, kx[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))[0, :, :, 0]
    gy = jax.lax.conv_general_dilated(x4, ky[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))[0, :, :, 0]
    return gx, gy


def _loss_grad_l1(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    pgx, pgy = _sobel(pred.astype(jnp.float32)); tgx, tgy = _sobel(tar.astype(jnp.float32))
    return jnp.sum(jnp.abs(pgx - tgx) + jnp.abs(pgy - tgy))


def _loss_edge_aware_l2(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    tgx, tgy = _sobel(tar.astype(jnp.float32))
    w = jnp.sqrt(tgx * tgx + tgy * tgy)
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum((w + 1.0) * r * r)


def _loss_ngf(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    pgx, pgy = _sobel(pred.astype(jnp.float32)); tgx, tgy = _sobel(tar.astype(jnp.float32))
    num = (pgx * tgx + pgy * tgy) ** 2
    den = (pgx * pgx + pgy * pgy + eps) * (tgx * tgx + tgy * tgy + eps)
    return jnp.sum(1.0 - num / den)


def _loss_grad_orient(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    pgx, pgy = _sobel(pred.astype(jnp.float32)); tgx, tgy = _sobel(tar.astype(jnp.float32))
    dot = pgx * tgx + pgy * tgy
    cos = dot / (jnp.sqrt(pgx * pgx + pgy * pgy + eps) * jnp.sqrt(tgx * tgx + tgy * tgy + eps))
    return jnp.sum(1.0 - cos)


def _fft2(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x.astype(jnp.complex64))


def _loss_phase_corr_soft(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    beta = float(st.params.get("beta", 10.0))
    A = _fft2(pred); B = _fft2(tar)
    num = A * jnp.conj(B); den = jnp.abs(num) + 1e-6
    C = jnp.fft.ifft2(num / den).real
    w = jax.nn.softmax(beta * C.ravel())
    exp_corr = jnp.sum(w * C.ravel())
    return -exp_corr * jnp.float32(pred.size)


def _loss_fft_mag(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    Ap = _fft2(pred); At = _fft2(tar)
    mp = jnp.abs(Ap); mt = jnp.abs(At)
    mp = mp / (jnp.mean(mp) + 1e-6); mt = mt / (jnp.mean(mt) + 1e-6)
    return 0.5 * jnp.sum((mp - mt) ** 2)


def _required_state_precompute[T](value: T | None, message: str) -> T:
    if value is None:
        raise ValueError(message)
    return value


def _loss_chamfer_edge(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    dt_edge = _required_state_precompute(st.dt_edge, "dt_edge precompute required")
    gx, gy = _sobel(pred.astype(jnp.float32))
    mag = jnp.sqrt(gx * gx + gy * gy)
    e = mag / (jnp.mean(mag) + 1e-6)
    return jnp.sum(e * dt_edge)


def _loss_poisson_nll(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = 1e-6
    predp = jnp.clip(pred.astype(jnp.float32), eps, None)
    y = jnp.clip(tar.astype(jnp.float32), 0.0, None)
    return jnp.sum(predp - y * jnp.log(predp))


def _loss_pwls(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    a = float(st.params.get("a", 1.0)); b = float(st.params.get("b", 0.0))
    y = jnp.clip(tar.astype(jnp.float32), 0.0, None)
    w = 1.0 / (a * y + b + 1e-6)
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum(w * r * r)


def _loss_student_t(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    nu = max(1e-3, float(st.params.get("nu", 4.0)))
    sigma = max(1e-6, float(st.params.get("sigma", 1.0)))
    r2 = ((pred - tar).astype(jnp.float32) / sigma) ** 2
    return ((nu + 1.0) / 2.0) * jnp.sum(jnp.log1p(r2 / nu))


def _loss_barron(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    a = float(st.params.get("alpha", 1.0)); c = max(1e-6, float(st.params.get("c", 1.0)))
    x2 = ((pred - tar).astype(jnp.float32) / c) ** 2
    if abs(a - 2.0) < 1e-6:
        return 0.5 * jnp.sum(x2)
    if abs(a) < 1e-6:
        return jnp.sum(jnp.log1p(0.5 * x2))
    beta = a - 2.0
    return (1.0 / a) * jnp.sum(jnp.power(1.0 + x2 / (abs(beta) + 1e-8), a / 2.0) - 1.0)


def _loss_correntropy(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    sigma = max(1e-6, float(st.params.get("sigma", 1.0)))
    r = (pred - tar).astype(jnp.float32)
    c = jnp.exp(-0.5 * (r / sigma) ** 2)
    return (1.0 - jnp.mean(c)) * jnp.float32(pred.size)


def _kde_probabilities(
    pred: jnp.ndarray,
    tar: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
    bwx: float,
    bwy: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pr = pred.ravel()[:, None]
    tr = tar.ravel()[:, None]
    Wx = jnp.exp(-0.5 * ((pr - bx[None, :]) / bwx) ** 2)
    Wy = jnp.exp(-0.5 * ((tr - by[None, :]) / bwy) ** 2)
    Wx = Wx / (jnp.sum(Wx, axis=1, keepdims=True) + 1e-12)
    Wy = Wy / (jnp.sum(Wy, axis=1, keepdims=True) + 1e-12)
    sample_count = jnp.asarray(Wx.shape[0], dtype=Wx.dtype)
    Px = jnp.clip(Wx.mean(axis=0), 1e-12, 1.0)
    Py = jnp.clip(Wy.mean(axis=0), 1e-12, 1.0)
    Pxy = jnp.clip((Wx.T @ Wy) / sample_count, 1e-12, 1.0)
    return Px, Py, Pxy


def _loss_mi_kde(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    bins = int(st.params.get("bins", 32))
    bx = st.bins_x; by = st.bins_y
    bwx = st.bw_x or 0.1; bwy = st.bw_y or 0.1
    if bx is None or by is None:
        lo = jnp.minimum(pred.min(), tar.min()); hi = jnp.maximum(pred.max(), tar.max())
        bx = jnp.linspace(lo, hi, bins); by = jnp.linspace(lo, hi, bins)
    Px, Py, Pxy = _kde_probabilities(pred, tar, bx, by, bwx, bwy)
    Hx = -jnp.sum(Px * jnp.log(Px)); Hy = -jnp.sum(Py * jnp.log(Py)); Hxy = -jnp.sum(Pxy * jnp.log(Pxy))
    mi = Hx + Hy - Hxy
    if int(st.params.get("nmi", 0)) == 1:
        mi = mi / (jnp.sqrt(Hx * Hy) + 1e-12)
        return (1.0 - mi) * jnp.float32(pred.size)
    return (-mi) * jnp.float32(pred.size)


def _loss_renyi_mi(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    a = _validated_renyi_alpha(st.params)
    bins = int(st.params.get("bins", 32))
    bx = st.bins_x; by = st.bins_y
    bwx = st.bw_x or 0.1; bwy = st.bw_y or 0.1
    if bx is None or by is None:
        lo = jnp.minimum(pred.min(), tar.min()); hi = jnp.maximum(pred.max(), tar.max())
        bx = jnp.linspace(lo, hi, bins); by = jnp.linspace(lo, hi, bins)
    Px, Py, Pxy = _kde_probabilities(pred, tar, bx, by, bwx, bwy)
    Hx = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Px ** a))
    Hy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Py ** a))
    Hxy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Pxy ** a))
    mi = Hx + Hy - Hxy
    return (-mi) * jnp.float32(pred.size)


def _compute_otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).any():
        return float(np.nan)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax <= vmin:
        return float(vmin)
    hist, bin_edges = np.histogram(x, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    p = hist / max(1e-12, hist.sum())
    omega = np.cumsum(p); mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


def _loss_l2_otsu_soft(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    mask = _required_state_precompute(st.mask, "Otsu mask required")
    temp = _safe_epsilon(st.params, "temp", 0.5)
    base = mask.astype(jnp.float32)
    r = (pred - tar).astype(jnp.float32)
    w = jax.nn.sigmoid((base - 0.5) / temp)
    return 0.5 * jnp.sum((r * w) ** 2)


def _loss_ssim_otsu(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    mask = _required_state_precompute(st.mask, "Otsu mask required").astype(jnp.float32)
    return _loss_ssim(pred * mask, tar * mask, st)


def _loss_tversky(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    thr = _required_state_precompute(st.thr, "Otsu threshold required")
    temp = _safe_epsilon(st.params, "temp", 0.5)
    alpha = float(st.params.get("alpha", 0.7)); beta = float(st.params.get("beta", 0.3))
    gamma = float(st.params.get("gamma", 1.0))
    p = jax.nn.sigmoid((pred - thr) / temp)
    t = (tar >= thr).astype(jnp.float32)
    tp = jnp.sum(p * t); fp = jnp.sum(p * (1.0 - t)); fn = jnp.sum((1.0 - p) * t)
    tv = tp / (tp + alpha * fn + beta * fp + 1e-6)
    loss = 1.0 - tv
    if gamma != 1.0:
        loss = loss ** gamma
    return loss * jnp.float32(pred.size)


def _loss_swd(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    n = int(st.params.get("n_samples", -1)); p = int(st.params.get("p", 1))
    xr = pred.ravel(); yr = tar.ravel()
    if n > 0 and n < xr.size:
        idx = jax.random.choice(jax.random.PRNGKey(0), xr.size, (n,), replace=False)
        xr = xr[idx]; yr = yr[idx]
    xs = jnp.sort(xr); ys = jnp.sort(yr)
    d = jnp.mean((xs - ys) ** 2) if p == 2 else jnp.mean(jnp.abs(xs - ys))
    return d * jnp.float32(pred.size)


def _gauss_kernel(size: int, sigma: float) -> jnp.ndarray:
    ax = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    kernel = jnp.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return kernel


def _mind_descriptor(x: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """Simplified MIND features with 4 neighbors and 3x3 patch smoothing."""
    x = x.astype(jnp.float32)
    # Smooth with 3x3 mean filter
    x4 = x[None, :, :, None]
    k = jnp.ones((3, 3, 1, 1), jnp.float32) / 9.0
    xs = jax.lax.conv_general_dilated(x4, k, (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))[0, :, :, 0]
    # Offsets: right, left, down, up
    offs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    feats = []
    for dy, dx in offs:
        shifted = jnp.roll(xs, shift=(dy, dx), axis=(0, 1))
        d2 = (xs - shifted) ** 2
        feats.append(jnp.exp(-d2 / (jnp.mean(d2) + 1e-6)))
    return jnp.stack(feats, axis=-1)  # (H, W, 4)


def _loss_mind(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    fp = _mind_descriptor(pred, st); ft = _mind_descriptor(tar, st)
    return 0.5 * jnp.sum((fp - ft) ** 2)


def _build_loss_from_kind(
    kind: str,
    params: Optional[Dict[str, float]],
    targets: jnp.ndarray,
) -> Tuple[PerViewLossFn, LossState]:
    k = str(kind).lower()
    p = {} if params is None else {str(a): float(b) for a, b in params.items()}
    state = LossState(kind=k, params=p)

    if k == "l2":
        f = _loss_l2
    elif k in ("charbonnier", "charb"):
        f = _loss_charbonnier
    elif k == "huber":
        f = _loss_huber
    elif k in ("cauchy", "lorentzian"):
        f = _loss_cauchy
    elif k in ("welsch", "leclerc"):
        f = _loss_welsch
    elif k in ("zncc", "ncc"):
        f = _loss_zncc
    elif k == "ssim":
        f = _loss_ssim
    elif k in ("ms-ssim", "msssim", "ms_ssim"):
        f = _loss_ms_ssim
    elif k in ("grad_l1", "gdl"):
        f = _loss_grad_l1
    elif k in ("edge_l2", "edge_aware_l2"):
        f = _loss_edge_aware_l2
    elif k == "ngf":
        f = _loss_ngf
    elif k in ("go", "grad_orient"):
        f = _loss_grad_orient
    elif k in ("phasecorr", "phase_corr_soft"):
        f = _loss_phase_corr_soft
    elif k in ("fft_mag", "fftmag"):
        f = _loss_fft_mag
    elif k in ("chamfer_edge", "chamfer"):
        if distance_transform_edt is None:
            raise ValueError("scipy.ndimage is required for chamfer_edge loss")
        Ys = np.asarray(targets, np.float32)
        dts = []
        for i in range(Ys.shape[0]):
            gy, gx = np.gradient(Ys[i])
            mag = np.sqrt(gx * gx + gy * gy)
            edges = mag > (0.5 * max(1e-6, float(np.mean(mag))))
            dts.append(distance_transform_edt(~edges).astype(np.float32))
        state.dt_edge = jax.device_put(jnp.asarray(np.stack(dts, axis=0)))
        f = _loss_chamfer_edge
    elif k in ("poisson", "poisson_nll"):
        f = _loss_poisson_nll
    elif k in ("pwls",):
        f = _loss_pwls
    elif k in ("student_t", "student-t"):
        f = _loss_student_t
    elif k in ("barron", "robust_general"):
        f = _loss_barron
    elif k in ("correntropy", "mcc"):
        f = _loss_correntropy
    elif k in ("mi", "mi_kde"):
        bins = int(p.get("bins", 32))
        lo = float(np.min(targets)); hi = float(np.max(targets)); hi = hi if hi > lo else (lo + 1.0)
        state.bins_x = jnp.linspace(lo, hi, bins); state.bins_y = jnp.linspace(lo, hi, bins)
        state.bw_x = float(p.get("bw_x", (hi - lo) / max(8.0, bins)))
        state.bw_y = float(p.get("bw_y", (hi - lo) / max(8.0, bins)))
        f = _loss_mi_kde
    elif k in ("nmi", "nmi_kde"):
        bins = int(p.get("bins", 32))
        lo = float(np.min(targets)); hi = float(np.max(targets)); hi = hi if hi > lo else (lo + 1.0)
        state.bins_x = jnp.linspace(lo, hi, bins); state.bins_y = jnp.linspace(lo, hi, bins)
        state.bw_x = float(p.get("bw_x", (hi - lo) / max(8.0, bins)))
        state.bw_y = float(p.get("bw_y", (hi - lo) / max(8.0, bins)))
        state.params["nmi"] = 1.0
        f = _loss_mi_kde
    elif k in ("renyi_mi", "tsallis_mi"):
        _validated_renyi_alpha(state.params)
        bins = int(p.get("bins", 32))
        lo = float(np.min(targets)); hi = float(np.max(targets)); hi = hi if hi > lo else (lo + 1.0)
        state.bins_x = jnp.linspace(lo, hi, bins); state.bins_y = jnp.linspace(lo, hi, bins)
        state.bw_x = float(p.get("bw_x", (hi - lo) / max(8.0, bins)))
        state.bw_y = float(p.get("bw_y", (hi - lo) / max(8.0, bins)))
        f = _loss_renyi_mi
    elif k in ("l2_otsu", "l2-otsu", "otsu-l2"):
        temp = _safe_epsilon(p, "temp", 0.5)
        Ys = np.asarray(targets)
        thr = np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)
        base = jax.device_put(jax.nn.sigmoid((targets - jnp.asarray(thr)[:, None, None]) / temp))
        state.mask = base
        f = _loss_l2_otsu_soft
    elif k == "ssim_otsu":
        Ys = np.asarray(targets)
        thr = np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)
        mask = (Ys >= thr[:, None, None]).astype(np.float32)
        state.mask = jax.device_put(jnp.asarray(mask))
        f = _loss_ssim_otsu
    elif k in ("tversky", "focal_tversky"):
        Ys = np.asarray(targets)
        thr = np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)
        state.thr = jax.device_put(jnp.asarray(thr)[:, None, None])
        f = _loss_tversky
    elif k in ("swd", "sliced_wasserstein"):
        f = _loss_swd
    elif k in ("mind",):
        f = _loss_mind
    else:
        raise ValueError(f"Unknown loss kind: {kind}")

    def per_view_fn(
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: Optional[jnp.ndarray],
        view_indices: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Vectorized application over batch of views (b, nv, nu)
        if (
            mask_chunk is None
            and state.mask is None
            and state.dt_edge is None
            and state.bins_x is None
            and state.thr is None
        ):
            return jax.vmap(lambda a, b: f(a, b, state))(pred_chunk, tar_chunk)

        bsz = pred_chunk.shape[0]
        local_indices = jnp.arange(bsz, dtype=jnp.int32)
        if view_indices is None:
            global_indices = local_indices
        else:
            global_indices = jnp.asarray(view_indices, dtype=jnp.int32)

        def apply_one(a, b, local_idx, global_idx):
            ls = LossState(kind=state.kind, params=state.params)
            if state.mask is not None:
                ls.mask = state.mask[global_idx]
            if mask_chunk is not None:
                ls.mask = mask_chunk[local_idx]
            if state.dt_edge is not None:
                ls.dt_edge = state.dt_edge[global_idx]
            ls.bins_x = state.bins_x; ls.bins_y = state.bins_y; ls.bw_x = state.bw_x; ls.bw_y = state.bw_y
            if state.thr is not None:
                ls.thr = state.thr[global_idx]
            return f(a, b, ls)

        return jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            pred_chunk,
            tar_chunk,
            local_indices,
            global_indices,
        )

    return per_view_fn, state


def _edge_aware_ls_weights(y_chunk: jnp.ndarray) -> jnp.ndarray:
    edge_kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], jnp.float32) / 8.0
    edge_ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], jnp.float32) / 8.0
    y4 = y_chunk[..., None]
    gx = jax.lax.conv_general_dilated(
        y4,
        edge_kx[:, :, None, None],
        (1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    gy = jax.lax.conv_general_dilated(
        y4,
        edge_ky[:, :, None, None],
        (1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    mag = jnp.sqrt(gx[..., 0] ** 2 + gy[..., 0] ** 2)
    return jnp.sqrt(1.0 + mag)


def _gauss_newton_weight_builder(spec: AlignmentLossSpec) -> tuple[bool, GaussNewtonWeightFn]:
    if isinstance(spec, L2LossSpec):
        return True, lambda y_chunk, mask_chunk: jnp.ones_like(y_chunk)
    if isinstance(spec, L2OtsuLossSpec):
        temp = jnp.float32(max(float(spec.temp), 1e-6))

        def _otsu_weights(y_chunk: jnp.ndarray, mask_chunk: Optional[jnp.ndarray]) -> jnp.ndarray:
            if mask_chunk is None:
                return jnp.ones_like(y_chunk)
            base = mask_chunk.astype(jnp.float32)
            return jax.nn.sigmoid((base - 0.5) / temp)

        return True, _otsu_weights
    if isinstance(spec, PWLSLossSpec):
        a = jnp.float32(spec.a)
        b = jnp.float32(spec.b)
        return True, lambda y_chunk, mask_chunk: jnp.sqrt(
            1.0 / (a * jnp.clip(y_chunk, 0.0) + b + 1e-6)
        )
    if isinstance(spec, EdgeL2LossSpec):
        return True, lambda y_chunk, mask_chunk: _edge_aware_ls_weights(y_chunk)
    return False, lambda y_chunk, mask_chunk: jnp.ones_like(y_chunk)


def build_loss_adapter(spec: AlignmentLossSpec, targets: jnp.ndarray) -> LossAdapter:
    name = loss_spec_name(spec)
    params = loss_spec_params(spec)
    per_view_loss, state = _build_loss_from_kind(name, params, targets)
    supports_gauss_newton, gauss_newton_weights = _gauss_newton_weight_builder(spec)
    return LossAdapter(
        spec=spec,
        name=name,
        state=state,
        per_view_loss=per_view_loss,
        supports_gauss_newton=supports_gauss_newton,
        gauss_newton_weights=gauss_newton_weights,
    )


def build_loss(
    kind: str,
    params: Optional[Dict[str, float]],
    targets: jnp.ndarray,
) -> Tuple[PerViewLossFn, LossState]:
    adapter = build_loss_adapter(parse_loss_spec(kind, params), targets)
    return adapter.per_view_loss, adapter.state
