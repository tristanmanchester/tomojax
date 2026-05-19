"""Projection-domain loss kernels for alignment scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .loss_state import LossState

T = TypeVar("T")


def _safe_epsilon(p: dict[str, float], key: str, default: float) -> float:
    v = float(p.get(key, default))
    return max(v, 1e-12)


def _validated_renyi_alpha(params: dict[str, float]) -> float:
    a = float(params.get("alpha", 1.5))
    a = min(max(a, 0.1), 3.0)
    if abs(a - 1.0) < 1e-6:
        raise ValueError(
            "alpha=1 is undefined for renyi_mi/tsallis_mi in this implementation; "
            "use mi_kde/nmi_kde for Shannon MI or choose alpha away from 1.0"
        )
    return a


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
    p = pred.astype(jnp.float32)
    y = tar.astype(jnp.float32)
    p0 = p - jnp.mean(p)
    y0 = y - jnp.mean(y)
    num = jnp.sum(p0 * y0)
    den = jnp.sqrt(jnp.sum(p0 * p0) * jnp.sum(y0 * y0) + eps)
    ncc = num / den
    return (1.0 - ncc) * jnp.float32(p.size)


def _avgpool2(x: jnp.ndarray, k: int) -> jnp.ndarray:
    x4 = x[None, :, :, None]
    kernel = jnp.ones((k, k, 1, 1), dtype=x4.dtype) / float(k * k)
    y4 = jax.lax.conv_general_dilated(
        x4, kernel, (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return y4[0, :, :, 0]


def _loss_ssim(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    K1 = float(st.params.get("K1", 0.01))
    K2 = float(st.params.get("K2", 0.03))
    win = max(3, int(st.params.get("window", 7)) | 1)
    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2
    x = pred.astype(jnp.float32)
    y = tar.astype(jnp.float32)
    mu_x = _avgpool2(x, win)
    mu_y = _avgpool2(y, win)
    sigma_x2 = _avgpool2(x * x, win) - mu_x * mu_x
    sigma_y2 = _avgpool2(y * y, win) - mu_y * mu_y
    sigma_xy = _avgpool2(x * y, win) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return (1.0 - jnp.mean(ssim_map)) * jnp.float32(x.size)


def _loss_ms_ssim(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    levels = int(st.params.get("levels", 3))
    win = max(3, int(st.params.get("window", 7)) | 1)
    K1 = float(st.params.get("K1", 0.01))
    K2 = float(st.params.get("K2", 0.03))
    x = pred.astype(jnp.float32)
    y = tar.astype(jnp.float32)
    mssim = 1.0
    for li in range(levels):
        C1 = (K1 * 1.0) ** 2
        C2 = (K2 * 1.0) ** 2
        mu_x = _avgpool2(x, win)
        mu_y = _avgpool2(y, win)
        sigma_x2 = _avgpool2(x * x, win) - mu_x * mu_x
        sigma_y2 = _avgpool2(y * y, win) - mu_y * mu_y
        sigma_xy = _avgpool2(x * y, win) - mu_x * mu_y
        ssim = jnp.mean(
            ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2))
            / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2))
        )
        mssim = mssim * jnp.clip(ssim, 0.0, 1.0)
        if li < levels - 1:
            x = _avgpool2(x, 2)[::2, ::2]
            y = _avgpool2(y, 2)[::2, ::2]
    return (1.0 - mssim) * jnp.float32(pred.size)


def _sobel(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], jnp.float32) / 8.0
    ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], jnp.float32) / 8.0
    x4 = x[None, :, :, None]
    gx = jax.lax.conv_general_dilated(
        x4, kx[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )[0, :, :, 0]
    gy = jax.lax.conv_general_dilated(
        x4, ky[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )[0, :, :, 0]
    return gx, gy


def _loss_grad_l1(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    del st
    pgx, pgy = _sobel(pred.astype(jnp.float32))
    tgx, tgy = _sobel(tar.astype(jnp.float32))
    return jnp.sum(jnp.abs(pgx - tgx) + jnp.abs(pgy - tgy))


def _loss_edge_aware_l2(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    del st
    tgx, tgy = _sobel(tar.astype(jnp.float32))
    w = jnp.sqrt(tgx * tgx + tgy * tgy)
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum((w + 1.0) * r * r)


def _loss_ngf(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    pgx, pgy = _sobel(pred.astype(jnp.float32))
    tgx, tgy = _sobel(tar.astype(jnp.float32))
    num = (pgx * tgx + pgy * tgy) ** 2
    den = (pgx * pgx + pgy * pgy + eps) * (tgx * tgx + tgy * tgy + eps)
    return jnp.sum(1.0 - num / den)


def _loss_grad_orient(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    pgx, pgy = _sobel(pred.astype(jnp.float32))
    tgx, tgy = _sobel(tar.astype(jnp.float32))
    dot = pgx * tgx + pgy * tgy
    cos = dot / (jnp.sqrt(pgx * pgx + pgy * pgy + eps) * jnp.sqrt(tgx * tgx + tgy * tgy + eps))
    return jnp.sum(1.0 - cos)


def _fft2(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x.astype(jnp.complex64))


def _loss_phase_corr_soft(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    beta = float(st.params.get("beta", 10.0))
    A = _fft2(pred)
    B = _fft2(tar)
    num = A * jnp.conj(B)
    den = jnp.abs(num) + 1e-6
    C = jnp.fft.ifft2(num / den).real
    w = jax.nn.softmax(beta * C.ravel())
    exp_corr = jnp.sum(w * C.ravel())
    return -exp_corr * jnp.float32(pred.size)


def _loss_fft_mag(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    del st
    Ap = _fft2(pred)
    At = _fft2(tar)
    mp = jnp.abs(Ap)
    mt = jnp.abs(At)
    mp = mp / (jnp.mean(mp) + 1e-6)
    mt = mt / (jnp.mean(mt) + 1e-6)
    return 0.5 * jnp.sum((mp - mt) ** 2)


def _required_state_precompute[T](value: T | None, message: str) -> T:
    if value is None:
        raise ValueError(message)
    return value


def _loss_chamfer_edge(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    del tar
    dt_edge = _required_state_precompute(st.dt_edge, "dt_edge precompute required")
    gx, gy = _sobel(pred.astype(jnp.float32))
    mag = jnp.sqrt(gx * gx + gy * gy)
    e = mag / (jnp.mean(mag) + 1e-6)
    return jnp.sum(e * dt_edge)


def _loss_poisson_nll(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    del st
    eps = 1e-6
    predp = jnp.clip(pred.astype(jnp.float32), eps, None)
    y = jnp.clip(tar.astype(jnp.float32), 0.0, None)
    return jnp.sum(predp - y * jnp.log(predp))


def _loss_pwls(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    a = float(st.params.get("a", 1.0))
    b = float(st.params.get("b", 0.0))
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
    a = float(st.params.get("alpha", 1.0))
    c = max(1e-6, float(st.params.get("c", 1.0)))
    x2 = ((pred - tar).astype(jnp.float32) / c) ** 2
    if abs(a - 2.0) < 1e-6:
        return 0.5 * jnp.sum(x2)
    if abs(a) < 1e-6:
        return jnp.sum(jnp.log1p(0.5 * x2))
    beta = abs(a - 2.0)
    return (beta / a) * jnp.sum(jnp.power(1.0 + x2 / (beta + 1e-8), a / 2.0) - 1.0)


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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    bx = st.bins_x
    by = st.bins_y
    bwx = st.bw_x or 0.1
    bwy = st.bw_y or 0.1
    if bx is None or by is None:
        lo = jnp.minimum(pred.min(), tar.min())
        hi = jnp.maximum(pred.max(), tar.max())
        bx = jnp.linspace(lo, hi, bins)
        by = jnp.linspace(lo, hi, bins)
    Px, Py, Pxy = _kde_probabilities(pred, tar, bx, by, bwx, bwy)
    Hx = -jnp.sum(Px * jnp.log(Px))
    Hy = -jnp.sum(Py * jnp.log(Py))
    Hxy = -jnp.sum(Pxy * jnp.log(Pxy))
    mi = Hx + Hy - Hxy
    if int(st.params.get("nmi", 0)) == 1:
        mi = mi / (jnp.sqrt(Hx * Hy) + 1e-12)
        return (1.0 - mi) * jnp.float32(pred.size)
    return (-mi) * jnp.float32(pred.size)


def _loss_renyi_mi(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    a = _validated_renyi_alpha(st.params)
    bins = int(st.params.get("bins", 32))
    bx = st.bins_x
    by = st.bins_y
    bwx = st.bw_x or 0.1
    bwy = st.bw_y or 0.1
    if bx is None or by is None:
        lo = jnp.minimum(pred.min(), tar.min())
        hi = jnp.maximum(pred.max(), tar.max())
        bx = jnp.linspace(lo, hi, bins)
        by = jnp.linspace(lo, hi, bins)
    Px, Py, Pxy = _kde_probabilities(pred, tar, bx, by, bwx, bwy)
    Hx = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Px**a))
    Hy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Py**a))
    Hxy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Pxy**a))
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
    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
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
    alpha = float(st.params.get("alpha", 0.7))
    beta = float(st.params.get("beta", 0.3))
    gamma = float(st.params.get("gamma", 1.0))
    p = jax.nn.sigmoid((pred - thr) / temp)
    t = (tar >= thr).astype(jnp.float32)
    tp = jnp.sum(p * t)
    fp = jnp.sum(p * (1.0 - t))
    fn = jnp.sum((1.0 - p) * t)
    tv = tp / (tp + alpha * fn + beta * fp + 1e-6)
    loss = 1.0 - tv
    if gamma != 1.0:
        loss = loss**gamma
    return loss * jnp.float32(pred.size)


def _loss_swd(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    n = int(st.params.get("n_samples", -1))
    p = int(st.params.get("p", 1))
    xr = pred.ravel()
    yr = tar.ravel()
    if n > 0 and n < xr.size:
        idx = jax.random.choice(jax.random.key(0), xr.size, (n,), replace=False)
        xr = xr[idx]
        yr = yr[idx]
    xs = jnp.sort(xr)
    ys = jnp.sort(yr)
    d = jnp.mean((xs - ys) ** 2) if p == 2 else jnp.mean(jnp.abs(xs - ys))
    return d * jnp.float32(pred.size)


def _gauss_kernel(size: int, sigma: float) -> jnp.ndarray:
    ax = jnp.arange(-size // 2 + 1.0, size // 2 + 1.0)
    kernel = jnp.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / jnp.sum(kernel)


def _mind_descriptor(x: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """Simplified MIND features with 4 neighbors and 3x3 patch smoothing."""
    del st
    x = x.astype(jnp.float32)
    # Smooth with 3x3 mean filter
    x4 = x[None, :, :, None]
    k = jnp.ones((3, 3, 1, 1), jnp.float32) / 9.0
    xs = jax.lax.conv_general_dilated(
        x4, k, (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )[0, :, :, 0]
    # Offsets: right, left, down, up
    offs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    feats = []
    for dy, dx in offs:
        shifted = jnp.roll(xs, shift=(dy, dx), axis=(0, 1))
        d2 = (xs - shifted) ** 2
        feats.append(jnp.exp(-d2 / (jnp.mean(d2) + 1e-6)))
    return jnp.stack(feats, axis=-1)  # (H, W, 4)


def _loss_mind(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    fp = _mind_descriptor(pred, st)
    ft = _mind_descriptor(tar, st)
    return 0.5 * jnp.sum((fp - ft) ** 2)


__all__ = [
    "_compute_otsu_threshold",
    "_loss_barron",
    "_loss_cauchy",
    "_loss_chamfer_edge",
    "_loss_charbonnier",
    "_loss_correntropy",
    "_loss_edge_aware_l2",
    "_loss_fft_mag",
    "_loss_grad_l1",
    "_loss_grad_orient",
    "_loss_huber",
    "_loss_l2",
    "_loss_l2_otsu_soft",
    "_loss_mi_kde",
    "_loss_mind",
    "_loss_ms_ssim",
    "_loss_ngf",
    "_loss_phase_corr_soft",
    "_loss_poisson_nll",
    "_loss_pwls",
    "_loss_renyi_mi",
    "_loss_ssim",
    "_loss_ssim_otsu",
    "_loss_student_t",
    "_loss_swd",
    "_loss_tversky",
    "_loss_welsch",
    "_loss_zncc",
    "_safe_epsilon",
    "_sobel",
    "_validated_renyi_alpha",
]
