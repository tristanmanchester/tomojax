from __future__ import annotations

from tomojax.core import format_duration

from ._config import AlignConfig
from ._observer import OuterStat


def _recon_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    digits = 2 if compact else 3
    recon_time = stat.get("recon_time")
    if recon_time is not None:
        prefix = "" if compact else "time "
        parts.append(f"{prefix}{format_duration(recon_time)}")
    if stat.get("recon_retry"):
        parts.append("retry" if compact else "fallback retry")
    l_meas = stat.get("L_meas")
    l_next = stat.get("L_next")
    if (l_meas is not None) and (l_next is not None):
        parts.append(f"L {float(l_meas):.{digits}e}->{float(l_next):.{digits}e}")
    f_first = stat.get("recon_loss_first")
    f_last = stat.get("recon_loss_last")
    f_min = stat.get("recon_loss_min")
    if (f_first is not None) and (f_last is not None):
        loss = f"loss {float(f_first):.{digits}e}->{float(f_last):.{digits}e}"
        if f_min is not None:
            loss += f" (min {float(f_min):.{digits}e})"
        parts.append(loss)
    return parts


def _gauge_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    if stat.get("gauge_fix") == "none":
        return ["gauge none"]
    if stat.get("gauge_fix") != "mean_translation":
        return []
    dxm = stat.get("dx_mean_before_gauge")
    dzm = stat.get("dz_mean_before_gauge")
    if compact:
        if dxm is not None and dzm is not None:
            return [f"gauge mean dx,dz {float(dxm):+.2e},{float(dzm):+.2e}->0"]
        return []
    dxa = stat.get("dx_mean_after_gauge")
    dza = stat.get("dz_mean_after_gauge")
    if dxm is not None and dzm is not None and dxa is not None and dza is not None:
        return [
            "gauge mean dx,dz "
            f"{float(dxm):+.3e},{float(dzm):+.3e}->"
            f"{float(dxa):+.3e},{float(dza):+.3e}"
        ]
    return []


def _gn_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    parts: list[str] = []
    rot_mean = stat.get("rot_mean")
    trans_mean = stat.get("trans_mean")
    if rot_mean is not None:
        label = "|drot|" if compact else "|drot|_mean"
        suffix = "" if compact else " rad"
        parts.append(f"{label} {float(rot_mean):.{digits}e}{suffix}")
    if trans_mean is not None:
        label = "|dtrans|" if compact else "|dtrans|_mean"
        parts.append(f"{label} {float(trans_mean):.{digits}e}")
    return parts


def _lbfgs_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    status = "accepted" if stat.get("lbfgs_accepted") else "rejected"
    if stat.get("lbfgs_fallback_to_gd"):
        status = "fallback->gd" if compact else "fallback to GD"
    parts.append(status if compact else f"L-BFGS {status}")
    if compact:
        best = stat.get("lbfgs_best_loss")
        if best is not None:
            parts.append(f"best {float(best):.2e}")
    else:
        for src, label in (
            ("lbfgs_initial_loss", "initial"),
            ("lbfgs_final_loss", "final"),
            ("lbfgs_best_loss", "best"),
        ):
            value = stat.get(src)
            if value is not None:
                parts.append(f"{label} {float(value):.3e}")
    nit = stat.get("lbfgs_nit")
    nfev = stat.get("lbfgs_nfev")
    if nit is not None:
        parts.append(f"nit {int(nit)}")
    if nfev is not None:
        parts.append(f"nfev {int(nfev)}")
    if not compact:
        message = stat.get("lbfgs_message")
        if message:
            parts.append(str(message))
    for src, label in (("rot_mean", "|drot|"), ("trans_mean", "|dtrans|")):
        value = stat.get(src)
        if value is not None and compact:
            parts.append(f"{label} {float(value):.2e}")
    return parts


def _gd_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    parts: list[str] = []
    if stat.get("lbfgs_fallback_to_gd"):
        message = stat.get("lbfgs_message")
        parts.append(
            "lbfgs fallback"
            if compact
            else "L-BFGS fallback to GD" + (f": {message}" if message else "")
        )
    rot_rms = stat.get("rot_rms")
    trans_rms = stat.get("trans_rms")
    if rot_rms is not None:
        label = "rotRMS" if compact else "rot RMS"
        parts.append(f"{label} {float(rot_rms):.{digits}e}")
    if trans_rms is not None:
        label = "transRMS" if compact else "trans RMS"
        parts.append(f"{label} {float(trans_rms):.{digits}e}")
    return parts


def _step_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    step_kind = stat.get("step_kind")
    if step_kind == "gn":
        return _gn_summary_parts(stat, compact=compact, digits=digits)
    if step_kind == "lbfgs":
        return _lbfgs_summary_parts(stat, compact=compact)
    if step_kind == "gd":
        return _gd_summary_parts(stat, compact=compact, digits=digits)
    return []


def _align_loss_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    loss_before = stat.get("loss_before")
    loss_after = stat.get("loss_after")
    if (loss_before is None) or (loss_after is None):
        return []
    loss_delta = stat.get("loss_delta")
    rel_pct = stat.get("loss_rel_pct")
    sep = " " if compact else ", "
    rel = f"{sep}{float(rel_pct):+.2f}%" if rel_pct is not None else ""
    return [
        f"loss {float(loss_before):.{digits}e}->{float(loss_after):.{digits}e} "
        f"(Δ {float(loss_delta):+.{digits}e}{rel})"
    ]


def _align_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    digits = 2 if compact else 3
    align_time = stat.get("align_time")
    if align_time is not None:
        prefix = "" if compact else "time "
        parts.append(f"{prefix}{format_duration(align_time)}")
    parts.extend(_step_summary_parts(stat, compact=compact, digits=digits))
    parts.extend(_align_loss_summary_parts(stat, compact=compact, digits=digits))
    parts.extend(_gauge_summary_parts(stat, compact=compact))
    return parts


def _format_outer_summary_lines(
    stat: OuterStat,
    *,
    cfg: AlignConfig,
    recon_algo: str,
) -> list[str]:
    outer_idx = int(stat.get("outer_idx", 0))
    total_iters = int(cfg.outer_iters)
    total_time = format_duration(stat.get("outer_time"))
    elapsed = format_duration(stat.get("cumulative_time"))
    solver_label = str(stat.get("recon_algo") or recon_algo).upper()
    if cfg.log_compact:
        parts: list[str] = [f"Outer {outer_idx}/{total_iters}"]
        recon_parts = _recon_summary_parts(stat, compact=True)
        align_parts = _align_summary_parts(stat, compact=True)
        if recon_parts:
            parts.append(f"recon {solver_label.lower()} " + " ".join(recon_parts))
        if align_parts:
            parts.append("align " + " ".join(align_parts))
        parts.append(f"elapsed {elapsed}")
        return [" | ".join(parts)]
    recon_parts = _recon_summary_parts(stat, compact=False)
    align_parts = _align_summary_parts(stat, compact=False)
    return [
        f"Outer {outer_idx}/{total_iters} | total {total_time} | elapsed {elapsed}",
        f"  Recon ({solver_label}) | {' | '.join(recon_parts) if recon_parts else '-'}",
        f"  Align | {' | '.join(align_parts) if align_parts else '-'}",
    ]
