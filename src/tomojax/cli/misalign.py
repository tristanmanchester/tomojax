from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import jax
import jax.numpy as jnp
from contextlib import nullcontext as _nullcontext

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..align.parametrizations import se3_from_5d
from ..core.projector import forward_project_view_T
from ..utils.logging import setup_logging, log_jax_env


def _transfer_guard_ctx(mode: str | None = None):
    # Allow overriding via env var: off|log|disallow
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
    if mode in ("off", "none", "disable", "disabled"):
        return _nullcontext()
    try:
        tg = getattr(jax, "transfer_guard", None)
        if tg is not None:
            return tg(mode)
        try:
            from jax.experimental import transfer_guard as _tg  # type: ignore
            return _tg(mode)
        except Exception:
            return _nullcontext()
    except Exception:
        return _nullcontext()


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {s}")


def _parse_number_with_unit(val: str) -> tuple[float, str | None]:
    """Parse a number with optional unit suffix.

    Supports suffixes: deg, rad, px. Returns (value, unit_or_None).
    """
    s = val.strip().lower()
    for suf in ("deg", "rad", "px"):
        if s.endswith(suf):
            num = float(s[: -len(suf)])
            return num, suf
    return float(s), None


def _parse_pert(spec: str) -> tuple[str, str, dict[str, str]]:
    """Parse a --pert specification: dof:shape[:k=v[,k=v...]]"""
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid --pert spec (need dof:shape): {spec}")
    dof = parts[0].strip().lower()
    shape = parts[1].strip().lower()
    params: dict[str, str] = {}
    if len(parts) == 3 and parts[2].strip():
        kvs = parts[2].split(",")
        for kv in kvs:
            if not kv:
                continue
            if "=" not in kv:
                raise ValueError(f"Invalid param in --pert: '{kv}' (expected k=v)")
            k, v = kv.split("=", 1)
            params[k.strip().lower()] = v.strip()
    # normalize aliases
    if dof in ("x", "u"):
        dof = "dx"
    if dof in ("y", "v"):
        dof = "dz"
    return dof, shape, params


def _domain_from_params(params: dict[str, str]) -> str:
    d = params.get("domain", "angle").lower()
    if d not in ("angle", "index"):
        raise ValueError(f"Invalid domain: {d}")
    return d


def _window_indices_for_domain(
    thetas_deg: jnp.ndarray,
    n_views: int,
    params: dict[str, str],
) -> tuple[int, int]:
    """Compute [start_idx, end_idx] inclusive for an optional window.

    - angle domain: use start_deg/end_deg if provided, else full span
    - index domain: use start_index/end_index if provided, else full span
    """
    d = _domain_from_params(params)
    if d == "index":
        si = int(params.get("start_index", 0))
        ei = int(params.get("end_index", n_views - 1))
        si = max(0, min(n_views - 1, si))
        ei = max(si, min(n_views - 1, ei))
        return si, ei
    # angle domain
    if "start_deg" in params or "end_deg" in params:
        th = jnp.asarray(thetas_deg)
        th0 = float(params.get("start_deg", float(th.min())))
        th1 = float(params.get("end_deg", float(th.max())))
        # pick closest indices bounding the window
        si = int(jnp.argmin(jnp.abs(th - th0)))
        ei = int(jnp.argmin(jnp.abs(th - th1)))
        if ei < si:
            si, ei = ei, si
        return si, ei
    return 0, n_views - 1


def _apply_linear(schedule: np.ndarray, thetas_deg: jnp.ndarray, params: dict[str, str]) -> None:
    """Apply a linear ramp inside a window to the given schedule array (in-place)."""
    n = schedule.shape[0]
    si, ei = _window_indices_for_domain(thetas_deg, n, params)
    if ei <= si:
        return
    # Determine start and end offsets
    start_off = 0.0
    end_off = 0.0
    if "delta" in params:
        v, _u = _parse_number_with_unit(params["delta"])
        end_off = float(v)
    if "start" in params:
        v, _u = _parse_number_with_unit(params["start"])
        start_off = float(v)
    if "end" in params:
        v, _u = _parse_number_with_unit(params["end"])
        end_off = float(v)
    # Ramp
    m = ei - si
    ramp = np.linspace(0.0, 1.0, m + 1, dtype=np.float32)
    vals = start_off + (end_off - start_off) * ramp
    schedule[si : ei + 1] += vals


def _apply_sin_window(schedule: np.ndarray, thetas_deg: jnp.ndarray, params: dict[str, str]) -> None:
    """Apply a single-lobe sine window (0->amp->0) within a window."""
    n = schedule.shape[0]
    si, ei = _window_indices_for_domain(thetas_deg, n, params)
    if ei <= si:
        return
    amp = float(_parse_number_with_unit(params.get("amp", "0"))[0])
    m = ei - si
    t = np.linspace(0.0, 1.0, m + 1, dtype=np.float32)
    vals = amp * np.sin(np.pi * t)  # 0 at edges, amp at center
    schedule[si : ei + 1] += vals


def _apply_step(schedule: np.ndarray, thetas_deg: jnp.ndarray, params: dict[str, str]) -> None:
    """Apply a step at angle or index.

    Params:
      - at_deg or at_index: where the step begins
      - to: set absolute value to this within the step window (computed relative to current)
      - delta: add this offset within the window
      - width_deg / width_index: optional duration; if missing, holds to the end
    """
    n = schedule.shape[0]
    d = _domain_from_params(params)
    if d == "index":
        at = int(params.get("at_index", 0))
        at = max(0, min(n - 1, at))
    else:
        th = np.asarray(thetas_deg)
        at_deg = float(_parse_number_with_unit(params.get("at", params.get("at_deg", "0")))[0])
        at = int(np.argmin(np.abs(th - at_deg)))
    if at >= n:
        return
    # Width / until
    if d == "index":
        if "width_index" in params:
            end = min(n - 1, at + int(params["width_index"]))
        elif "until_index" in params:
            end = max(at, min(n - 1, int(params["until_index"])) )
        else:
            end = n - 1
    else:
        if "width_deg" in params:
            width = float(_parse_number_with_unit(params["width_deg"])[0])
            th = np.asarray(thetas_deg)
            target = float(th[at]) + width
            end = int(np.argmin(np.abs(th - target)))
            end = max(at, min(n - 1, end))
        elif "until_deg" in params:
            target = float(_parse_number_with_unit(params["until_deg"])[0])
            th = np.asarray(thetas_deg)
            end = int(np.argmin(np.abs(th - target)))
            end = max(at, min(n - 1, end))
        else:
            end = n - 1
    # Delta or absolute target
    if "to" in params:
        tgt = float(_parse_number_with_unit(params["to"])[0])
        # Compute relative delta to reach absolute target
        delta = tgt - schedule[at]
    else:
        delta = float(_parse_number_with_unit(params.get("delta", "0"))[0])
    schedule[at : end + 1] += delta


def _apply_box(schedule: np.ndarray, thetas_deg: jnp.ndarray, params: dict[str, str]) -> None:
    """Apply a box pulse: step up by delta at 'at', then down after a width."""
    # Step up
    _apply_step(schedule, thetas_deg, params)
    # Step down
    # Build params for the end step
    end_params = dict(params)
    if "width_index" in params or "until_index" in params:
        if "width_index" in params:
            end_params["at_index"] = str(int(params.get("at_index", params.get("at", 0))) + int(params["width_index"]))
        # if until_index is present, we step down at that index
        if "until_index" in params:
            end_params["at_index"] = params["until_index"]
    else:
        # angle domain width
        if "width_deg" in params:
            # at + width
            at_deg = float(_parse_number_with_unit(params.get("at", params.get("at_deg", "0")))[0])
            end_params["at"] = str(at_deg + float(_parse_number_with_unit(params["width_deg"])[0]))
        elif "until_deg" in params:
            end_params["at"] = params["until_deg"]
    # invert the step
    if "to" in end_params:
        # Returning to zero level (absolute)
        end_params["to"] = "0"
    elif "delta" in end_params:
        v, _ = _parse_number_with_unit(end_params["delta"])
        end_params["delta"] = str(-float(v))
    _apply_step(schedule, thetas_deg, end_params)


def _build_schedules(
    thetas_deg: jnp.ndarray,
    n_views: int,
    perts: list[tuple[str, str, dict[str, str]]],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, str]]]]:
    """Construct per-DOF schedules and return also a normalized spec for metadata."""
    schedules: dict[str, np.ndarray] = {k: np.zeros((n_views,), np.float32) for k in ("angle", "alpha", "beta", "phi", "dx", "dz")}
    norm_spec: dict[str, list[dict[str, str]]] = {}
    for dof, shape, params in perts:
        if dof not in schedules:
            raise ValueError(f"Unsupported DOF in --pert: {dof}")
        norm_spec.setdefault(dof, []).append({"kind": shape, **params})
        sched = schedules[dof]
        if shape in ("linear", "lin"):
            _apply_linear(sched, thetas_deg, params)
        elif shape in ("sin-window", "sinwin", "sin"):
            _apply_sin_window(sched, thetas_deg, params)
        elif shape in ("step",):
            _apply_step(sched, thetas_deg, params)
        elif shape in ("box",):
            _apply_box(sched, thetas_deg, params)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    return schedules, norm_spec


def _load_spec_file(path: str) -> list[tuple[str, str, dict[str, str]]]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[tuple[str, str, dict[str, str]]] = []
    # Two accepted layouts:
    # 1) { "dx": [{"kind":"step", ...}], "angle": [{...}], ... }
    # 2) { "schedules": [{"dof":"dx","kind":"step",...}, ...] }
    if isinstance(data, dict) and any(k in data for k in ("angle", "alpha", "beta", "phi", "dx", "dz")):
        for dof, lst in data.items():
            if dof not in ("angle", "alpha", "beta", "phi", "dx", "dz"):
                continue
            if not isinstance(lst, list):
                raise ValueError(f"Spec field for {dof} must be a list")
            for item in lst:
                if not isinstance(item, dict) or "kind" not in item:
                    raise ValueError(f"Spec items for {dof} must be dicts with 'kind'")
                    
                kind = str(item["kind"]).lower()
                params = {k: str(v) for k, v in item.items() if k != "kind"}
                out.append((dof, kind, params))
    elif isinstance(data, dict) and "schedules" in data and isinstance(data["schedules"], list):
        for it in data["schedules"]:
            if not isinstance(it, dict) or "dof" not in it or "kind" not in it:
                raise ValueError("Each schedule must have 'dof' and 'kind'")
            dof = str(it["dof"]).lower()
            kind = str(it["kind"]).lower()
            params = {k: str(v) for k, v in it.items() if k not in ("dof", "kind")}
            out.append((dof, kind, params))
    else:
        raise ValueError("Unrecognized spec file schema")
    # Normalize aliases in dof
    normed = []
    for dof, kind, params in out:
        if dof in ("x", "u"):
            dof = "dx"
        if dof in ("y", "v"):
            dof = "dz"
        normed.append((dof, kind, params))
    return normed


def main() -> None:
    p = argparse.ArgumentParser(description="Create a misaligned (and optionally noisy) dataset from a ground-truth NXtomo file.")
    p.add_argument("--data", required=True, help="Input .nxs containing ground-truth volume and geometry")
    p.add_argument("--out", required=True, help="Output .nxs path")
    p.add_argument("--rot-deg", type=float, default=1.0, help="Max abs rotation per-axis (alpha,beta,phi) in degrees (used for --with-random)")
    p.add_argument("--trans-px", type=float, default=10.0, help="Max abs translation in detector pixels (dx,dz) (used for --with-random)")
    # New deterministic perturbation modes
    p.add_argument("--pert", action="append", default=[], help="Additive schedule spec: dof:shape[:k=v[,k=v...]]; dof in {angle,alpha,beta,phi,dx,dz}")
    p.add_argument("--spec", type=str, default=None, help="JSON file with schedules; see docs/misalign_modes.md")
    p.add_argument("--with-random", action="store_true", help="Combine random misalignment on top of deterministic schedules")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for misalignment")
    p.add_argument("--poisson", type=float, default=0.0, help="Photons per pixel for Poisson noise (0 disables)")
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
    )
    args = p.parse_args()

    setup_logging(); log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    grid_d = meta.get("grid"); det_d = meta.get("detector")
    # Grid: infer from volume if not provided
    if grid_d is None:
        if "volume" in meta:
            nx, ny, nz = map(int, meta["volume"].shape)
            grid_d = {"nx": nx, "ny": ny, "nz": nz, "vx": 1.0, "vy": 1.0, "vz": 1.0}
        else:
            raise ValueError("Missing grid metadata and no ground-truth volume to infer from.")
    if det_d is None:
        # Fallback: infer from projections shape
        n_views, nv, nu = meta["projections"].shape
        det_d = {"nu": int(nu), "nv": int(nv), "du": 1.0, "dv": 1.0, "det_center": (0.0, 0.0)}
    if "volume" not in meta:
        raise ValueError("Input file does not contain a ground-truth volume under /entry/processing/tomojax/volume.")

    grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    det = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    thetas = meta.get("thetas_deg")
    geom_type = meta.get("geometry_type", "parallel")
    if geom_type == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)

    vol = jnp.asarray(meta["volume"], jnp.float32)
    n_views = int(len(thetas))

    # Build deterministic schedules if requested
    pert_specs: list[tuple[str, str, dict[str, str]]] = []
    if args.spec:
        pert_specs.extend(_load_spec_file(args.spec))
    for s in args.pert:
        pert_specs.append(_parse_pert(s))

    # Set base angles and params
    thetas_used = np.asarray(thetas, dtype=np.float32)
    angle_offset = np.zeros((n_views,), dtype=np.float32)
    params5_np = np.zeros((n_views, 5), dtype=np.float32)

    misalign_spec_dict = None
    if pert_specs:
        schedules, norm_spec = _build_schedules(jnp.asarray(thetas, jnp.float32), n_views, pert_specs)
        misalign_spec_dict = norm_spec
        # Apply angle offset
        angle_offset = schedules.get("angle", angle_offset).astype(np.float32)
        thetas_used = thetas_used + angle_offset
        # Apply DOF schedules
        # Rotations: alpha,beta,phi input assumed in degrees unless rad suffix was used;
        # schedules are raw numeric; interpret as degrees for rotations, convert to radians here.
        for i, k in enumerate(["alpha", "beta", "phi"]):
            if k in schedules:
                params5_np[:, i] += np.deg2rad(schedules[k].astype(np.float32))
        # Translations in pixels -> world units by du/dv
        if "dx" in schedules:
            params5_np[:, 3] += schedules["dx"].astype(np.float32) * float(det.du)
        if "dz" in schedules:
            params5_np[:, 4] += schedules["dz"].astype(np.float32) * float(det.dv)

    # Random per-view parameters (optional)
    if (not pert_specs) or args.with_random:
        rng = np.random.default_rng(args.seed)
        rot_scale = np.float32(np.deg2rad(float(args.rot_deg)))
        params5_np[:, 0] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)  # alpha
        params5_np[:, 1] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)  # beta
        params5_np[:, 2] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)  # phi
        params5_np[:, 3] += rng.uniform(-float(args.trans_px), float(args.trans_px), n_views).astype(np.float32) * float(det.du)
        params5_np[:, 4] += rng.uniform(-float(args.trans_px), float(args.trans_px), n_views).astype(np.float32) * float(det.dv)

    # Rebuild T_nom with possibly modified thetas
    if geom_type == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas_used)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas_used, tilt_deg=tilt_deg, tilt_about=tilt_about)

    # Recompute nominal poses with final geometry (possibly modified angles)
    T_nom = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(n_views)], axis=0)
    params5 = jnp.asarray(params5_np, jnp.float32)

    with _transfer_guard_ctx(args.transfer_guard):
        T_aug = T_nom @ jax.vmap(se3_from_5d)(params5)
        from ..core.projector import get_detector_grid_device
        det_grid = get_detector_grid_device(det)
        vm_project = jax.vmap(lambda T, v: forward_project_view_T(T, grid, det, v, use_checkpoint=True, det_grid=det_grid), in_axes=(0, None))
        proj = vm_project(T_aug, vol).astype(jnp.float32)

    # Optional noise
    if args.poisson and float(args.poisson) > 0:
        s = float(args.poisson)
        lam = np.clip(np.asarray(proj), 0.0, None) * s
        noisy = np.random.default_rng(args.seed + 1).poisson(lam=lam).astype(np.float32) / max(1e-6, s)
        proj = jnp.asarray(noisy, jnp.float32)

    save_nxtomo(
        args.out,
        projections=np.asarray(proj),
        thetas_deg=np.asarray(thetas_used),
        grid=grid.to_dict(),
        detector=det.to_dict(),
        geometry_type=geom_type,
        geometry_meta=meta.get("geometry_meta"),
        volume=np.asarray(vol),
        align_params=np.asarray(params5),
        angle_offset_deg=np.asarray(angle_offset) if pert_specs else None,
        misalign_spec=misalign_spec_dict,
        frame=str(meta.get("frame", "sample")),
    )
    logging.info("Wrote dataset: %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
