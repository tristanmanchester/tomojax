"""Developer CLI for generating projection datasets with known misalignment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from tomojax.align.api import se3_from_5d
from tomojax.core import log_jax_env, setup_logging
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    stack_view_poses,
)
from tomojax.io import (
    JsonValue,
    build_geometry_from_dataset_metadata,
    load_projection_payload,
    normalize_json,
    save_projection_payload,
)

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type MisalignTransferGuardMode = Literal["off", "log", "disallow"]
type MisalignGeometry = ParallelGeometry | LaminographyGeometry
type FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class MisalignCommand:
    """Typed command plan for the developer misalignment generator."""

    data: str
    out: str
    rot_deg: float
    trans_px: float
    pert: tuple[str, ...]
    spec: str | None
    with_random: bool
    seed: int
    poisson: float
    progress: bool
    transfer_guard: MisalignTransferGuardMode


def _jnp_float32_array(value: object) -> jax.Array:
    return jnp.asarray(value, dtype=np.float32)  # pyright: ignore[reportUnknownMemberType]


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
    """Parse a --pert specification: dof:shape[:k=v[,k=v...]]."""
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
    thetas_deg: FloatArray,
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
        th = np.asarray(thetas_deg, dtype=np.float32)
        th0 = float(params.get("start_deg", float(np.min(th).item())))
        th1 = float(params.get("end_deg", float(np.max(th).item())))
        # pick closest indices bounding the window
        si = int(np.argmin(np.abs(th - th0)))
        ei = int(np.argmin(np.abs(th - th1)))
        if ei < si:
            si, ei = ei, si
        return si, ei
    return 0, n_views - 1


def _apply_linear(schedule: FloatArray, thetas_deg: FloatArray, params: dict[str, str]) -> None:
    """Apply a linear ramp inside a window to the given schedule array (in-place)."""
    n = len(schedule)
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


def _apply_sin_window(schedule: FloatArray, thetas_deg: FloatArray, params: dict[str, str]) -> None:
    """Apply a single-lobe sine window (0->amp->0) within a window."""
    n = len(schedule)
    si, ei = _window_indices_for_domain(thetas_deg, n, params)
    if ei <= si:
        return
    amp = float(_parse_number_with_unit(params.get("amp", "0"))[0])
    m = ei - si
    t = np.linspace(0.0, 1.0, m + 1, dtype=np.float32)
    vals = amp * np.sin(np.pi * t)  # 0 at edges, amp at center
    schedule[si : ei + 1] += vals


def _apply_step(schedule: FloatArray, thetas_deg: FloatArray, params: dict[str, str]) -> None:
    """Apply a step at angle or index.

    Params:
      - at_deg or at_index: where the step begins
      - to: set absolute value to this within the step window (computed relative to current)
      - delta: add this offset within the window
      - width_deg / width_index: optional duration; if missing, holds to the end
    """
    n = len(schedule)
    d = _domain_from_params(params)
    if d == "index":
        at = int(params.get("at_index", 0))
        at = max(0, min(n - 1, at))
    else:
        th = np.asarray(thetas_deg, dtype=np.float32)
        at_deg = float(_parse_number_with_unit(params.get("at", params.get("at_deg", "0")))[0])
        at = int(np.argmin(np.abs(th - at_deg)))
    if at >= n:
        return
    # Width / until
    if d == "index":
        if "width_index" in params:
            end = min(n - 1, at + int(params["width_index"]))
        elif "until_index" in params:
            end = max(at, min(n - 1, int(params["until_index"])))
        else:
            end = n - 1
    elif "width_deg" in params:
        width = float(_parse_number_with_unit(params["width_deg"])[0])
        th = np.asarray(thetas_deg, dtype=np.float32)
        target = float(th.item(at)) + width
        end = int(np.argmin(np.abs(th - target)))
        end = max(at, min(n - 1, end))
    elif "until_deg" in params:
        target = float(_parse_number_with_unit(params["until_deg"])[0])
        th = np.asarray(thetas_deg, dtype=np.float32)
        end = int(np.argmin(np.abs(th - target)))
        end = max(at, min(n - 1, end))
    else:
        end = n - 1
    # Delta or absolute target
    if "to" in params:
        tgt = float(_parse_number_with_unit(params["to"])[0])
        # Compute relative delta to reach absolute target
        delta = tgt - float(schedule.item(at))
    else:
        delta = float(_parse_number_with_unit(params.get("delta", "0"))[0])
    schedule[at : end + 1] += delta


def _apply_box(schedule: FloatArray, thetas_deg: FloatArray, params: dict[str, str]) -> None:
    """Apply a finite box pulse over the requested width/until window.

    `_apply_step` already supports finite windows via width/until parameters, so
    a box is just a bounded step. Trying to add a synthetic "step down" after the
    window creates an inverse pulse or a persistent tail instead of returning to
    baseline.
    """
    _apply_step(schedule, thetas_deg, params)


def _build_schedules(
    thetas_deg: FloatArray,
    n_views: int,
    perts: list[tuple[str, str, dict[str, str]]],
) -> tuple[dict[str, FloatArray], dict[str, list[dict[str, str]]]]:
    """Construct per-DOF schedules and return also a normalized spec for metadata."""
    schedules: dict[str, FloatArray] = {
        k: np.zeros((n_views,), np.float32) for k in ("angle", "alpha", "beta", "phi", "dx", "dz")
    }
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


def _load_spec_file(path: str) -> list[tuple[str, str, dict[str, str]]]:  # noqa: PLR0912
    import json

    with Path(path).open(encoding="utf-8") as f:
        data = cast("object", json.load(f))
    out: list[tuple[str, str, dict[str, str]]] = []
    # Two accepted layouts:
    # 1) { "dx": [{"kind":"step", ...}], "angle": [{...}], ... }
    # 2) { "schedules": [{"dof":"dx","kind":"step",...}, ...] }
    if isinstance(data, dict) and any(
        k in data for k in ("angle", "alpha", "beta", "phi", "dx", "dz")
    ):
        data_by_dof = cast("dict[str, object]", data)
        for dof, lst in data_by_dof.items():
            if dof not in ("angle", "alpha", "beta", "phi", "dx", "dz"):
                continue
            if not isinstance(lst, list):
                raise ValueError(f"Spec field for {dof} must be a list")
            items = cast("list[object]", lst)
            for item in items:
                if not isinstance(item, dict) or "kind" not in item:
                    raise ValueError(f"Spec items for {dof} must be dicts with 'kind'")

                item_map = cast("dict[str, object]", item)
                kind = str(item_map["kind"]).lower()
                params = {k: str(v) for k, v in item_map.items() if k != "kind"}
                out.append((dof, kind, params))
    elif isinstance(data, dict) and "schedules" in data and isinstance(data["schedules"], list):
        data_with_schedules = cast("dict[str, object]", data)
        schedules = cast("list[object]", data_with_schedules["schedules"])
        for it in schedules:
            if not isinstance(it, dict) or "dof" not in it or "kind" not in it:
                raise ValueError("Each schedule must have 'dof' and 'kind'")
            schedule_item = cast("dict[str, object]", it)
            dof = str(schedule_item["dof"]).lower()
            kind = str(schedule_item["kind"]).lower()
            params = {k: str(v) for k, v in schedule_item.items() if k not in ("dof", "kind")}
            out.append((dof, kind, params))
    else:
        raise ValueError("Unrecognized spec file schema")
    # Normalize aliases in dof
    normed: list[tuple[str, str, dict[str, str]]] = []
    for raw_dof, kind, params in out:
        normalized_dof = raw_dof
        if normalized_dof in ("x", "u"):
            normalized_dof = "dx"
        if normalized_dof in ("y", "v"):
            normalized_dof = "dz"
        normed.append((normalized_dof, kind, params))
    return normed


def _build_parser() -> argparse.ArgumentParser:
    """Build the developer misalignment generator parser."""
    p = argparse.ArgumentParser(
        description=(
            "Create a misaligned (and optionally noisy) dataset from a ground-truth NXtomo file."
        )
    )
    _ = p.add_argument(
        "--data",
        required=True,
        help="Input .nxs containing ground-truth volume and geometry",
    )
    _ = p.add_argument("--out", required=True, help="Output .nxs path")
    _ = p.add_argument(
        "--rot-deg",
        type=float,
        default=1.0,
        help="Max abs rotation per-axis (alpha,beta,phi) in degrees (used for --with-random)",
    )
    _ = p.add_argument(
        "--trans-px",
        type=float,
        default=10.0,
        help="Max abs translation in detector pixels (dx,dz) (used for --with-random)",
    )
    # New deterministic perturbation modes
    _ = p.add_argument(
        "--pert",
        action="append",
        default=[],
        help=(
            "Additive schedule spec: dof:shape[:k=v[,k=v...]]; dof in {angle,alpha,beta,phi,dx,dz}"
        ),
    )
    _ = p.add_argument(
        "--spec",
        type=str,
        default=None,
        help="JSON file with schedules; see docs/misalign_modes.md",
    )
    _ = p.add_argument(
        "--with-random",
        action="store_true",
        help="Combine random misalignment on top of deterministic schedules",
    )
    _ = p.add_argument("--seed", type=int, default=0, help="RNG seed for misalignment")
    _ = p.add_argument(
        "--poisson",
        type=float,
        default=0.0,
        help="Photons per pixel for Poisson noise (0 disables)",
    )
    _ = p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
    )
    _ = p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for diagnostics)"
        ),
    )
    return p


def _parse_command(argv: Sequence[str] | None) -> MisalignCommand:
    """Parse CLI arguments into a typed misalignment generation command."""
    args = _build_parser().parse_args(argv)
    return MisalignCommand(
        data=cast("str", args.data),
        out=cast("str", args.out),
        rot_deg=cast("float", args.rot_deg),
        trans_px=cast("float", args.trans_px),
        pert=tuple(cast("list[str]", args.pert)),
        spec=cast("str | None", args.spec),
        with_random=cast("bool", args.with_random),
        seed=cast("int", args.seed),
        poisson=cast("float", args.poisson),
        progress=cast("bool", args.progress),
        transfer_guard=cast("MisalignTransferGuardMode", args.transfer_guard),
    )


def _generate_misaligned_dataset(command: MisalignCommand) -> None:  # noqa: PLR0915
    """Generate a misaligned projection dataset from a typed command plan."""
    setup_logging()
    log_jax_env()
    if command.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    dataset = load_projection_payload(command.data)
    source_metadata = dataset.copy_metadata()
    if source_metadata.volume is None:
        raise ValueError(
            "Input file does not contain a ground-truth volume under "
            "/entry/processing/tomojax/volume."
        )

    source_volume: FloatArray = np.asarray(source_metadata.volume, dtype=np.float32)
    volume_shape = cast("tuple[int, int, int]", source_volume.shape)
    grid, det, base_geom = cast(
        "tuple[Grid, Detector, MisalignGeometry]",
        build_geometry_from_dataset_metadata(
            dataset.geometry_inputs(),
            apply_saved_alignment=False,
            volume_shape=volume_shape,
        ),
    )
    thetas: FloatArray = np.asarray(dataset.angles_deg, dtype=np.float32)
    vol = _jnp_float32_array(source_volume)
    n_views = len(thetas)

    # Build deterministic schedules if requested
    pert_specs: list[tuple[str, str, dict[str, str]]] = []
    if command.spec:
        pert_specs.extend(_load_spec_file(command.spec))
    pert_specs.extend(_parse_pert(s) for s in command.pert)

    # Set base angles and params
    thetas_used: FloatArray = np.asarray(thetas, dtype=np.float32)
    angle_offset = cast("FloatArray", np.zeros((n_views,), dtype=np.float32))
    params5_np = cast("FloatArray", np.zeros((n_views, 5), dtype=np.float32))

    misalign_spec_dict = None
    if pert_specs:
        schedules, norm_spec = _build_schedules(thetas, n_views, pert_specs)
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
    if (not pert_specs) or command.with_random:
        rng = np.random.default_rng(command.seed)
        rot_scale = math.radians(command.rot_deg)
        params5_np[:, 0] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
        params5_np[:, 1] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
        params5_np[:, 2] += rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
        random_dx = rng.uniform(-float(command.trans_px), float(command.trans_px), n_views)
        random_dz = rng.uniform(-float(command.trans_px), float(command.trans_px), n_views)
        params5_np[:, 3] += random_dx.astype(np.float32) * float(det.du)
        params5_np[:, 4] += random_dz.astype(np.float32) * float(det.dv)

    # Rebuild T_nom with possibly modified thetas
    thetas_used_list = [float(thetas_used.item(index)) for index in range(len(thetas_used))]
    if isinstance(base_geom, ParallelGeometry):
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas_used_list)
    else:
        geom = LaminographyGeometry(
            grid=grid,
            detector=det,
            thetas_deg=thetas_used_list,
            tilt_deg=base_geom.tilt_deg,
            tilt_about=base_geom.tilt_about,
        )

    # Recompute nominal poses with final geometry (possibly modified angles)
    T_nom = stack_view_poses(geom, n_views)
    params5 = _jnp_float32_array(params5_np)

    with transfer_guard_context(command.transfer_guard):
        T_aug = T_nom @ jax.vmap(se3_from_5d)(params5)
        from tomojax.core.projector import get_detector_grid_device

        det_grid = get_detector_grid_device(det)

        def project_one(T: jax.Array, v: jax.Array) -> jax.Array:
            return forward_project_view_T(T, grid, det, v, use_checkpoint=True, det_grid=det_grid)

        vm_project = cast(
            "Callable[[jax.Array, jax.Array], jax.Array]",
            jax.vmap(  # pyright: ignore[reportUnknownMemberType]
                project_one,
                in_axes=(0, None),
            ),
        )
        proj = vm_project(T_aug, vol).astype(np.float32)

    # Optional noise
    if command.poisson and float(command.poisson) > 0:
        s = float(command.poisson)
        lam = np.clip(np.asarray(proj), 0.0, None) * s
        noisy = np.random.default_rng(command.seed + 1).poisson(lam=lam).astype(np.float32) / max(
            1e-6, s
        )
        proj = _jnp_float32_array(noisy)

    save_meta = dataset.copy_metadata()
    save_meta.thetas_deg = np.asarray(thetas_used)
    save_meta.grid = grid.to_dict()
    save_meta.detector = det.to_dict()
    save_meta.geometry_type = "parallel" if isinstance(base_geom, ParallelGeometry) else "lamino"
    save_meta.volume = np.asarray(vol)
    save_meta.align_params = np.asarray(params5)
    save_meta.angle_offset_deg = np.asarray(angle_offset) if pert_specs else None
    save_meta.misalign_spec = cast(
        "dict[str, JsonValue] | None",
        None if misalign_spec_dict is None else normalize_json(misalign_spec_dict),
    )
    save_meta.frame = str(source_metadata.frame or "sample")
    save_projection_payload(
        command.out,
        projections=np.asarray(proj),
        metadata=save_meta,
    )
    logging.info("Wrote dataset: %s", command.out)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate a misaligned projection dataset from a ground-truth NXtomo file."""
    _generate_misaligned_dataset(_parse_command(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
