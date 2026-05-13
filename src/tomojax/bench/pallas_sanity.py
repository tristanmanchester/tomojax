from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.pallas_projector import forward_project_views_T_pallas
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device


def _rel_l2(a: jnp.ndarray, b: jnp.ndarray) -> float:
    denom = float(jnp.linalg.norm(b.ravel())) or 1.0
    return float(jnp.linalg.norm((a - b).ravel()) / denom)


def _make_volume(size: int) -> jnp.ndarray:
    vol = np.zeros((size, size, size), dtype=np.float32)
    a = max(2, size // 5)
    b = max(3, size // 4)
    vol[a : a + max(3, size // 4), b : b + max(3, size // 5), a : a + max(3, size // 6)] = 1.0
    vol[size // 2 : size // 2 + max(3, size // 5), a : a + max(3, size // 6), size // 2 :] = 0.5
    return jnp.asarray(vol, dtype=jnp.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe that Pallas forward projection reacts to changed inputs."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=45)
    parser.add_argument("--git-branch", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    grid = Grid(args.size, args.size, args.size, 1.0, 1.0, 1.0)
    detector = Detector(args.size, args.size, 1.0, 1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.linspace(0.0, 180.0, int(args.views), endpoint=False).astype(np.float32),
    )
    poses = stack_view_poses(geom, args.views)
    det_grid = get_detector_grid_device(detector)
    volume = _make_volume(args.size)
    scaled_volume = volume * jnp.float32(1.01)
    shifted_poses = poses.at[:, 0, 3].add(jnp.float32(0.5))

    def pallas_project(v: jnp.ndarray, p: jnp.ndarray = poses) -> jnp.ndarray:
        return forward_project_views_T_pallas(
            p,
            grid,
            detector,
            v,
            gather_dtype="fp32",
            det_grid=det_grid,
            tile_shape=(8, 16),
            num_warps=4,
            kernel_variant="auto",
            layout_variant="detector_vu",
            state_mode="inline",
        )

    def jax_project(v: jnp.ndarray, p: jnp.ndarray = poses) -> jnp.ndarray:
        return jax.vmap(
            lambda T: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=True,
                gather_dtype="fp32",
                det_grid=det_grid,
            )
        )(p)

    # Compile/cache both paths before the measured sanity calls.
    pallas_project(volume).block_until_ready()
    jax_project(volume).block_until_ready()

    start = time.perf_counter()
    base = pallas_project(volume)
    base.block_until_ready()
    base_runtime_sec = time.perf_counter() - start

    start = time.perf_counter()
    scaled = pallas_project(scaled_volume)
    scaled.block_until_ready()
    scaled_runtime_sec = time.perf_counter() - start

    start = time.perf_counter()
    shifted = pallas_project(volume, shifted_poses)
    shifted.block_until_ready()
    shifted_runtime_sec = time.perf_counter() - start

    oracle = jax_project(volume)
    oracle.block_until_ready()
    scaled_oracle = jax_project(scaled_volume)
    scaled_oracle.block_until_ready()

    base_sum = float(base.sum())
    scaled_sum = float(scaled.sum())
    scaled_sum_ratio = scaled_sum / base_sum if base_sum else None
    scaled_rel = _rel_l2(scaled, base)
    shifted_rel = _rel_l2(shifted, base)
    base_vs_jax = _rel_l2(base, oracle)
    scaled_vs_jax = _rel_l2(scaled, scaled_oracle)
    scaled_ratio_ok = scaled_sum_ratio is not None and abs(scaled_sum_ratio - 1.01) <= 5e-4
    passed = bool(
        math.isfinite(base_vs_jax)
        and math.isfinite(scaled_vs_jax)
        and base_vs_jax <= 1e-5
        and scaled_vs_jax <= 1e-5
        and scaled_rel > 1e-3
        and shifted_rel > 1e-3
        and scaled_ratio_ok
    )

    report: dict[str, Any] = {
        "benchmark": "tomojax_pallas_changed_input_sanity",
        "created_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "note": args.note,
            "git_branch": args.git_branch,
            "git_commit": args.git_commit,
        },
        "config": {
            "size": args.size,
            "views": args.views,
        },
        "status": "pass" if passed else "fail",
        "timing": {
            "base_runtime_sec": float(base_runtime_sec),
            "scaled_runtime_sec": float(scaled_runtime_sec),
            "shifted_pose_runtime_sec": float(shifted_runtime_sec),
        },
        "checks": {
            "base_sum": base_sum,
            "scaled_sum": scaled_sum,
            "scaled_sum_ratio": scaled_sum_ratio,
            "scaled_sum_ratio_ok": bool(scaled_ratio_ok),
            "scaled_rel_l2_vs_base": scaled_rel,
            "shifted_pose_rel_l2_vs_base": shifted_rel,
            "base_pallas_vs_jax_rel_l2": base_vs_jax,
            "scaled_pallas_vs_jax_rel_l2": scaled_vs_jax,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
