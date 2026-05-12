# 2026-05-13 Synthetic128 Production Gates

This report records the first production-hardening pass on the original `128^3` synthetic tomography gates. These are real `128^3` volume runs on the laptop GPU, not 32^3 smoke results.

## CUDA/JAX Setup

JAX selected `cuda:0` only when the CUDA wheel libraries from `.venv/lib/python3.12/site-packages/nvidia/*/lib` were placed on `LD_LIBRARY_PATH`. Without that explicit path, JAX failed CUDA plugin initialization because cuSPARSE was not found, even though `nvidia-smi` saw the RTX 4070 Laptop GPU.

Common CUDA prefix used for the successful runs:

```bash
CUDA_LIBS=$(python3 - <<'PY'
from pathlib import Path
base = Path('.venv/lib/python3.12/site-packages/nvidia')
print(':'.join(str(p / 'lib') for p in base.iterdir() if (p / 'lib').is_dir()))
PY
)
export LD_LIBRARY_PATH="$CUDA_LIBS"
export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Summary

| Case | Size | Views | Profile | Status | Runtime | Peak GPU Mem | Artifact |
|---|---:|---:|---|---|---:|---:|---|
| `synth128_setup_global_tomo` | 128^3 | 16 | `diagnostic-fast` / internal `smoke32` | passed geometry gate | 5:36 wall, 203.37 s reported | 771 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_compile_probe` |
| `synth128_setup_global_tomo` | 128^3 | 16 | `fast` / internal `lightning` | failed geometry gate | 3:05 wall, 53.85 s reported | 755 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_fast_probe` |
| `synth128_setup_global_tomo` | 128^3 | 16 | `balanced` | failed geometry gate | 3:57 wall, 87.39 s reported | 759 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_balanced_probe` |
| `synth128_setup_global_tomo` | 128^3 | 256 | `diagnostic-fast` / internal `smoke32` | terminated: host CPU/runtime blocker | >27 min before termination | 1275 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_128` |
| `synth128_pose_random_extreme` | 128^3 | 16 | `diagnostic-fast` / internal `smoke32` | failed pose gate | 5:14 wall, 196.87 s reported | 769 MiB | `.artifacts/production_hardening_synthetic/synth128_pose_random_16views_compile_probe` |

## Setup-Global Answer

Did `synth128_setup_global_tomo` recover setup/COR/roll/axis/theta at `128^3`?

Partially. At `128^3` with 16 views and the diagnostic schedule, yes:

- `det_u_realized_rmse_px = 0.00014972686767578125` against `< 0.5`
- `theta_realized_rmse_rad = 6.784388294267529e-06` against `< 0.1 deg`
- `detector_roll_error_rad = 7.4714474751092635e-06` against `< 0.05 deg`
- `axis_error_rad = 1.4542516068081725e-05` against `< 0.1 deg`

The full 256-view manifest count has not completed. It was stopped after more than 27 minutes because GPU memory was stable at about 1.25 GiB but `nvidia-smi pmon` showed 0% SM utilization while Python consumed about 123% CPU. This is a compile/orchestration runtime blocker, not a VRAM/OOM failure.

The lower-update `fast` and `balanced` schedules are not valid substitutes: both failed all four setup-global manifest criteria at `128^3`/16 views.

## Pose-Random Answer

Did `synth128_pose_random_extreme` recover per-view dx/dz/phi/alpha/beta at `128^3`?

No. The `128^3`/16-view diagnostic run now evaluates all required pose metrics, but fails all three manifest criteria:

- `dx_dz_rmse_px = 1.347414703523529` against `< 1.0`
- `phi_rmse_rad = 0.22649383775390775` against `< 0.25 deg`
- `alpha_beta_rmse_rad = 0.018122811693180637` against `< 0.25 deg`

This is no longer an artifact/reporting hole. The dx/dz and phi metrics are present in `benchmark_result.json` and `benchmark_report.md`; the solver did not recover the pose gate at this scale.

## Compile/Runtime Diagnosis

The 128^3/16-view diagnostic setup run completed, but `JAX_LOG_COMPILES=1` recorded 2467 JAX compilations. The dominant compile names were:

- `scan`: 884 compilations, about 167.6 s cumulative compile time
- `cond`: 814 compilations, about 43.5 s cumulative compile time

The matching pose-random run recorded 2445 compilations, with similar `scan`/`cond` dominance. This explains why the process can look CPU-bound despite selecting CUDA: the current reference Schur/reconstruction path is repeatedly compiling many small JAX programs around GPU kernels.

## Commands

Setup-global diagnostic lower-view gate:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_LOG_COMPILES=1 \
  uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/production_hardening_synthetic/synth128_setup_global_16views_compile_probe \
  --synthetic-case setup-global --size 128 --views 16
```

Pose-random diagnostic lower-view gate:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_LOG_COMPILES=1 \
  uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/production_hardening_synthetic/synth128_pose_random_16views_compile_probe \
  --synthetic-case pose-random --size 128 --views 16
```

Setup-global full-view attempt:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/production_hardening_synthetic/synth128_setup_global_128 \
  --synthetic-case setup-global --size 128 --views 256
```

## Next Fixes

1. Reduce the compile storm in the Schur normal-equation path. The immediate target is repeated `jit(scan)`/`jit(cond)` compilation from the streamed joint Schur/reconstruction path.
2. Rerun `synth128_setup_global_tomo` at 128^3/256 views after the compile/orchestration fix, with the diagnostic schedule unless a lower-update schedule can be made to pass.
3. Debug `synth128_pose_random_extreme` as a solver/recovery failure: all five pose DOFs are active and all required pose metrics are now evaluated, but the 128^3 lower-view gate still fails.
