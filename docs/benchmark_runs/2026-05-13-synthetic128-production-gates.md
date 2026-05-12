# 2026-05-13 Synthetic128 Production Gates

This report records the production-hardening pass on the original `128^3`
synthetic tomography gates. These are real `128^3` volume runs on the laptop
GPU, not 32^3 wiring checks.

## CUDA/JAX Setup

JAX selected `cuda:0` only when the CUDA wheel libraries from
`.venv/lib/python3.12/site-packages/nvidia/*/lib` were placed on
`LD_LIBRARY_PATH`. Without that explicit path, JAX failed CUDA plugin
initialization because cuSPARSE was not found, even though `nvidia-smi` saw the
RTX 4070 Laptop GPU.

Common CUDA prefix:

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
| `synth128_setup_global_tomo` | 128^3 | 16 | `diagnostic-fast` | passed | 302 s wrapper, 171.76 s benchmark | 760 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_after_schur_cache` |
| `synth128_setup_global_tomo` | 128^3 | 16 | `diagnostic-fast` after loss-cache fix | passed | 154 s wrapper, 33.26 s benchmark | 766 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_after_loss_cache` |
| `synth128_setup_global_tomo` | 128^3 | 256 | `diagnostic-fast` after loss-cache fix | passed | 500 s wrapper, 164.03 s benchmark | 1402 MiB | `.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache` |
| `synth128_pose_random_extreme` | 128^3 | 256 | `diagnostic-fast` after loss-cache fix | failed pose gate | 419 s wrapper, 139.94 s benchmark | 1402 MiB | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_after_loss_cache` |
| `synth128_pose_random_extreme` | 128^3 | 16 | `reference` diagnostic | failed, worse than diagnostic-fast | 183 s wrapper, 60.56 s benchmark | not sampled | `.artifacts/production_hardening_synthetic/synth128_pose_random_16views_reference_probe` |

Earlier baseline evidence kept for comparison:

- `synth128_setup_global_tomo` at 128^3/256 views previously stalled for more
  than 27 minutes with stable GPU memory and little GPU execution:
  `.artifacts/production_hardening_synthetic/synth128_setup_global_128`.
- `synth128_pose_random_extreme` at 128^3/16 views previously failed all pose
  criteria:
  `.artifacts/production_hardening_synthetic/synth128_pose_random_16views_compile_probe`.

## Setup-Global Answer

Did `synth128_setup_global_tomo` recover setup/COR/roll/axis/theta at `128^3`?

Yes. The mandatory 128^3/256-view CUDA gate now completes and passes all four
manifest geometry criteria:

- `det_u_realized_rmse_px = 4.482269287109375e-05` against `< 0.5`
- `theta_realized_rmse_rad = 6.989872584294871e-06` against `< 0.1 deg`
- `detector_roll_error_rad = 9.0500392390825e-06` against `< 0.05 deg`
- `axis_error_rad = 1.3220908589456338e-05` against `< 0.1 deg`

The runtime blocker was reduced by caching the streamed Schur normal-equation,
scalar loss, and per-view loss JAX programs inside each LM solve. The 16-view
compile probe dropped from 2447 compile lines to 1239, and `jit(scan)` lines
dropped from 867 to 289. That was enough to complete the full 256-view gate.

## Pose-Random Answer

Did `synth128_pose_random_extreme` recover per-view dx/dz/phi/alpha/beta at
`128^3`?

No. The mandatory 128^3/256-view CUDA gate runs to completion and evaluates all
required pose metrics, but fails `phi` and `alpha/beta`:

- `dx_dz_rmse_px = 0.10121920919147176` against `< 1.0`: passed
- `phi_rmse_rad = 0.08440607756112731` against `< 0.25 deg`: failed
- `alpha_beta_rmse_rad = 0.009437649551612661` against `< 0.25 deg`: failed

This is a solver/recovery failure, not a missing metric, unsupported active DOF,
CPU fallback, or memory failure:

- Active pose DOFs were `alpha_rad`, `beta_rad`, `phi_residual_rad`, `dx_px`,
  and `dz_px`.
- Active setup parameters were empty.
- The run used `fixed_synthetic_truth`, so reconstruction absorption is not the
  blocker for this oracle gate.
- The final Schur update was accepted and reduced loss, but one native-level
  update did not recover the strict pose criteria.
- A 16-view `reference` profile diagnostic did not fix under-iteration; it made
  recovery worse (`dx_dz_rmse_px = 2.030042186911155`,
  `phi_rmse_rad = 0.1737614744829108`,
  `alpha_beta_rmse_rad = 0.0186320877574164`).

The next functional fix should stay in pose solver conditioning/update policy:
the oracle fixed-volume pose gate needs a native-level pose refinement that can
continue reducing accepted pose residual without falling into setup/pose gauge
coupling or prematurely declaring the update small.

## Commands

Setup-global full-view gate:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run tomojax-align-auto \
  --out-dir .artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache \
  --synthetic-case setup-global --size 128 --views 256
```

Pose-random full-view gate:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run tomojax-align-auto \
  --out-dir .artifacts/production_hardening_synthetic/synth128_pose_random_128_after_loss_cache \
  --synthetic-case pose-random --size 128 --views 256
```

Pose-random under-iteration diagnostic:

```bash
env LD_LIBRARY_PATH="$CUDA_LIBS" JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run tomojax-align-auto \
  --out-dir .artifacts/production_hardening_synthetic/synth128_pose_random_16views_reference_probe \
  --synthetic-case pose-random --size 128 --views 16 --profile reference
```

## Next Fixes

1. Keep the compile-cache changes; they turned the setup-global 256-view gate
   from a runtime blocker into a passing gate with about 1.4 GiB peak sampled
   GPU memory.
2. Debug `synth128_pose_random_extreme` as an oracle fixed-volume pose solver
   failure. The highest-signal next step is to inspect accepted Schur pose
   updates across levels and add a native-resolution pose refinement policy that
   continues while loss is still reducing and pose criteria are still outside
   tolerance.
3. Do not treat the 4-view or 16-view diagnostics as alignment-quality gates.
   They are now wiring/triage checks only.
