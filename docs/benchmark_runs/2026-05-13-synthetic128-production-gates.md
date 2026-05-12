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
| `synth128_lamino_axis_roll_pose` | 128^3 | 256 | explicit setup+pose oracle diagnostic | failed laminography geometry | 500 s wrapper, 227.86 s benchmark | 1406 MiB | `.artifacts/production_hardening_synthetic/synth128_lamino_axis_roll_pose_128_classification` |
| `synth128_thermal_object_drift` | 128^3 | 256 | explicit setup+pose oracle diagnostic | failed object-motion recovery | 524 s wrapper, 174.19 s benchmark | 1404 MiB | `.artifacts/production_hardening_synthetic/synth128_thermal_object_drift_128_classification` |
| `synth128_combined_nuisance_jumps` | 128^3 | 320 | explicit setup+pose oracle diagnostic with nuisance applied | failed hard-case recovery | 614 s wrapper, 279.28 s benchmark | 1436 MiB | `.artifacts/production_hardening_synthetic/synth128_combined_nuisance_jumps_128_classification` |

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

Existing polish knobs were tested on 128^3/16-view diagnostics and did not
solve the pose gate:

- `synth128_pose_random_16views_final_pose_polish_probe`: 16 final pose-polish
  updates failed all three pose criteria and revealed a det-u setup leak in the
  final polish path.
- `synth128_pose_random_16views_pose_only_polish_probe`: after fixing final
  pose polish to respect configured setup parameters, pose-only polishing still
  failed all three criteria.
- `synth128_pose_random_16views_phi_polish_probe`: 16 phi-only polish updates
  also failed all three criteria.

These probes rule out simply adding more of the existing polish stages as the
next production fix.

## Remaining Original Scenarios

The remaining original synthetic128 cases were exercised as diagnostic
classification runs rather than marked green through unsupported terms.

### Laminography Axis/Roll/Pose

`synth128_lamino_axis_roll_pose` ran at 128^3/256 views on `cuda:0` with the
core trilinear backend. It failed 2/5 criteria:

- Passed: `det_u_error_px_lt`, `backend_policy`.
- Failed: `axis_error_deg_lt`, `detector_roll_error_deg_lt`.
- Not evaluated: `det_v_policy`, because the result does not yet contain
  unobservability-policy evidence.

The backend report correctly records a fallback for the calibrated
noncanonical detector grid policy. The blocker is laminography axis/roll
recovery and det-v observability reporting, not dataset generation.

### Thermal Object Drift

`synth128_thermal_object_drift` ran at 128^3/256 views on `cuda:0`. It failed
1/2 criteria:

- Passed: `core_solver = flags_object_motion_suspected`.
- Failed: `object_motion_enabled_tx_rmse_px_lt`, because object-frame motion
  recovery is not enabled. The zero-model `tx_rmse_px` was
  `7.318335768364758` against `< 1.5`.

This is a correctly classified unsupported model term: v2 can flag object
motion suspicion, but it does not yet solve object-frame drift.

### Combined Nuisance/Jumps

`synth128_combined_nuisance_jumps` ran at 128^3/320 views on `cuda:0` with
synthetic nuisance applied. It failed 3/6 criteria:

- Passed: `bad_views_flagged` and `pose_dx_dz_rmse_px_lt_except_jumps`.
- Failed: `axis_roll_error_deg_lt`, `det_u_error_px_lt`, and
  `theta_offset_error_deg_lt`.
- Not evaluated: `beats_current_default_nmse`, because no current-default
  baseline was supplied in the benchmark result.

The bad-view detector flagged 23 high-residual views, so bad-view observability
is present. The blocker is hard-case setup/axis/roll/theta recovery under
nuisance and jump structure, plus missing baseline comparison evidence.

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

Remaining scenario classification commands used the same CUDA prefix and
explicit `--synthetic-dataset` names, with active setup and pose DOFs enabled
where applicable. Artifacts are listed in the summary table.

## Next Fixes

1. Keep the compile-cache changes; they turned the setup-global 256-view gate
   from a runtime blocker into a passing gate with about 1.4 GiB peak sampled
   GPU memory.
2. Debug `synth128_pose_random_extreme` as an oracle fixed-volume pose solver
   failure. The highest-signal next step is to inspect accepted Schur pose
   updates across levels and add a native-resolution pose refinement policy that
   continues while loss is still reducing and pose criteria are still outside
   tolerance.
3. Add real object-frame drift recovery before treating
   `synth128_thermal_object_drift` as passable.
4. Improve laminography axis/roll/theta recovery and det-v policy evidence
   before treating the laminography and combined hard cases as passable.
5. Do not treat the 4-view or 16-view diagnostics as alignment-quality gates.
   They are now wiring/triage checks only.
