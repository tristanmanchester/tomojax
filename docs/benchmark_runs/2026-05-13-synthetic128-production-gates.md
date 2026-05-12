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
| `synth128_pose_random_extreme` | 128^3 | 256 | fixed-truth full-mask pose oracle with bounded polish | passed | 570.91 s benchmark | 1361 MiB | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe` |
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

Yes, after fixing pose-only Schur gauge carry and running the existing bounded
final pose-polish stage. The passing artifact is:

`.artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe`

The 128^3/256-view CUDA gate uses `fixed_synthetic_truth`, all five pose DOFs,
and no active setup parameters. Runtime summary reports
`total_wall_seconds = 570.9120919359848`; an in-flight `nvidia-smi` sample
reported `1361 MiB` used on the RTX 4070 Laptop GPU. It now passes the strict
pose criteria:

- `dx_dz_rmse_px = 0.000194251102341051`
- `phi_rmse_rad = 0.00014290254547410654`
- `alpha_beta_rmse_rad = 9.94198190694663e-06` against `< 0.25 deg`
- `det_u_realized_rmse_px = 0.00025558973015988315`
- `theta_realized_rmse_rad = 0.00014297660063987256`
- `det_v_realized_rmse_px = 0.00010428826557905991`

The blocker was not underconverged preview reconstruction, CPU fallback, memory,
or missing active DOFs. It was a pose-only solver state bug: accepted LM
iterations canonicalized mean `dx`/`phi` into setup and then repacked a
pose-only parameter vector that could not carry the setup gauge values into the
next iteration. Preserving those mean gauges until final canonicalisation makes
the existing final pose-polish stage productive instead of polishing the wrong
gauge.

Key diagnostics:

- `synth128_pose_random_16views_pose_gauge_fix_probe`: gauge fix alone improved
  det-u from `3.5144701261685487 px` to `0.03307838407963632 px`, theta from
  `0.23013369370493456 rad` to `0.016624980177947193 rad`, and phi from
  `0.22649383775390775 rad` to `0.015918240150468742 rad`.
- `synth128_pose_random_128_pose_gauge_fix`: the full 256-view gate with the
  gauge fix alone passed det-u/det-v/theta/dx-dz/phi but still failed
  alpha/beta at `0.008984073962632476 rad`.
- `synth128_pose_random_128_fullmask_polish64_probe`: the same full gate with
  64 final pose-polish updates passed all manifest pose criteria, including
  alpha/beta at `9.94198190694663e-06 rad` and phi at
  `0.00014290254547410654 rad`.

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
  --out-dir .artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe \
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
2. Keep the pose-only Schur gauge-carry fix and bounded final pose polish for
   `synth128_pose_random_extreme`; the 128^3/256-view oracle gate is now green.
3. Add real object-frame drift recovery before treating
   `synth128_thermal_object_drift` as passable.
4. Improve laminography axis/roll/theta recovery and det-v policy evidence
   before treating the laminography and combined hard cases as passable.
5. Do not treat the 4-view or 16-view diagnostics as alignment-quality gates.
   They are now wiring/triage checks only.
