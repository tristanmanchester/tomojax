# 2026-05-13 Synthetic128 Product Truth

This report records the honest status of the original `128^3` synthetic gates
after the overnight product-truth run on `vivobook-ts`.

The purpose is not to make every JSON green. The purpose is to separate:

- production-supported behavior,
- oracle/fixed-volume geometry evidence,
- real unsupported research work,
- and failures that were caused by runner/reporting plumbing.

## Artifact Root

Laptop artifact root:

```text
/home/tristan/projects/tomojax-v2-goal/.artifacts/product_truth_20260513
```

Synced local artifact root:

```text
.artifacts/product_truth_20260513
```

The runner used CUDA on the RTX 4070 Laptop GPU with:

```text
JAX_PLATFORMS=cuda,cpu
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Original Five Cases

| Case | Size/views | Artifact | Status | Honest interpretation |
|---|---:|---|---|---|
| `synth128_setup_global_tomo` | 128^3 / 256 | `.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache` | passed | Oracle/fixed-volume setup geometry pass |
| `synth128_pose_random_extreme` | 128^3 / 256 | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe` | passed | Oracle/fixed-volume pose geometry pass |
| `synth128_lamino_axis_roll_pose` | 128^3 / 256 | `.artifacts/product_truth_20260513/synth128_lamino_axis_roll_pose_detv_policy_rerun/run` | failed | det-u and det-v policy pass; laminography axis/roll still fail |
| `synth128_thermal_object_drift` | 128^3 / 256 | `.artifacts/product_truth_20260513/synth128_thermal_object_drift/run` | failed | object-frame motion solver is not implemented |
| `synth128_combined_nuisance_jumps` | 128^3 / 320 | `.artifacts/product_truth_20260513/synth128_combined_nuisance_jumps/run` | failed | bad views and det-u pass; axis/roll/theta and baseline comparison remain unsolved |

The first two are important geometry-solver evidence, but they are not
truth-free stopped-reconstruction production passes.

## Runner Fixes During This Run

The first thermal run failed before producing a benchmark result because the
overnight runner requested `det_v_px` on a dataset whose geometry declares
detector-v inactive. That was fixed rather than ignored:

- `tools/run_product_truth_overnight_20260513.sh` now records per-case exit
  status and can continue after a failing long case.
- The runner can resume and skip completed result artifacts.
- Thermal and combined cases no longer request inactive `det_v_px`.
- `det_v_policy` now treats `freeze_or_prior_required` weak-DOF evidence as a
  reported detector-v policy outcome.

The laminography case was rerun after that policy fix.

## Laminography Axis/Roll/Pose

Artifact:

```text
.artifacts/product_truth_20260513/synth128_lamino_axis_roll_pose_detv_policy_rerun/run/benchmark_result.json
```

Manifest criteria:

| Criterion | Status | Value | Threshold |
|---|---|---:|---:|
| `det_u_error_px_lt` | passed | `0.10245630932626788` | `1.0` |
| `det_v_policy` | passed | `0.2886772835720011` | `recovered_or_reported_unobservable` |
| `backend_policy` | passed | `1` | `calibrated_grid_fallback_explicit` |
| `axis_error_deg_lt` | failed | `0.19178020406063725 rad` | `0.002617993877991494 rad` |
| `detector_roll_error_deg_lt` | failed | `0.034477245845192656 rad` | `0.0017453292519943296 rad` |

Runtime:

```text
total_wall_seconds = 251.2185372140957
geometry_updates_executed = 17
reconstruction_calls = 3
```

Interpretation:

The stale det-v policy issue is fixed. This case remains a real solver failure:
laminography axis and detector roll are not recovered at the original threshold.

## Thermal Object Drift

Artifact:

```text
.artifacts/product_truth_20260513/synth128_thermal_object_drift/run/benchmark_result.json
```

Manifest criteria:

| Criterion | Status | Value | Threshold |
|---|---|---:|---:|
| `core_solver` | passed | `1` | `flags_object_motion_suspected` |
| `object_motion_enabled_tx_rmse_px_lt` | failed | `7.318335768364758 px` | `1.5 px` |

Geometry side evidence from the same run:

```text
det_u_realized_rmse_px = 0.3927611989442176
dx_dz_rmse_px = 0.3980283593505957
phi_rmse_rad = 0.00033506694966637827
alpha_beta_rmse_rad = 0.0005114427894105145
```

Runtime:

```text
total_wall_seconds = 156.7285065019969
geometry_updates_executed = 17
reconstruction_calls = 3
```

Interpretation:

The run is no longer blocked by an invalid active-DOF request. It correctly
flags object motion, but fails because v2 does not yet solve object-frame
thermal drift.

## Combined Nuisance/Jumps

Artifact:

```text
.artifacts/product_truth_20260513/synth128_combined_nuisance_jumps/run/benchmark_result.json
```

Manifest criteria:

| Criterion | Status | Value | Threshold |
|---|---|---:|---:|
| `bad_views_flagged` | passed | `23` | `true` |
| `det_u_error_px_lt` | passed | `0.6266117844067105` | `1.5` |
| `pose_dx_dz_rmse_px_lt_except_jumps` | passed | `0.4852793918818292` | `2.0` |
| `axis_roll_error_deg_lt` | failed | `0.4857224402168479 rad` | `0.003490658503988659 rad` |
| `theta_offset_error_deg_lt` | failed | `0.34698174498201306 rad` | `0.003490658503988659 rad` |
| `beats_current_default_nmse` | not evaluated | `null` | `true` |

Runtime:

```text
total_wall_seconds = 292.7987781780539
geometry_updates_executed = 17
reconstruction_calls = 3
```

Interpretation:

This is no longer a runner crash. It is a real hard-case failure: bad-view
flagging, det-u, and non-jump dx/dz work; axis/roll/theta under nuisance and
jumps remain research work, and the current-default NMSE comparison still needs
a baseline artifact before it can be evaluated.

## Stopped det-u Scout/Tangent Follow-up

Artifact:

```text
.artifacts/product_truth_20260513/stopped_detu_scout_tangent/run/summary.json
```

| Level | det_u RMSE px | Volume NMSE | Classification |
|---|---:|---:|---|
| `32^3` | `0.36958909034729004` | `0.7683126926422119` | `independent_projection_losses_consistent` |
| `64^3` | `0.665705680847168` | `0.07319390028715134` | `reconstruction_absorbed_geometry` |
| `128^3` | `1.7046318054199219` | `0.12031387537717819` | `reconstruction_absorbed_geometry` |

The stopped path improved versus the previous scout/tangent stopped artifact
at `128^3` (`1.924456 px -> 1.704632 px`, `0.218229 NMSE -> 0.120314 NMSE`),
but still fails detector-center recovery. Schur accepted the update at every
level, so this remains a reconstruction/gauge handoff problem rather than a
missing scalar geometry signal.

## Production Truth Summary

What is production-supported enough to claim:

- fixed-volume/oracle setup-global and pose-random geometry gates pass at
  `128^3 / 256 views`;
- real-data staged laminography has a production-facing report path and a
  positive final-vs-COR-only reconstruction-quality result;
- the benchmark harness now records runner failures and weak-DOF policy
  evidence more honestly.

What is not production-supported yet:

- truth-free stopped detector-center recovery at `128^3`;
- laminography axis/roll recovery at the original synthetic threshold;
- object-frame thermal drift recovery;
- combined nuisance/jump axis/roll/theta recovery;
- current-default NMSE comparison for the combined hard case.

