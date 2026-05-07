# Phase 8 Supported-Only Oracle

Date: 2026-05-07

## Scope

- Dataset: `synth128_setup_global_tomo` supported-only variant.
- Size/views: 64^3 volume, 64 views.
- Nuisance: disabled.
- Geometry update volume: `fixed_synthetic_truth`.
- Pose DOFs: frozen.
- Active setup DOFs: `theta_offset_rad`, `det_u_px`.
- Backend: JAX GPU, selected device `cuda:0`.

## Commands

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -)" \
JAX_PLATFORMS=cuda \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass \
  --profile balanced \
  --synthetic-dataset-dir .artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-pose-frozen
```

```bash
uv run tomojax-synthetic-benchmark-compare \
  .artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/benchmark_result.json \
  --out .artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only.md
```

## Result

Status: passed.

| Metric | Value |
|---|---:|
| det_u realised RMSE | 0.0890718 px |
| theta realised RMSE | 0.00109812 rad |
| final residual | 0.000185589 |
| volume NMSE | 0.576863 |
| geometry updates executed | 5 |
| total wall time | 15.7484 s |

Manifest criteria:

- `det_u_error_px_lt=0.5`: passed.
- `theta_offset_error_deg_lt=0.1`: passed after conversion to radians.

## Diagnosis

The first balanced oracle run improved geometry but failed because trust-radius
adaptation compared mean-loss actual reductions against raw residual-sum
predicted reductions. The tiny reduction ratio shrank setup trust after
accepted clipped steps, stopping `det_u_px` around 5 px instead of the true
scaled 7.25 px. Scaling predicted reduction by the number of data residual rows
fixed the trust adaptation scale and the same balanced oracle passed.

## Artifacts

- Dataset: `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`
- Passing run: `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/`
- Benchmark result: `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/benchmark_result.json`
- Benchmark report: `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/benchmark_report.md`
- Compare report: `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only.md`

## Next

Use this as the fixed-truth setup oracle baseline.

## Fixed-Truth Joint Follow-Up

The unconstrained joint setup+pose fixed-truth run failed by absorbing setup
motion into per-view pose:

- Run:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_baseline/`
- det_u realised RMSE: 6.72424 px.
- theta realised RMSE: 0.021352 rad.
- final setup: `det_u_px=0.526379`, `theta_offset_rad=0.00083023`.

A strong pose-prior diagnostic passed the same supported-only sidecar:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -)" \
JAX_PLATFORMS=cuda \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000 \
  --profile balanced \
  --synthetic-dataset-dir .artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-pose-prior-strength 1000000.0
```

| Metric | Value |
|---|---:|
| det_u realised RMSE | 0.0890279 px |
| theta realised RMSE | 0.00109136 rad |
| final residual | 0.000189463 |
| total wall time | 76.5889 s |

Artifacts:

- Passing joint run:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000/`
- Fixed-truth comparison:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_fixed_truth.md`

Interpretation: fixed-truth joint setup+pose can pass the supported-only case
when pose is staged/prior-constrained. The unconstrained joint failure is setup
absorption into per-view pose, so the production path still needs a principled
block-wise or staged trust policy before stopped-reconstruction can be judged.

## Block-Wise Trust Follow-Up

After implementing separate setup and pose trust scales, the unconstrained
fixed-truth joint run still failed:

- Run:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_block_trust/`
- det_u realised RMSE: 7.25 px.
- theta realised RMSE: 0.0218166 rad.
- First final-level step:
  `setup_trust_scale=0.617204`, `pose_trust_scale=0.0187305`.
- The candidate setup update pointed in the wrong detector direction:
  `[theta=-1.04e-04, det_u=-0.5]`, and the candidate was rejected.

Compare artifact:

- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_block_trust.md`

Interpretation: block-wise trust removes aggregate pose clipping as the blocker.
The remaining unconstrained joint blocker is setup/pose gauge coupling, likely
requiring staged pose activation or a zero-mean/anchored pose parameterisation.

## Staged Pose Activation Follow-Up

The code now supports explicit active pose DOF subsets and delayed activation by
continuation level. Two staged GPU diagnostics were run:

- All pose DOFs activated only at level factor 1:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1/`
- Detector pose only (`dx_px,dz_px`) activated only at level factor 1:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1_no_phi/`

Both failed strict criteria with det_u realised RMSE 0.583686 px and theta
realised RMSE 0.00940296 rad. The coarse pose-frozen setup updates nearly
recover setup, but the final pose-active candidate is rejected and leaves setup
outside tolerance.

Compare artifact:

- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_staged_pose.md`

Interpretation: staged pose activation is available but still needs an anchored
or zero-mean pose parameterisation, or final pose refinement must avoid changing
already verified setup.

## Zero-Mean Pose Step Follow-Up

The LM path now transfers mean pose steps into setup before trust scaling and
candidate evaluation.

Runs:

- Unconstrained zero-mean joint:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_pose_step/`
- Zero-mean with `phi_residual_rad` frozen:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi/`
- Reference schedule, zero-mean with `phi_residual_rad` frozen:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi_reference/`

Best non-hard-prior result:

| Metric | Value |
|---|---:|
| det_u realised RMSE | 0.201021 px |
| theta realised RMSE | 1.36732e-08 rad |
| final residual | 0 |
| total wall time | 102.388 s |

This passes the manifest criteria (`det_u < 0.5 px`, `theta < 0.1 deg`) but
misses the internal verification gate (`det_u < 0.2 px`) by about 0.001 px.

Compare artifact:

- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_zero_mean.md`

Interpretation: zero-mean pose step projection almost resolves fixed-truth joint
setup+pose without a hard pose prior. The remaining narrow issue is detector
shift accuracy when detector pose remains active.

## Stopped-Reconstruction Classification

Stopped reconstruction was rerun with the same strong pose prior that passes the
fixed-truth joint diagnostic:

- Run:
  `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000/`
- det_u realised RMSE: 7.25 px.
- theta realised RMSE: 0.0218166 rad.
- final residual: 0.367724.
- volume NMSE: 0.576871.

Compare artifact:

- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_stopped_reconstruction.md`

Interpretation: fixed-truth joint can pass, but stopped-reconstruction does not
move geometry under the same strong pose-prior settings. The next blocker is
reconstruction/volume gauge handling or reconstruction absorption of geometry.

## Reporting-Provenance Refresh

After separating Schur training loss from independent all-view projection
losses, the strong-pose-prior fixed-truth and stopped-reconstruction diagnostics
were rerun on `cuda:0`.

Runs:

- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_reporting/`
- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_reporting.md`

| Mode | Status | det_u RMSE px | theta RMSE rad | Final residual | Schur train loss | True vol/final geom | True vol/true geom | Classification | Time s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth` | passed | 0.0890679 | 0.00109202 | 1.39905 | 0.000189625 | 0.000809495 | 0 | `independent_projection_losses_consistent` | 75.3539 |
| `stopped_reconstruction` | failed | 7.25 | 0.0218166 | 1.05102 | 0.367724 | 0.884522 | 0 | `reconstruction_absorbed_geometry` | 78.2245 |

Interpretation: the refreshed artifacts confirm that the fixed-truth oracle is
geometrically consistent while stopped reconstruction is absorbing supported
setup error into the reconstructed volume/geometry gauge.
