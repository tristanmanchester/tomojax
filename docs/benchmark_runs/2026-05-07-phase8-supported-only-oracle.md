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
