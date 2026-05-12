# 2026-05-12 Synthetic Tomography MVP Gates

This is the bounded productionization check for the two required tomography MVP
scenarios:

- `synth128_setup_global_tomo`
- `synth128_pose_random_extreme`

The runs intentionally used 32^3 volumes and 8 views on CPU as wiring and
artifact gates. They are not final 128^3 quality evidence. Both cases produced
sidecar datasets, benchmark results, benchmark reports, compare output, and
visual/diagnostic artifacts, but both failed their benchmark criteria.

## Commands

Setup-global oracle Schur geometry smoke:

```bash
env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache \
  uv run tomojax-align-auto-smoke \
    --out-dir .artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32 \
    --profile smoke32 \
    --synthetic-dataset synth128_setup_global_tomo \
    --views 8 \
    --geometry-update-volume-source fixed_synthetic_truth \
    --geometry-update-pose-frozen \
    --geometry-update-active-setup-parameters det_u_px,theta_offset_rad \
    --geometry-update-solver joint_schur
```

Pose-random fixed-truth Schur smoke:

```bash
env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache \
  uv run tomojax-align-auto-smoke \
    --out-dir .artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32 \
    --profile smoke32 \
    --synthetic-dataset synth128_pose_random_extreme \
    --views 8 \
    --geometry-update-volume-source fixed_synthetic_truth \
    --geometry-update-active-setup-parameters none \
    --geometry-update-active-pose-dofs phi_residual_rad,dx_px,dz_px \
    --geometry-update-alpha-beta-activate-at-level-factor 1 \
    --geometry-update-pose-trust-radius -1
```

Comparison report:

```bash
env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  uv run tomojax-synthetic-benchmark-compare \
    .artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32/benchmark_result.json \
    .artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32/benchmark_result.json \
    --out docs/benchmark_runs/2026-05-12-synthetic-tomo-mvp-comparison.md
```

## Results

| Scenario | Size | Views | Status | Criteria | Runtime | Classification |
|---|---:|---:|---|---|---:|---|
| `synth128_setup_global_tomo` | 32^3 | 8 | failed | 2 passed, 2 failed | 62.53 s | supported det_u/theta wiring works; axis/roll criteria fail |
| `synth128_pose_random_extreme` | 32^3 | 8 | failed | 0 passed, 1 failed, 2 not evaluated | 63.49 s | pose recovery/reporting remains incomplete |

## Setup-Global Evidence

Artifacts:

- Result JSON:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32/benchmark_result.json`
- Report:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32/benchmark_report.md`
- Sidecar dataset:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32/datasets/synth128_setup_global_tomo_32`
- Visual/diagnostic artifacts include `preview_slices/summary.json`,
  `detu_loss_curves.png`, `detu_gradient_curves.png`,
  `reduced_objective_curves.png`, `geometry_trace.csv`,
  `schur_diagnostics.json`, `final_volume.npy`, and
  `ground_truth_volume.npy`.

Metrics:

- `det_u_realized_rmse_px`: `0.002961397171020508`, passed `0.5 px`.
- `theta_realized_rmse_rad`: `0.0008581585078144579`, passed `0.1 deg`.
- `axis_error_rad`: `0.009439311165949147`, failed `0.1 deg`.
- `detector_roll_error_rad`: `0.011344640137963142`, failed `0.05 deg`.
- `final_residual`: `33.1478157043457`.
- `volume_nmse`: `368.46868896484375`.
- `schur_train_loss`: `0.0003279988595750183`.

Interpretation: the bounded fixed-truth setup smoke proves that the sidecar
ingestion, core projector, Schur loop, artifact writer, and criteria evaluation
are wired for the setup-global MVP. It does not prove the full setup-global
case passes: the current 8-view smoke leaves axis and detector-roll outside the
manifest tolerances, and final reconstruction quality is poor.

## Pose-Random Evidence

Artifacts:

- Result JSON:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32/benchmark_result.json`
- Report:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32/benchmark_report.md`
- Sidecar dataset:
  `.artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32/datasets/synth128_pose_random_extreme_32`
- Visual/diagnostic artifacts include `preview_slices/summary.json`,
  `pose_decomposition.csv`, `geometry_trace.csv`, `schur_diagnostics.json`,
  `reduced_objective_curves.png`, `final_volume.npy`, and
  `ground_truth_volume.npy`.

Metrics:

- `alpha_beta_rmse_rad`: `0.023499976541118742`, failed `0.25 deg`.
- `dx_dz_rmse_px`: not evaluated by the benchmark manifest path.
- `phi_rmse_deg_lt`: not evaluated by the benchmark manifest path.
- `final_residual`: `33.26259231567383`.
- `volume_nmse`: `301.8994445800781`.
- `schur_train_loss`: `0.00963222049176693`.

Interpretation: the pose-random smoke proves the fixed-truth pose path can run
and emit the required artifacts, but it is not a passing MVP gate. The current
result still exposes a pose-recovery and/or benchmark-evaluation capability gap:
alpha/beta fail tolerance, while dx/dz and phi are not evaluated despite the
run enabling pose DOFs.

## Comparison Artifact

The compare CLI output is recorded at:

- `docs/benchmark_runs/2026-05-12-synthetic-tomo-mvp-comparison.md`

It reports both required MVP scenarios as failed. This is an honest artifact
gate, not a success claim.

## Remaining Work

- Run the same two gates at 64^3 or 128^3 on CUDA when spending GPU time is
  appropriate.
- For `synth128_setup_global_tomo`, separate supported det_u/theta evidence
  from unsupported or underconverged axis/roll criteria.
- For `synth128_pose_random_extreme`, make dx/dz and phi recovery metrics
  evaluable when those DOFs are active, then address the failing pose recovery.
- Do not use the 32^3 8-view smoke as an alignment-quality gate; keep it as
  bounded wiring and artifact coverage only.
