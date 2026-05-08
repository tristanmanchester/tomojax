# Phase 8 Phi-Only Polish Stage Gate

Date: 2026-05-08

## Scope

Validated the opt-in final `phi_residual_rad`-only Schur polish stage on the
canonical `synth128_pose_random_extreme` fixed-truth 128^3 CUDA gate.

## Command

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_phi_polish_stage/runs/pose_random_fixed_truth_phi_polish16_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --geometry-update-alpha-beta-activate-at-level-factor 1 \
  --geometry-update-pose-trust-radius -1 \
  --geometry-update-phi-polish-updates 16 \
  --apply-synthetic-nuisance
```

## Artifacts

- Run directory:
  `.artifacts/phase8_phi_polish_stage/runs/pose_random_fixed_truth_phi_polish16_cuda/`
- Command log:
  `.artifacts/phase8_phi_polish_stage/logs/pose_random_fixed_truth_phi_polish16_cuda.log`
- Key files: `benchmark_result.json`, `verification.json`,
  `alignment_summary.csv`, `schur_diagnostics.json`, `config_resolved.toml`.

## Result

- Status: failed benchmark criteria.
- Selected JAX device: `cuda:0`.
- Total wall time from artifact: `327.57` seconds.
- `/usr/bin/time` wall time: `5:40.07`.
- Host max RSS: `3655620` KB.
- Volume NMSE: `0.177530`.
- Final residual: `0.647845`.
- Schur train loss: `0.065966`.
- Final polish Schur accepted: `true`.
- `alpha_beta_rmse_rad=0.012410`.
- `theta_realized_rmse_rad=0.045132`.
- `det_u_realized_rmse_px=0.901970`.
- `det_v_realized_rmse_px=0.954342`.

## Interpretation

The stage promotes the direct diagnostic cleanly: the added polish row ran 16
phi-only updates, reduced train loss from `0.100273` to `0.065966`, and reduced
theta-realized error from the staged baseline `0.125796` rad to `0.045132` rad.
It does not recover detector translations or alpha/beta to tolerance, so
`synth128_pose_random_extreme` remains a functional blocker rather than a
reporting problem.
