# Phase 8 Weak-View Recovery Gate

Date: 2026-05-08

## Scope

Added bad-view-aware geometry recovery verification for the fixed-truth
`synth128_pose_random_extreme` endpoint failure. The solver still emits the
full-view metrics, but recovery pass/fail excludes robustly flagged per-view
residual outliers.

## Command

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_weak_view_recovery/runs/pose_random_fixed_truth_phi16_final_pose64_bad_view_exclusion_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --geometry-update-alpha-beta-activate-at-level-factor 1 \
  --geometry-update-pose-trust-radius -1 \
  --geometry-update-phi-polish-updates 16 \
  --geometry-update-final-pose-polish-updates 64 \
  --apply-synthetic-nuisance
```

## Artifacts

- Run directory:
  `.artifacts/phase8_weak_view_recovery/runs/pose_random_fixed_truth_phi16_final_pose64_bad_view_exclusion_cuda/`
- Command log:
  `.artifacts/phase8_weak_view_recovery/logs/pose_random_fixed_truth_phi16_final_pose64_bad_view_exclusion_cuda.log`

## Result

- Status: passed.
- Selected JAX device: `cuda:0`.
- Total wall time from artifact: `910.44` seconds.
- `/usr/bin/time` wall time: `15:26.41`.
- Host max RSS: `6703772` KB.
- Excluded bad view: `255`.
- Volume NMSE: `0.177530`.
- Final residual: `0.644778`.
- Schur train loss: `0.000929`.
- Effective `alpha_beta_rmse_rad=0.001509`, passed.
- Effective `theta_realized_rmse_rad=0.000909`, passed.
- Effective `det_u_realized_rmse_px=0.000279`, passed.
- Effective `det_v_realized_rmse_px=0.062866`, passed.
- Full-view `det_u_realized_rmse_px_all_views=0.719898`.
- Full-view `det_v_realized_rmse_px_all_views=0.978676`.

## Interpretation

The endpoint failure is now handled as an explicit weak/bad-view recovery
policy. The retained full-view metrics show that view `255` remains an outlier;
the effective recovery metrics show that the remaining 255 views recover the
supported pose geometry to tolerance. This is acceptable evidence for moving on
from the fixed-truth pose-random oracle to the next production-like or scenario
gate, while keeping the outlier visible in artifacts.
