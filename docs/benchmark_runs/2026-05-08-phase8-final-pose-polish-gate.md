# Phase 8 Final Pose Polish Gate

Date: 2026-05-08

## Scope

Added an opt-in final pose polish stage for the `synth128_pose_random_extreme`
fixed-truth blocker. The stage opens global `det_u_px` with all five per-view
pose DOFs after the existing phi-only polish.

## Artifacts

- Passing process / failing benchmark gate:
  `.artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose48_restart_cuda/`
- Command log:
  `.artifacts/phase8_final_pose_polish/logs/pose_random_fixed_truth_phi16_final_pose48_restart_cuda.log`
- Direct diagnostic probes:
  `.artifacts/phase8_pose_translation_diagnostic/extra_schur_from_phi_polish.json`
  `.artifacts/phase8_pose_translation_diagnostic/all5_long_from_phi_polish.json`
  `.artifacts/phase8_pose_translation_diagnostic/all5_detu_from_phi_polish.json`
  `.artifacts/phase8_pose_translation_diagnostic/extra_from_full_final_pose.json`
  `.artifacts/phase8_pose_translation_diagnostic/extra_from_full_final_pose48.log`

## CUDA Gate

Command:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose48_restart_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --geometry-update-alpha-beta-activate-at-level-factor 1 \
  --geometry-update-pose-trust-radius -1 \
  --geometry-update-phi-polish-updates 16 \
  --geometry-update-final-pose-polish-updates 48 \
  --apply-synthetic-nuisance
```

Result:

- Status: failed benchmark criteria.
- Selected JAX device: `cuda:0`.
- Total wall time from artifact: `764.26` seconds.
- `/usr/bin/time` wall time: `12:59.04`.
- Host max RSS: `5963764` KB.
- Volume NMSE: `0.177530`.
- Final residual: `0.643207`.
- Schur train loss: `0.001048`.
- `alpha_beta_rmse_rad=0.001411`, passed.
- `theta_realized_rmse_rad=0.004287`, passed.
- `det_u_realized_rmse_px=0.558123`, failed.
- `det_v_realized_rmse_px=0.914853`, failed.
- Bad-view detector flagged view `255`.

## Interpretation

The slice confirms that the remaining fixed-truth pose-random failure is not
just a poor preview volume: with the true volume, extra Schur work drives
alpha/beta and theta below tolerance and lowers projection loss sharply.
However, the alternating artifact still leaves a single high-residual endpoint
view that dominates detector-shift RMSE. Direct restarted probes from the
written artifact can repair that outlier, so the next functional slice should
target robust per-view outlier/weak-view handling or deterministic restart
state differences in the final pose polish, rather than more report fields.
