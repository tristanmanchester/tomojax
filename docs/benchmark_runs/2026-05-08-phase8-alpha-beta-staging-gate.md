# 2026-05-08 Phase 8 Alpha/Beta Staging Gate

Ran the fixed-truth `synth128_pose_random_extreme` 128^3/256-view CUDA oracle
with alpha/beta delayed until the final continuation level. Pose trust clipping
was disabled to keep the translation-heavy pose solve from being globally
capped.

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_alpha_beta_staging/runs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --geometry-update-alpha-beta-activate-at-level-factor 1 \
  --geometry-update-pose-trust-radius -1 \
  --apply-synthetic-nuisance
```

## Result

| Metric | Phi/dx/dz no trust | Alpha/beta final |
|---|---:|---:|
| JAX device | `cuda:0` | `cuda:0` |
| Benchmark status | failed | failed |
| Wall time s | 213.10 | 215.97 |
| Volume NMSE | 0.177530 | 0.177530 |
| Final residual | 0.644243 | 0.642810 |
| `alpha_beta_rmse_rad` | 0.020097 | 0.012410 |
| `theta_realized_rmse_rad` | 0.125635 | 0.125796 |
| `det_u_realized_rmse_px` | 0.907141 | 0.901970 |
| `det_v_realized_rmse_px` | 1.005214 | 0.954342 |

Artifacts:

- Run directory:
  `.artifacts/phase8_alpha_beta_staging/runs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda/`
- Command log:
  `.artifacts/phase8_alpha_beta_staging/logs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda.log`

## Interpretation

Delayed alpha/beta activation is useful: it improves alpha/beta recovery and
slightly improves detector-shift realized errors without hurting the
translation-heavy solve. The benchmark still fails because phi/theta-realized
recovery remains poor. The next pose slice should focus on phi/theta recovery
or pose-angle acceptance, not alpha/beta staging.
