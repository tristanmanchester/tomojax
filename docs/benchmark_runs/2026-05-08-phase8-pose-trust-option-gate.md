# 2026-05-08 Phase 8 Pose Trust-Option Gate

Ran the fixed-truth `synth128_pose_random_extreme` 128^3/256-view CUDA oracle
through `align-auto` with the new explicit pose trust-radius option disabled.

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_pose_trust_option/runs/pose_random_fixed_truth_phi_dxdz_no_trust_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --geometry-update-pose-trust-radius -1 \
  --apply-synthetic-nuisance
```

## Result

| Field | Value |
|---|---:|
| JAX device | `cuda:0` |
| Process status | succeeded |
| Benchmark status | failed |
| Wall time | 213.10 s |
| Host max RSS KB | 2803516 |
| Volume NMSE | 0.177530 |
| Final residual | 0.644243 |
| `alpha_beta_rmse_rad` | 0.020097 |
| `theta_realized_rmse_rad` | 0.125635 |
| `det_u_realized_rmse_px` | 0.907141 |
| `det_v_realized_rmse_px` | 1.005214 |
| Final `next_pose_trust_radius` | null |

Artifacts:

- Run directory:
  `.artifacts/phase8_pose_trust_option/runs/pose_random_fixed_truth_phi_dxdz_no_trust_cuda/`
- Command log:
  `.artifacts/phase8_pose_trust_option/logs/pose_random_fixed_truth_phi_dxdz_no_trust_cuda.log`

## Interpretation

The option works through the public CLI/artifact path and exposes a useful
translation-heavy pose mode. It reduces realized detector shift errors from
about 9 px in the fixed-truth baseline to about 1 px, but the benchmark still
fails because phi/theta and alpha/beta remain unresolved. This should remain
opt-in until a staged/angular pose acceptance policy is implemented.
