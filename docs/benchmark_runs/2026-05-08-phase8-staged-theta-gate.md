# 2026-05-08 Phase 8 Staged Theta Gate

Reran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
with `theta_offset_rad` frozen until the final continuation level and preview
center-gauge weight `100.0`.

Command shape:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_staged_theta_gate/runs/128_supported_only_256views_staged_theta_gpu \
  --synthetic-dataset-dir .artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only \
  --synthetic-dataset synth128_setup_global_tomo \
  --profile reference \
  --geometry-update-volume-source stopped_reconstruction \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters theta_offset_rad,det_u_px \
  --geometry-update-theta-activate-at-level-factor 1 \
  --geometry-update-active-pose-dofs none \
  --stopped-preview-policy constant_cylindrical_first_level \
  --preview-center-l2-weight 100.0 \
  --preview-volume-support none \
  --preview-initialization backprojection \
  --preview-tv-scale 1.0 \
  --preview-residual-filter-mode continuation
```

## Result

| Run | Status | Device | Wall time | Peak GPU MiB | Volume NMSE | Final residual | Schur accepted | det_u RMSE px | theta RMSE rad | Classification |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Center gauge w=100 | failed | `cuda:0` | 3:50.27 | 6075 | 0.262462 | 1.047002 | true | 2.990294 | 0.025660 | `reconstruction_absorbed_geometry` |
| Staged theta final | failed | `cuda:0` | 3:48.57 | 6075 | 0.288342 | 1.091514 | true | 3.539086 | 0.023007 | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_staged_theta_gate/runs/128_supported_only_256views_staged_theta_gpu/`
- Command log:
  `.artifacts/phase8_staged_theta_gate/logs/128_supported_staged_theta_command.log`
- GPU memory samples:
  `.artifacts/phase8_staged_theta_gate/logs/128_supported_staged_theta_gpu_memory.csv`

## Interpretation

Staging `theta_offset_rad` to the final level improved theta relative to the
center-gauge w=100 run, but det_u and volume quality worsened and both manifest
criteria still failed. The failure remains `reconstruction_absorbed_geometry`.

This suggests theta and det_u need a joint setup-validation objective or a more
direct orientation anchor. Freezing theta early is not enough by itself.
