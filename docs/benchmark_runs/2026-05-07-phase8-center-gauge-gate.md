# 2026-05-07 Phase 8 Preview Center-Gauge Gate

Reran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
with nonzero preview center-of-mass gauge penalties.

Command shape:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
uv run tomojax-align-auto-smoke \
  --synthetic-dataset-dir .artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only \
  --synthetic-dataset synth128_setup_global_tomo \
  --profile reference \
  --geometry-update-volume-source stopped_reconstruction \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters theta_offset_rad,det_u_px \
  --geometry-update-active-pose-dofs none \
  --stopped-preview-policy constant_cylindrical_first_level \
  --preview-center-l2-weight <weight> \
  --preview-volume-support none \
  --preview-initialization backprojection \
  --preview-tv-scale 1.0 \
  --preview-residual-filter-mode continuation
```

## Result

| Run | Status | Device | Wall time | Peak GPU MiB | Volume NMSE | Final residual | Schur accepted | det_u RMSE px | theta RMSE rad | Classification |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Held-out Schur gate | failed | `cuda:0` | 3:46.25 | 6075 | 0.299566 | 1.106979 | true | 4.131494 | 0.024615 | `reconstruction_absorbed_geometry` |
| Center gauge w=10 | failed | `cuda:0` | 3:47.94 | 6075 | 0.282807 | 1.089416 | true | 3.589689 | 0.022927 | `reconstruction_absorbed_geometry` |
| Center gauge w=100 | failed | `cuda:0` | 3:50.27 | 6075 | 0.262462 | 1.047002 | true | 2.990294 | 0.025660 | `reconstruction_absorbed_geometry` |

Artifacts:

- Weight 10 run:
  `.artifacts/phase8_center_gauge_gate/runs/128_supported_only_256views_center_gauge_gpu/`
- Weight 100 run:
  `.artifacts/phase8_center_gauge_gate/runs/128_supported_only_256views_center_gauge_w100_gpu/`
- Logs and memory samples:
  `.artifacts/phase8_center_gauge_gate/logs/`

## Interpretation

The center-gauge penalty helps det_u and volume quality monotonically across
these two weights, but it does not solve setup-global recovery. det_u remains
well outside the 0.5 px tolerance, theta remains far outside the 0.1 degree
tolerance, and the projection-loss classification stays
`reconstruction_absorbed_geometry`.

This suggests the next functional step needs theta-specific setup validation or
orientation anchoring, not only a lateral volume-gauge penalty.
