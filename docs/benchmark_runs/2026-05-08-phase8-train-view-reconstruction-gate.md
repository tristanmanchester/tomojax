# 2026-05-08 Phase 8 Train-View Reconstruction Gate

Ran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
with preview reconstruction excluding the held-out validation view:
`preview_reconstruction_mask_source = "train_views"`.

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_train_view_reconstruction/runs/128_supported_only_256views_train_views_no_skip_gpu \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only \
  --geometry-update-volume-source stopped_reconstruction \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters theta_offset_rad,det_u_px \
  --geometry-update-active-pose-dofs dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 1 \
  --geometry-update-theta-activate-at-level-factor 1 \
  --preview-volume-support cylindrical \
  --preview-initialization zero \
  --preview-reconstruction-mask-source train_views \
  --preview-tv-scale 0.0 \
  --preview-residual-filter-mode raw \
  --preview-center-l2-weight 100.0 \
  --stopped-preview-policy standard
```

## Result

| Field | Value |
|---|---:|
| JAX device | `cuda:0` |
| Process status | succeeded |
| Benchmark status | failed |
| Wall time | 218.47 s |
| Host max RSS KB | 2900024 |
| Volume NMSE | 0.450992 |
| Final residual | 1.767721 |
| det_u RMSE px | 3.861245 |
| theta RMSE rad | 0.021822 |
| Held-out loss | 0.010941 |
| Projection-loss classification | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_train_view_reconstruction/runs/128_supported_only_256views_train_views_no_skip_gpu/`
- Command log:
  `.artifacts/phase8_train_view_reconstruction/logs/128_supported_only_256views_train_views_no_skip_gpu.log`

## Interpretation

The first train-view attempt allowed the coarse held-out check to skip finer
geometry updates despite failing manifest recovery. The implementation now
disables coarse early exit when `train_views` reconstruction is selected.

With all levels forced to run, train-view reconstruction still failed and
remained classified as `reconstruction_absorbed_geometry`. It improved over the
accidental early-exit run but did not beat the best center-gauge stopped run.
The held-out-view split alone is therefore not enough; the next functional
slice should make the geometry objective itself less dependent on a single
absorbed stopped volume.
