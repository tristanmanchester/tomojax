# 2026-05-08 Phase 8 No-FISTA First-Preview Gate

Ran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
with `stopped_preview_policy = "constant_cylindrical_first_level_no_fista"`.
The first stopped-preview level used the existing constant cylindrical volume
but skipped FISTA before the first Schur update.

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_no_fista_first_preview/runs/128_supported_only_256views_no_fista_first_gpu \
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
  --preview-tv-scale 0.0 \
  --preview-residual-filter-mode raw \
  --preview-center-l2-weight 100.0 \
  --stopped-preview-policy constant_cylindrical_first_level_no_fista
```

## Result

| Field | Value |
|---|---:|
| JAX device | `cuda:0` |
| Process status | succeeded |
| Benchmark status | failed |
| Wall time | 178.47 s |
| Host max RSS KB | 2856892 |
| Volume NMSE | 0.491490 |
| Final residual | 2.400696 |
| det_u RMSE px | 1.808249 |
| theta RMSE rad | 0.021008 |
| Projection-loss classification | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_no_fista_first_preview/runs/128_supported_only_256views_no_fista_first_gpu/`
- Command log:
  `.artifacts/phase8_no_fista_first_preview/logs/128_supported_only_256views_no_fista_first_gpu.log`

## Interpretation

Skipping first-level FISTA improved det_u compared with the longer stopped
8/32/32 run (`4.227196` px to `1.808249` px), so constraining the early x-step
does affect the absorbed-geometry failure mode. It did not recover theta and it
made the final volume/residual worse, which means a constant unoptimized volume
is too crude as the geometry-update volume.

The next functional slice should use a constrained but informative early volume
or validation objective, not simply add more stopped reconstruction iterations.
