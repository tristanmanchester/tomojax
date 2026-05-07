# 2026-05-07 Phase 8 Constrained Stopped-Preview Gate

Ran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
with the new stopped-preview policy:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_constrained_stopped_gate/runs/128_supported_only_256views_stopped_constant_cyl_gpu \
  --synthetic-dataset-dir .artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only \
  --synthetic-dataset synth128_setup_global_tomo \
  --profile reference \
  --geometry-update-volume-source stopped_reconstruction \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters theta_offset_rad,det_u_px \
  --geometry-update-active-pose-dofs none \
  --stopped-preview-policy constant_cylindrical_first_level \
  --preview-volume-support none \
  --preview-initialization backprojection \
  --preview-tv-scale 1.0 \
  --preview-residual-filter-mode continuation
```

The first attempt used `JAX_PLATFORM_NAME=gpu` without the venv NVIDIA library
paths and failed before running because the JAX CUDA plugin could not load
cuSPARSE. The successful command used the same `LD_LIBRARY_PATH`/`JAX_PLATFORMS`
pattern as previous CUDA gates.

## Result

| Field | Value |
|---|---:|
| JAX device | `cuda:0` |
| Process status | succeeded |
| Benchmark status | failed |
| Wall time | 3:41.61 |
| Host max RSS KB | 2931140 |
| Peak sampled GPU MiB | 6075 |
| Volume NMSE | 0.312830 |
| Final residual | 1.104883 |
| Schur accepted | true at final level |
| det_u RMSE px | 5.345676 |
| theta RMSE rad | 0.024685 |
| det_v RMSE px | 0 |
| Projection-loss classification | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_constrained_stopped_gate/runs/128_supported_only_256views_stopped_constant_cyl_gpu/`
- Command log:
  `.artifacts/phase8_constrained_stopped_gate/logs/128_supported_stopped_constant_cyl_command.log`
- GPU memory samples:
  `.artifacts/phase8_constrained_stopped_gate/logs/128_supported_stopped_constant_cyl_gpu_memory.csv`

## Interpretation

The simple first-preview constraint is not sufficient. It made the full
theta+det_u stopped run worse than the previous det_u-only anchored stopped
diagnostic: det_u ended at `5.345676` px here versus `0.594401` px for the
det_u-only anchored run, and theta still missed the manifest tolerance.

Because the fixed-truth oracle already passes at this scale, the blocker remains
stopped reconstruction/volume gauge handling. The next functional slice should
not be more iterations or report polishing; it should implement a stronger
setup-validation or reconstruction-gauge constraint that prevents the stopped
volume from choosing a geometry-dependent gauge before theta/det_u recovery.
