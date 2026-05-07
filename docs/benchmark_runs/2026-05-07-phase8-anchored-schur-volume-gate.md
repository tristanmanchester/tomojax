# 2026-05-07 Phase 8 Anchored Schur-Volume Gate

Reran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
after changing `constant_cylindrical_first_level` so later Schur geometry
updates reuse the constrained first-preview volume.

Command shape:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_anchored_schur_volume_gate/runs/128_supported_only_256views_anchor_schur_volume_gpu \
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

## Result

| Run | Status | Device | Wall time | Peak GPU MiB | Volume NMSE | Final residual | Schur accepted | det_u RMSE px | theta RMSE rad | Classification |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Previous constrained preview | failed | `cuda:0` | 3:41.61 | 6075 | 0.312830 | 1.104883 | true | 5.345676 | 0.024685 | `reconstruction_absorbed_geometry` |
| Anchored Schur volume | failed | `cuda:0` | 3:43.10 | 6075 | 0.299522 | 1.107017 | true | 4.127600 | 0.025110 | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_anchored_schur_volume_gate/runs/128_supported_only_256views_anchor_schur_volume_gpu/`
- Command log:
  `.artifacts/phase8_anchored_schur_volume_gate/logs/128_supported_anchor_schur_volume_command.log`
- GPU memory samples:
  `.artifacts/phase8_anchored_schur_volume_gate/logs/128_supported_anchor_schur_volume_gpu_memory.csv`

## Interpretation

Reusing the constrained first-preview volume for later Schur updates helped det_u
relative to the immediately previous constrained-preview run, but it is still
far outside the 0.5 px manifest tolerance and theta did not recover. The
failure remains classified as `reconstruction_absorbed_geometry`.

This rules out the simple anchored-volume policy as the full stopped
setup-global fix. The next implementation should move to a stronger
setup-validation objective or gauge-regularized reconstruction, rather than
only choosing a different stopped volume for Schur.
