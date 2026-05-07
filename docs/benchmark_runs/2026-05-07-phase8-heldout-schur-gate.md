# 2026-05-07 Phase 8 Held-Out Schur Gate

Reran the 128^3/256-view supported-only `synth128_setup_global_tomo` CUDA gate
after adding held-out acceptance for stopped-reconstruction Schur geometry
updates.

Command shape:

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_heldout_schur_gate/runs/128_supported_only_256views_heldout_schur_gpu \
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
| Anchored Schur volume | failed | `cuda:0` | 3:43.10 | 6075 | 0.299522 | 1.107017 | true | 4.127600 | 0.025110 | `reconstruction_absorbed_geometry` |
| Held-out Schur gate | failed | `cuda:0` | 3:46.25 | 6075 | 0.299566 | 1.106979 | true | 4.131494 | 0.024615 | `reconstruction_absorbed_geometry` |

Artifacts:

- Run directory:
  `.artifacts/phase8_heldout_schur_gate/runs/128_supported_only_256views_heldout_schur_gpu/`
- Command log:
  `.artifacts/phase8_heldout_schur_gate/logs/128_supported_heldout_schur_command.log`
- GPU memory samples:
  `.artifacts/phase8_heldout_schur_gate/logs/128_supported_heldout_schur_gpu_memory.csv`

## Interpretation

The held-out guard did not reject the bad stopped trajectory. Held-out loss also
improved slightly at each level, so a single held-out view with the same stopped
volume is not independent enough to detect the geometry absorption failure.

This supports moving to a stronger setup-validation objective or
gauge-regularized reconstruction. More stopped iterations, a different stopped
volume, and one-view held-out acceptance have now all failed to recover
setup-global theta/det_u at realistic scale.
