# Phase 8 Hard x-Gauge Diagnostic

Run:

```bash
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:... \
  /usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_volume_gauge_projection/runs/64_stopped_detu_only_hard_x_gauge_cuda \
  --profile balanced --size 64 --views 64 \
  --synthetic-dataset synth128_setup_global_tomo \
  --synthetic-dataset-dir .artifacts/phase8_core_projector/datasets/synth128_setup_global_tomo_64_supported_only \
  --geometry-update-volume-source stopped_reconstruction \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters det_u_px \
  --preview-volume-support cylindrical \
  --preview-initialization backprojection \
  --preview-tv-scale 1.0 \
  --preview-residual-filter-mode continuation \
  --preview-center-l2-weight 0.02
```

Artifact:

- `.artifacts/phase8_volume_gauge_projection/runs/64_stopped_detu_only_hard_x_gauge_cuda/`

Outcome:

| Metric | Value |
|---|---:|
| Status | failed |
| Selected JAX device | `cuda:0` |
| JAX backend | `gpu` |
| Schur accepted | true |
| Final det_u RMSE | 2.87242 px |
| Final residual | 0.484713 |
| Schur train loss | 1.05838 |
| Volume NMSE | 0.269356 |
| Runtime | 30.1096 s artifact, 36.40 s `/usr/bin/time` |
| Host max RSS | 2175352 KB |

The hard stopped-volume x recentering diagnostic was not retained: it did not
improve det_u recovery or residual relative to the candidate-refresh gate, and
the failure remained `reconstruction_absorbed_geometry`.
