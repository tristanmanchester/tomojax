# Phase 8 Axis/Gauge Stopped det_u Gate

Run:

```bash
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:... \
  /usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_axis_gauge/runs/64_stopped_detu_only_axis_fix_cuda \
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

- `.artifacts/phase8_axis_gauge/runs/64_stopped_detu_only_axis_fix_cuda/`

Outcome:

| Metric | Value |
|---|---:|
| Status | failed |
| Selected JAX device | `cuda:0` |
| JAX backend | `gpu` |
| Schur accepted | true |
| Initial det_u RMSE | 7.25 px |
| Final det_u RMSE | 2.87216 px |
| Final residual | 0.768342 |
| Volume NMSE | 0.333778 |
| Runtime | 24.7699 s artifact, 31.09 s `/usr/bin/time` |
| Host max RSS | 2107080 KB |

The axis/gauge correction materially improves det_u but does not pass the
0.5 px criterion. Projection-loss provenance still reports
`reconstruction_absorbed_geometry`, so the next functional slice should test
candidate-refresh acceptance or an explicit stopped-volume gauge constraint.
