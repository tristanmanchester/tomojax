# Phase 8 Candidate-Refresh Stopped det_u Gate

Run:

```bash
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:... \
  /usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_candidate_refresh/runs/64_stopped_detu_only_candidate_refresh_cuda \
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

- `.artifacts/phase8_candidate_refresh/runs/64_stopped_detu_only_candidate_refresh_cuda/`

Outcome:

| Metric | Value |
|---|---:|
| Status | failed |
| Selected JAX device | `cuda:0` |
| JAX backend | `gpu` |
| Schur accepted | true |
| Initial det_u RMSE | 7.25 px |
| Final det_u RMSE | 2.87217 px |
| Final residual | 0.484702 |
| Schur train loss | 1.05838 |
| Volume NMSE | 0.269351 |
| Runtime | 30.0065 s artifact, 36.51 s `/usr/bin/time` |
| Host max RSS | 2162304 KB |

Candidate-refresh acceptance improves the carried stopped volume and residual
relative to the axis/gauge gate, but geometry recovery remains stuck at roughly
2.87 px det_u RMSE and the artifact still classifies the failure as
`reconstruction_absorbed_geometry`.
