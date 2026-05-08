# Phase 8 Candidate-Refresh 128^3 Setup Scale Gate

Run:

```bash
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:... \
  /usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_candidate_refresh/runs/128_supported_only_256views_stopped_detu_only_candidate_refresh_cuda \
  --profile balanced --size 128 --views 256 \
  --synthetic-dataset synth128_setup_global_tomo \
  --synthetic-dataset-dir .artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only \
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

- `.artifacts/phase8_candidate_refresh/runs/128_supported_only_256views_stopped_detu_only_candidate_refresh_cuda/`

Outcome:

| Metric | Value |
|---|---:|
| Status | failed |
| Selected JAX device | `cuda:0` |
| JAX backend | `gpu` |
| Schur accepted | true |
| Initial det_u RMSE | 14.5 px |
| Final det_u RMSE | 6.58607 px |
| theta RMSE | 0.0218166 rad |
| Final residual | 2.07190 |
| Schur train loss | 2.75075 |
| Volume NMSE | 0.361283 |
| Runtime | 107.956 s artifact, 1:59.66 `/usr/bin/time` |
| Host max RSS | 2440012 KB |

The 128^3/256-view stopped det_u-only gate confirms the 64^3 result at
realistic scale: Schur accepts and det_u improves materially, but recovery is
still far outside tolerance and the failure remains
`reconstruction_absorbed_geometry`.
