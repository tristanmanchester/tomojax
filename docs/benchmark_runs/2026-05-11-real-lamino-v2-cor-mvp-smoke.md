# 2026-05-11 Real Laminography v2 COR-MVP Smoke

Reference target report:
`runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/real_mvp_summary.json`

Command:

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/real_laminography/run_real_lamino_v2_cor_mvp.py \
  --input /home/tristan/projects/tomojax/runs/real-lamo-256/k11-54014_corrected_log_256cube.nxs \
  --out runs/real_lamino_v2_cor_mvp_smoke_20260511 \
  --reference-report runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/real_mvp_summary.json \
  --smoke \
  --overwrite
```

## Result

- Phase complete: true.
- Implemented stages completed:
  `00_baseline`, `01_setup_geometry/01_cor`, `06_cor_only_fista`.
- Planned stages:
  `01_setup_geometry/02_detector_roll`, `01_setup_geometry/03_axis_direction`,
  `02_pose_phi`, `03_pose_dx_dz`, `04_pose_polish`, `05_final`.
- COR-only FISTA final loss: 9383.828125.
- Baseline/COR-only volume shape match: true.
- Peak sampled GPU memory: 1375 MiB.

## Artifacts

Local report directory:
`runs/real_lamino_v2_cor_mvp_smoke_20260511/v2_cor_mvp_report/`

Key files:

- `real_mvp_summary.json`
- `real_mvp_summary.md`
- `real_mvp_residual_trace.csv`
- `real_mvp_geometry_trace.json`
- `publication/before_orthos.png`
- `publication/cor_only_orthos.png`

This is not the full real-MVP success gate. It is the first v2 vertical slice
that proves the real reference input can run through baseline, det_u setup, and
COR-only FISTA while preserving the report/artifact shape for the partial path.
