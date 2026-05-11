# 2026-05-11 Real Laminography MVP Reference Report

Reference run:
`runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`

This slice records the real-data MVP success criterion as reconstruction-quality
improvement on the staged laminography workflow, not synthetic truth recovery.
The encoded staged path is:

baseline -> COR/det_u -> detector roll -> axis direction -> phi -> dx/dz ->
5DOF polish -> final recon.

## Result

- Pass: true
- Full staged final FISTA loss: 6438.1611328125
- COR-only FISTA loss: 6804.66845703125
- Absolute improvement: 366.50732421875
- Relative improvement: 0.05386115819353972
- Matching volume shape: true, `[256, 256, 96]`

## Artifacts

Generated local report directory:
`runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/`

Key files:

- `real_mvp_summary.json`
- `real_mvp_summary.md`
- `real_mvp_residual_trace.csv`
- `real_mvp_geometry_trace.json`
- `publication/before_orthos.png`
- `publication/cor_only_orthos.png`
- `publication/full_orthos.png`
- `publication/full_delta_xy_delta_xy_global_z209.png`

Truth metrics are intentionally marked not applicable for this real-data gate.
The COR-only path is retained only as the reference comparator.
