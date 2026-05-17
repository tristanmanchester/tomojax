# Real Laminography Scripts

These scripts support manual real-data laminography and TEM-grid diagnostic
runs. They assume local input files and write generated artifacts under ignored
output directories such as `runs/`.

- `run_real_lamino_staged.py`: clean staged v2 real-laminography workflow.
- `summarize_real_lamino_report.py`: report builder for an existing staged run.
- `run_real_lamino_reference_regression.py`: internal reference-regression
  runner retained for behavior comparison against earlier evidence.
- `plot_tem_grid_xy_intensity_profiles.py`: line-profile plots for grid-aligned
  TEM XY slices.
- `render_tem_grid_aligned_orthos.py`: grid-aligned orthogonal slice rendering.
- `render_tem_grid_pose_3d.py`: 3D pose-correction diagnostic rendering.

Shared staged-run profile, stage-path, and reference-comparison contracts live
in `tomojax.bench.real_laminography_profiles`; scripts should import that
module rather than another file from this directory.
