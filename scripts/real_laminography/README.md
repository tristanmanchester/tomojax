# Real Laminography Scripts

These scripts support manual real-data laminography and TEM-grid diagnostic
runs. They assume local input files and write generated artifacts under ignored
output directories such as `runs/`.

- `run_real_lamino_native_setup_pose_256.py`: end-to-end native setup/pose
  diagnostic runner.
- `plot_tem_grid_xy_intensity_profiles.py`: line-profile plots for grid-aligned
  TEM XY slices.
- `render_tem_grid_aligned_orthos.py`: grid-aligned orthogonal slice rendering.
- `render_tem_grid_pose_3d.py`: 3D pose-correction diagnostic rendering.
