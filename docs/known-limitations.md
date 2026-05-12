# Known Limitations

Current production-shaped coverage is intentionally narrower than the full v2
research plan.

- Real laminography is validated on the k11 staged workflow evidence, not on a
  broad multi-scan corpus.
- `synth128_setup_global_tomo` passes a 128^3/16-view geometry gate with the
  diagnostic schedule, but the full 256-view manifest run is blocked by
  compile/orchestration time.
- `synth128_pose_random_extreme` runs and evaluates dx/dz/phi/alpha/beta at
  128^3/16 views, but currently fails the strict pose recovery thresholds.
- Nuisance fitting, object drift/deformation, and end-to-end Pallas defaults
  remain research/diagnostic surfaces rather than default production paths.
- JAX GPU selection currently requires exporting the venv NVIDIA library path
  before launching CUDA runs.
