# Synthetic Tomography Workflows

The production-shaped synthetic entrypoint is:

```bash
uv run tomojax-align-auto --synthetic-case <case> --size 128 --out-dir <run-dir>
```

## Setup-Global

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/setup_global_128 \
  --synthetic-case setup-global \
  --size 128 \
  --views 16
```

This case exercises detector center, detector roll, axis direction, and theta
offset recovery for `synth128_setup_global_tomo`.

## Pose-Random

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/pose_random_128 \
  --synthetic-case pose-random \
  --size 128 \
  --views 16
```

This case exercises per-view dx/dz/phi/alpha/beta recovery for
`synth128_pose_random_extreme`.

## Artifacts

Each run writes:

- `benchmark_result.json`
- `benchmark_report.md`
- `final_geometry.json`
- recovered reconstruction arrays,
- verification and Schur diagnostics.

Current `128^3` evidence is in
[`docs/benchmark_runs/2026-05-13-synthetic128-production-gates.md`](benchmark_runs/2026-05-13-synthetic128-production-gates.md).
