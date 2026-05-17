# Synthetic Tomography Workflows

The package-facing synthetic path is:

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 128 --ny 128 --nz 128 \
  --nu 128 --nv 128 \
  --n-views 128 \
  --phantom random_shapes
uv run tomojax recon synthetic_scan.nxs --out synthetic_recon.nxs
```

The original `128^3` setup and pose gates are developer evidence runs, not the
normal user workflow. They remain useful for reproducing solver evidence and
checking regressions.

## Setup-Global

```bash
uv run tomojax dev align-auto \
  --out-dir .artifacts/synthetic/setup_global_128 \
  --synthetic-case setup-global \
  --size 128 \
  --views 16
```

This developer evidence case exercises detector center, detector roll, axis
direction, and theta offset recovery for `synth128_setup_global_tomo`.

## Pose-Random

```bash
uv run tomojax dev align-auto \
  --out-dir .artifacts/synthetic/pose_random_128 \
  --synthetic-case pose-random \
  --size 128 \
  --views 16
```

This developer evidence case exercises per-view dx/dz/phi/alpha/beta recovery for
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
