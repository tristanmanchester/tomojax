# TomoJAX v2 Production Readiness - 2026-05-13

## Status

TomoJAX v2 now has clean public entrypoints for the current staged
real-laminography workflow and the two mandatory synthetic tomography gates.
The package surface is production-shaped enough for review, but not all
tomography gates are green.

## Clean Commands

Real laminography:

```bash
uv run python scripts/real_laminography/run_real_lamino_staged.py \
  --input /path/to/scan.nxs \
  --out runs/real_lamino_staged_run \
  --profile staged-lamino \
  --overwrite
```

Synthetic setup-global:

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/setup_global_128 \
  --synthetic-case setup-global \
  --size 128 \
  --views 16
```

Synthetic pose-random:

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/pose_random_128 \
  --synthetic-case pose-random \
  --size 128 \
  --views 16
```

## Evidence Summary

| Area | Evidence | Status |
|---|---|---|
| Real laminography | k11 staged run summarized from retained historical artifacts | Validated real-data staged workflow |
| Setup-global synthetic | `.artifacts/production_hardening_synthetic/synth128_setup_global_16views_compile_probe` | Passes 128^3/16-view geometry gate |
| Setup-global full manifest | `.artifacts/production_hardening_synthetic/synth128_setup_global_128` | Blocked by compile/orchestration runtime before useful GPU work |
| Pose-random synthetic | `.artifacts/production_hardening_synthetic/synth128_pose_random_16views_compile_probe` | Runs and evaluates all pose metrics, but fails recovery thresholds |

## Direct Gate Answers

Did `synth128_setup_global_tomo` recover setup/COR/roll/axis/theta at 128^3?
Yes for the 16-view diagnostic gate. The full 256-view manifest run has not
completed and remains blocked by runtime/orchestration behavior.

Did `synth128_pose_random_extreme` recover per-view dx/dz/phi/alpha/beta at
128^3? No. The run now evaluates dx/dz, phi, alpha, and beta honestly, but the
solver misses the strict pose thresholds.

## What Remains Research Work

- Reduce the repeated JAX `scan`/`cond` compile storm in the Schur/reconstruction
  path, then rerun the 256-view setup-global gate.
- Diagnose the pose-random recovery failure as solver/reconstruction behavior,
  not as a reporting gap.
- Keep nuisance fitting, object drift, and default Pallas fast paths out of the
  production claim until they have dedicated gates.

## Validation References

- Real-laminography public script naming: `4208767`.
- Public auto-alignment command naming: `8925648`.
- Synthetic128 gate report: `7b9deef`.
- Mandatory synthetic pose metric wiring: `8b08a59`.
