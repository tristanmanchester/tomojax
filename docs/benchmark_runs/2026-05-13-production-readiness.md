# TomoJAX v2 Production Readiness - 2026-05-13

## Status

TomoJAX v2 now has clean public entrypoints for the current staged
real-laminography workflow and the two mandatory synthetic tomography gates.
The package surface is production-shaped enough for review. The two mandatory
128^3 tomography gates are green; the remaining original synthetic scenarios
are honestly classified as research/Phase 8+ capability work.

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
uv run tomojax dev align-auto \
  --out-dir .artifacts/synthetic/setup_global_128 \
  --synthetic-case setup-global \
  --size 128 \
  --views 256
```

Synthetic pose-random:

```bash
uv run tomojax dev align-auto \
  --out-dir .artifacts/synthetic/pose_random_128 \
  --synthetic-case pose-random \
  --size 128 \
  --views 256
```

## Evidence Summary

| Area | Evidence | Status |
|---|---|---|
| Real laminography | k11 staged run summarized from retained historical artifacts | Validated real-data staged workflow |
| Setup-global synthetic | `.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache` | Passes 128^3/256-view setup/COR/roll/axis/theta gate |
| Pose-random synthetic | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe` | Passes 128^3/256-view dx/dz/phi/alpha/beta gate |
| Remaining original synthetic scenarios | `synth128_lamino_axis_roll_pose`, `synth128_thermal_object_drift`, `synth128_combined_nuisance_jumps` classification artifacts | Runnable and honestly reported, but still red/partial for unsupported or research-path functionality |

## Direct Gate Answers

Did `synth128_setup_global_tomo` recover setup/COR/roll/axis/theta at 128^3?
Yes. The full 128^3/256-view CUDA gate passed at
`.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache`.

Did `synth128_pose_random_extreme` recover per-view dx/dz/phi/alpha/beta at
128^3? Yes. The full 128^3/256-view CUDA gate passed at
`.artifacts/production_hardening_synthetic/synth128_pose_random_128_fullmask_polish64_probe`
after fixing pose-only Schur gauge carry, using the full fixed-truth alignment
mask, and using the bounded final pose-polish stage.

## What Remains Research Work

- Improve laminography axis/roll/theta recovery and det-v observability evidence
  for `synth128_lamino_axis_roll_pose`.
- Add real object-frame drift recovery before treating
  `synth128_thermal_object_drift` as passable.
- Improve hard-case setup/axis/theta recovery under nuisance and bad-view jumps.
- Keep nuisance fitting, object drift, and default Pallas fast paths out of the
  production claim until they have dedicated gates.

## Validation References

- Real-laminography public script naming: `4208767`.
- Public auto-alignment command naming: `8925648`.
- Synthetic128 gate report: `7b9deef`.
- Mandatory synthetic pose metric wiring: `8b08a59`.
- Schur compile/loss-cache setup-global gate: `91403c1`.
- Pose-only Schur gauge carry and pose-random gate: `dfa71ec`.
