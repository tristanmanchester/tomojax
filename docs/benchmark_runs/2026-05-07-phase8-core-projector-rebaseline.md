# Phase 8 Core Projector Rebaseline

Date: 2026-05-07

Scope: replace the operational v2 rotate-and-sum projector path with the
existing core trilinear ray projector/backprojector and rebaseline supported-only
setup-global diagnostics.

## Artifacts

- Dataset:
  `.artifacts/phase8_core_projector/datasets/synth128_setup_global_tomo_64_supported_only/`
- Fixed-truth GPU reference:
  `.artifacts/phase8_core_projector/runs/64_supported_only_fixed_truth_reference_gpu/`
- Fixed-truth GPU full oracle:
  `.artifacts/phase8_core_projector/runs/64_supported_only_fixed_truth_full_oracle_gpu/`
- Stopped anchored GPU:
  `.artifacts/phase8_core_projector/runs/64_supported_only_stopped_anchor_unclipped_detu_gpu/`
- CPU smoke:
  `.artifacts/phase8_core_projector/runs/32_supported_only_fixed_truth_cpu/`

## Results

| Run | Device | Source | Status | det_u RMSE px | theta RMSE rad | Total time s |
|---|---|---|---|---:|---:|---:|
| 32^3 CPU smoke | `cpu:0` | fixed truth | failed | 1.62574 | 0.0155630 | n/a |
| 64^3 GPU balanced | `cuda:0` | fixed truth | failed | 6.75000 | 0.0203247 | 30.0770 |
| 64^3 GPU reference | `cuda:0` | fixed truth | failed | 7.12500 | 0.0224485 | 50.2916 |
| 64^3 GPU reference | `cuda:0` | fixed truth, full raw/no-prior oracle | passed | 1.43051e-06 | 1.06805e-07 | 52.0031 |
| 64^3 GPU reference | `cuda:0` | stopped reconstruction, original trust | det_u Gate 3 failed | 0.237177 | 0.0218166 | 48.7828 |
| 64^3 GPU reference | `cuda:0` | stopped reconstruction, det_u-only setup trust unclipped | det_u Gate 3 passed | 0.102502 | 0.0218166 | 42.1509 |

Stopped-volume Schur probe on the stopped run's final volume:

| Probe | det_u error px |
|---|---:|
| raw, no prior, setup trust 0.5 | 0.377899 |
| raw, no prior, setup trust 1.0 | 0.233153 |
| raw, no prior, no setup trust clip | 0.0611143 |

## Interpretation

The core operator path is wired and artifact provenance records
`core_trilinear_ray`. Fixed-truth geometry passes once the oracle Schur path uses
raw geometry residuals, disables the level metadata prior, and runs all scheduled
levels instead of coarse early-exit. The stopped det_u-only run now clears the
det_u Gate 3 target with setup trust unclipped for the pose-frozen det_u-only
update. Its benchmark manifest still marks theta failed because this diagnostic
intentionally freezes theta.
