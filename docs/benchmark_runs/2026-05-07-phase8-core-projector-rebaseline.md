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
- Stopped anchored GPU:
  `.artifacts/phase8_core_projector/runs/64_supported_only_stopped_anchor_gpu/`
- CPU smoke:
  `.artifacts/phase8_core_projector/runs/32_supported_only_fixed_truth_cpu/`

## Results

| Run | Device | Source | Status | det_u RMSE px | theta RMSE rad | Total time s |
|---|---|---|---|---:|---:|---:|
| 32^3 CPU smoke | `cpu:0` | fixed truth | failed | 1.62574 | 0.0155630 | n/a |
| 64^3 GPU balanced | `cuda:0` | fixed truth | failed | 6.75000 | 0.0203247 | 30.0770 |
| 64^3 GPU reference | `cuda:0` | fixed truth | failed | 7.12500 | 0.0224485 | 50.2916 |
| 64^3 GPU reference | `cuda:0` | stopped reconstruction | failed | 0.237177 | 0.0218166 | 48.7828 |

## Interpretation

The core operator path is wired and artifact provenance records
`core_trilinear_ray`, but fixed-truth setup recovery fails. That makes the next
blocker adapter/scaling/trust behavior under the real projector, not stopped
reconstruction quality. The stopped det_u-only run is close to the prior 0.2 px
Gate 3 threshold, but it cannot be accepted while fixed-truth recovery is still
failing under the same core operator.
