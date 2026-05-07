# Phase 8 128^3 Supported-Only Scale Gate

Date: 2026-05-07

Scope: realistic laptop-GPU scale gate for supported-only
`synth128_setup_global_tomo` before more benchmark/provenance cleanup.

## Dataset

- Scenario: `synth128_setup_global_tomo`
- Variant: supported-only setup-global
- Volume: 128^3
- Views: 256
- Detector: 160 x 160
- Nuisance fitting: disabled
- Dataset artifact:
  `.artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only/`

The first attempted fixed-truth command omitted `--views 256` and therefore
used the smoke CLI default of 4 views. It is not used as scale-gate evidence.

## Results

| Mode | JAX device | Result status | Manifest status | Peak GPU MB | Host RSS KB | Wall time | det_u RMSE px | theta RMSE rad | det_v RMSE px | Final residual | Volume NMSE | Schur accepted |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_synthetic_truth, pose frozen | `cuda:0` | failed | passed | 6071 | 2933348 | 2:55.81 | 2.28882e-05 | 4.10218e-06 | 0 | 618.138 | 4262.16 | true |
| stopped_reconstruction, anchored det_u-only setup | `cuda:0` | failed | failed | 6071 | 2775420 | 2:45.36 | 0.594401 | 0.0218166 | 0 | 4.06514 | 0.871336 | true |

## Artifacts

- Fixed-truth run:
  `.artifacts/phase8_supported128_scale_gate/runs/128_supported_only_256views_fixed_truth_reference_gpu/`
- Stopped anchored run:
  `.artifacts/phase8_supported128_scale_gate/runs/128_supported_only_256views_stopped_anchor_detu_gpu/`
- Comparison report:
  `.artifacts/phase8_supported128_scale_gate/benchmark_comparison_128_supported_only.md`
- GPU memory samples:
  `.artifacts/phase8_supported128_scale_gate/logs/fixed_truth_256views_gpu_memory.csv`
  and
  `.artifacts/phase8_supported128_scale_gate/logs/stopped_anchor_detu_gpu_memory.csv`

## Interpretation

The 128^3/256-view oracle Schur geometry update is viable on the laptop GPU and
recovers the supported setup DOFs under the existing manifest tolerances. Peak
observed device memory stayed near 6.1 GiB for both modes, so this scale gate did
not reproduce a 12 GiB allocation blow-up.

The stopped anchored det_u-only run improves the initial 14.5 px detector shift
to 0.594401 px but misses the strict 0.5 px manifest criterion. Theta remains at
the initial offset because this anchored diagnostic intentionally freezes theta
by activating only `det_u_px`. The result keeps the current blocker on stopped
reconstruction/volume gauge handling rather than setup/pose/theta convention
mapping for the oracle path.
