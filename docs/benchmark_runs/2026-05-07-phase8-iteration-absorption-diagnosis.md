# 2026-05-07 Phase 8 Iteration Absorption Diagnosis

This note classifies the current `synth128_setup_global_tomo` stopped-
reconstruction failure mode using existing 128^3 CUDA artifacts. It does not
change benchmark criteria or add new report fields.

## Evidence

All runs used the canonical `core_trilinear_ray` path on `cuda:0` where device
metadata was recorded.

| Run | Volume source | Iteration policy | Volume NMSE | Final residual | det_u RMSE px | theta RMSE rad | Axis error rad | Roll error rad | Classification | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Early-anchor stopped | stopped reconstruction | reference schedule | 0.375905 | 1.192685 | 4.234636 | 0.018991 | 0.005128 | 0.012903 | `reconstruction_absorbed_geometry` | `.artifacts/phase8_early_anchor/128_setup_global_stopped_cuda/` |
| Longer stopped | stopped reconstruction | 8/32/32 continuation | 0.212256 | 0.176570 | 4.227196 | 0.019514 | 0.012826 | 0.012509 | `reconstruction_absorbed_geometry` | `.artifacts/phase8_more_iterations_after_anchor/128_setup_global_stopped_8_32_32_cuda/` |
| True-geometry oracle | fixed synthetic truth | 32-iteration oracle reconstruction | n/a | n/a | 0.002894 | n/a | 0.000655 | 0.000366 | geometry passed | `.artifacts/phase8_true_geometry_recon_oracle/128_setup_global_true_recon32_schur_cuda/` |
| Constrained preview | stopped reconstruction | cylindrical support, TV scale 1 | 0.448293 | 1.989412 | 0.558276 | 0.018368 | 0.016179 | 0.012464 | `independent_projection_losses_consistent` | `.artifacts/phase8_constrained_preview/128_setup_global_stopped_cyl_tv1_cuda/` |

The longer stopped run materially improved reconstruction quality and projection
loss versus the early-anchor run:

- Volume NMSE improved from `0.375905` to `0.212256`.
- Final projection loss improved from `1.192685` to `0.176570`.
- The stopped volume preferred final geometry over true geometry:
  final-volume/final-geometry loss `0.176570` versus final-volume/true-geometry
  loss `0.589315`.

Geometry did not improve with the extra stopped reconstruction work:

- det_u stayed effectively unchanged: `4.234636` px to `4.227196` px.
- theta stayed outside tolerance: `0.018991` rad to `0.019514` rad.
- axis recovery worsened: `0.005128` rad to `0.012826` rad.
- detector roll stayed far outside tolerance: `0.012903` rad to `0.012509` rad.

The true-geometry oracle establishes that the Schur setup update can recover the
supported setup DOFs when the volume is sufficiently close to the correct gauge.
With 32 oracle reconstruction iterations, setup-global recovery passed.

## Interpretation

The current evidence matches the first branch of the operator hypothesis:

If more preview/reconstruction iterations improve final volume but geometry
stays bad, the reconstruction is absorbing geometry.

It does not support simply increasing stopped-reconstruction iterations as the
next fix. It also does not show that an unconstrained longer preview recovers
geometry. The constrained-preview probes are directionally useful because they
made det_u much closer, but they did not solve theta/axis/roll.

## Next Functional Slice

Prioritize a constrained early x-step for stopped reconstruction before the
full alternating update. The next implementation should keep the early volume
from freely absorbing setup error, then re-run the realistic setup-global CUDA
gate. Candidate constraints should be evaluated by geometry recovery first, not
only volume NMSE or projection loss.
