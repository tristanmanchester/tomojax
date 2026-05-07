# 2026-05-07 Phase 8 Anchored Preview Reconstruction

Ran supported-only `synth128_setup_global_tomo` diagnostics on the laptop GPU
with a 64^3 volume, 64 views, nuisance disabled, existing sidecar ingestion,
and setup-only stopped Schur updates with only `det_u_px` active.

## GPU

- JAX backend: `gpu`
- Selected JAX device: `cuda:0`
- Dataset:
  `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`

## Ladder

All runs used `--geometry-update-volume-source stopped_reconstruction`,
`--geometry-update-active-setup-parameters det_u_px`, and
`--geometry-update-pose-frozen`.

| Mode | Profile | Support | Init | TV scale | Preview filters | det_u RMSE px | theta RMSE rad | True vol/final geom | Final volume/init geom | Final volume/true geom | Classification | Schur accepted | Time s |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| baseline | reference | none | backprojection | 0 | raw | 7.25 | 0.0218166 | 0.884522 | 1.05091 | 1.40607 | `reconstruction_absorbed_geometry` | false | 18.9327 |
| support only | reference | cylindrical | backprojection | 0 | raw | 4.28274 | 0.0218166 | 0.483220 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 12.3957 |
| support + TV | reference | cylindrical | backprojection | 1 | raw | 4.28349 | 0.0218166 | 0.483309 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 12.3683 |
| support + TV + filters | reference | cylindrical | backprojection | 1 | continuation | 4.28358 | 0.0218166 | 0.483320 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 13.5487 |
| less geometry-aware init | reference | cylindrical | constant | 1 | continuation | 0.453199 | 0.0218166 | 0.0167246 | 1.87446 | 1.84208 | `independent_projection_losses_consistent` | true | 18.6021 |

Gate 1 passed for the constant-init anchored run:

- First stopped setup stage accepted a useful `det_u` movement.
- `det_u` RMSE improved from 7.25 px to 0.453199 px, below the 3 px gate.
- True-volume/final-geometry loss dropped from 0.884522 to 0.0167246.

Theta was intentionally frozen for this schedule; exact stopped theta recovery
is not required without an explicit orientation anchor.

## Fixed-Truth Regression

Fixed-truth supported-only Schur still passes after the anchored preview changes:

| Mode | Status | det_u RMSE px | theta RMSE rad | True vol/final geom | Device |
|---|---|---:|---:|---:|---|
| fixed truth, pose frozen | passed | 1.00136e-05 | 2.68284e-06 | 0 | `cuda:0` |

Artifact:

- `.artifacts/phase8_anchored_preview/runs/64_fixed_truth_pose_frozen_anchor_regression/`

## Detector Boundary Diagnostic

The current reference projector uses periodic detector-shift semantics. A
one-off diagnostic compared the successful stopped anchored run against
zero-fill and valid-overlap masked non-periodic shifts.

Artifact:

- `.artifacts/phase8_anchored_preview/detector_boundary_diagnostic.json`

Result:

- Current wrap losses: true-volume/true-geometry `0.0`,
  true-volume/final-geometry `0.0167246`,
  true-volume/initial-geometry `0.884522`.
- Valid-overlap masked zero-fill losses: true-volume/true-geometry `0.0`,
  true-volume/final-geometry `0.0191139`,
  true-volume/initial-geometry `1.01088`.
- Valid overlap fraction: `0.875`.

The non-periodic masked diagnostic penalizes the wrong detector shift more
strongly while preserving zero loss for true geometry. This supports keeping the
detector-boundary semantics question open for the next slice, after Gate 1.
