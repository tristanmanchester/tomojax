# TomoJAX Support Matrix

This matrix describes the public product workflows. Use these entries when you
describe supported TomoJAX behavior in docs, examples, and issue responses.

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection | Supported | `tomojax inspect scan.nxs` |
| Dataset validation | Supported | `tomojax validate scan.nxs` |
| TIFF stack ingest | Supported | `tomojax ingest ./tiffs --angles angles.csv --du ... --dv ... --out scan.nxs` |
| NX/HDF5 preprocessing | Supported | `tomojax preprocess raw.nxs corrected.nxs` |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections corrected.nxs --format tiff-stack --flats ./flats --darks ./darks --angles angles.csv` |
| Reconstruction from corrected projections | Supported | `tomojax recon --data corrected.nxs --out recon.nxs` |
| Per-projection 5-DOF pose alignment | Supported | `tomojax align --data corrected.nxs --mode pose --out aligned.nxs` |
| Detector-centre/COR alignment | Supported | `tomojax align --data corrected.nxs --mode cor --out aligned.nxs` |
| Expert mixed setup and pose alignment | Supported with explicit gauge policy | `tomojax align --data corrected.nxs --mode auto --gauge-policy anchor_mean --out aligned.nxs` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate --out synthetic_scan.nxs ...` |
| Public Python reconstruction smoke path | Supported | `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` facades |

## Claiming rules

Only the workflows in the table above are supported product entrypoints.
Describe workflows outside the matrix as research or expert diagnostics unless
they are promoted into the public CLI, public API, examples, and tests.

## Alignment interpretation

Alignment can improve reconstruction quality without proving that every
recovered parameter is physically calibrated. This distinction matters when a
pose-only run absorbs setup error.

- Use `--mode pose` as the first-line correction for per-projection sample
  motion.
- Use `--mode cor` when you need a detector-centre or centre-of-rotation
  correction that is physically interpretable.
- Use `--mode auto` with `--gauge-policy anchor_mean` only when you deliberately
  combine setup and pose correction.
- Don't present detector-v or sample-elevation reference shifts as normal
  recoverable alignment targets.
