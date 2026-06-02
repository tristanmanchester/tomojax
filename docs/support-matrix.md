# TomoJAX Support Matrix

Supported workflows and their CLI entrypoints.

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection | Supported | `tomojax inspect scan.nxs` |
| Dataset validation | Supported | `tomojax validate scan.nxs` |
| TIFF stack ingest | Supported | `tomojax ingest ./tiffs --angles angles.csv --du ... --dv ... --out scan.nxs` |
| NX/HDF5 preprocessing | Supported | `tomojax preprocess raw.nxs corrected.nxs` |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections corrected.nxs --format tiff-stack --flats ./flats --darks ./darks --angles angles.csv` |
| Reconstruction from corrected projections | Supported | `tomojax recon --data corrected.nxs --out recon.nxs` |
| Labelled reconstruction slice extraction | Supported | `tomojax slices --data recon.nxs --out quicklooks` |
| Per-projection 5-DOF pose alignment | Supported | `tomojax align --data corrected.nxs --mode pose --out aligned.nxs` |
| Detector-centre/COR alignment | Supported | `tomojax align --data corrected.nxs --mode cor --out aligned.nxs` |
| Expert mixed setup and pose alignment | Supported with explicit gauge policy | `tomojax align --data corrected.nxs --mode auto --gauge-policy anchor_mean --out aligned.nxs` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate --out synthetic_scan.nxs ...` |
| Python API reconstruction | Supported | `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` |

## Scope

Workflows outside the table above are research or expert diagnostics.

## Alignment interpretation

Alignment can improve reconstruction quality without every recovered parameter
being physically calibrated. This matters when a pose-only run absorbs setup
error.

- Use `--mode pose` as the first-line correction for per-projection sample
  motion.
- Use `--mode cor` when you need a detector-centre or centre-of-rotation
  correction that is physically interpretable.
- Use `--mode cor_then_pose` for detector-centre calibration followed by
  per-view pose refinement.
- Use `--mode auto` with `--gauge-policy anchor_mean` for combined setup and
  pose correction.
- Detector-v and sample-elevation reference shifts are not reliably
  recoverable.
