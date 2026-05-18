# TomoJAX Support Matrix

This matrix describes the product spine.

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection | Supported | `tomojax inspect scan.nxs` |
| Dataset validation | Supported | `tomojax validate scan.nxs` |
| TIFF stack ingest | Supported | `tomojax ingest ./tiffs --angles angles.csv --du ... --dv ... --out scan.nxs` |
| NX/HDF5 preprocessing | Supported | `tomojax preprocess raw.nxs corrected.nxs` |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections corrected.nxs --format tiff-stack --flats ./flats --darks ./darks --angles angles.csv` |
| Reconstruction from corrected projections | Supported | `tomojax recon --data corrected.nxs --out recon.nxs` |
| Detector-centre/COR alignment | Supported | `tomojax align --data corrected.nxs --mode cor --out aligned.nxs` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate --out synthetic_scan.nxs ...` |
| Public Python reconstruction smoke path | Supported | `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` facades |

## Claiming Rules

Only the workflows in the table above should be described as supported product
entrypoints. Workflows outside the matrix should not be presented as
user-facing workflows unless they are promoted into the public CLI, public API,
examples, and tests.
