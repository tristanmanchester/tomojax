# TomoJAX v2 Support Matrix

This matrix describes the publishable product spine. It deliberately excludes
one-off diagnostics, benchmark harnesses, historical parity gates, and
scan-specific article runners.

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection | Supported | `tomojax inspect scan.nxs` |
| Dataset validation | Supported | `tomojax validate scan.nxs` |
| TIFF stack ingest | Supported | `tomojax ingest ./tiffs --angles angles.csv --du ... --dv ... --out scan.nxs` |
| NX/HDF5 preprocessing | Supported | `tomojax preprocess raw.nxs corrected.nxs` |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections corrected.nxs --format tiff-stack --flats ./flats --darks ./darks --angles angles.csv` |
| Reconstruction from corrected projections | Supported | `tomojax recon corrected.nxs --out recon.nxs` |
| Detector-centre/COR alignment | Supported product path | `tomojax align corrected.nxs --mode cor --out aligned.nxs` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate --out synthetic_scan.nxs ...` |
| Public Python reconstruction smoke path | Supported | `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` facades |

## Not Shipped as Product Surface

The product spine intentionally does not expose benchmark suites, ASTRA
comparison scripts, fixed-truth solver gates, publication contact-sheet builders,
large development logs, v1-parity harnesses, or hidden developer subcommands.
Those materials were moved to the accompanying development archive so they can
be recovered without defining the public package.

## Claiming Rules

Only the workflows in the table above should be described as supported product
entrypoints. Fixed-truth geometry experiments, hard-case solver sweeps, and
scan-specific article scripts may be useful evidence, but they should not be
presented as user-facing workflows unless they are promoted into the public CLI,
public API, examples, and retained tests.
