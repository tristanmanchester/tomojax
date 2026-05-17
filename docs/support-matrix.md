# TomoJAX v2 Support Matrix

This page separates the current publication-facing TomoJAX v2 workflows from
developer diagnostics and active research. It is deliberately conservative:
fixed-truth geometry recovery and benchmark classification runs are useful
evidence, but they are not the same thing as a user running alignment from
projections alone.

## Production-Facing Workflows

| Workflow | Status | User command / evidence command | Evidence |
|---|---|---|---|
| Dataset inspection | `production_pass` | `tomojax inspect scan.nxs` | CLI contract tests and public IO dataset tests |
| Dataset validation | `production_pass` | `tomojax validate scan.nxs` | CLI validation tests and schema-negative tests |
| TIFF stack ingest | `production_pass` | `tomojax ingest ./tiffs --angles angles.csv --du ... --dv ... --out scan.nxs` | Public IO tests cover TIFF stack loading with explicit angle metadata |
| NX/HDF5 preprocessing | `production_pass` | `tomojax preprocess raw.nxs corrected.nxs` | Mixed-frame NXtomo/image_key preprocessing tests |
| TIFF flat/dark preprocessing | `production_pass` | `tomojax preprocess ./projections corrected.nxs --format tiff-stack --flats ./flats --darks ./darks --angles angles.csv` | Public IO and CLI tests cover TIFF projections/flats/darks plus angle sidecars |
| Reconstruction from corrected projections | `production_pass` | `tomojax recon corrected.nxs --out recon.nxs` | Reconstruction, IO, manifest, and quicklook tests |
| Detector-centre real-data alignment profile | `production_pass` for the retained k11 laminography evidence case | User command: `tomojax align corrected.nxs --mode cor --out aligned.nxs`; evidence command: staged real-laminography runner | Real-laminography reports under `docs/benchmark_runs/` and retained run artifacts |
| Real laminography staged workflow | `production_pass` for the retained k11 scan, not yet a broad corpus claim | Evidence command only: `scripts/real_laminography/run_real_lamino_staged.py` | Current staged real-data reports and PNG/contact-sheet artifacts |

## Developer Diagnostics

These are intentionally available under `tomojax dev`, not as normal user
workflows.

| Diagnostic | Status | Entrypoint | Notes |
|---|---|---|---|
| Synthetic setup-global geometry gate | `oracle_geometry_pass` today | `tomojax dev align-auto --synthetic-case setup-global --size 128 --views 256` | Proves the geometry solver path when the alignment volume is fixed to synthetic truth. This is not yet a truth-free stopped-alignment production claim. |
| Synthetic pose-random geometry gate | `oracle_geometry_pass` today | `tomojax dev align-auto --synthetic-case pose-random --size 128 --views 256` | Proves the all-5 pose solver in the fixed-volume diagnostic path. |
| Alignment benchmark runners | `diagnostic_only` | `tomojax dev alignment-diagnostic-bench`, `tomojax dev benchmark-suite` | Used for development and regression evidence. |
| Loss and runtime probes | `diagnostic_only` | `tomojax dev loss-bench`, `tomojax dev test-gpu`, `tomojax dev test-cpu` | Useful for validation, not user workflows. |

## Research Blockers

| Area | Status | Current blocker |
|---|---|---|
| Truth-free stopped detector-centre/COR recovery from a free preview volume | `research_blocker` | The preview reconstruction can absorb detector-centre error into the volume, so a later geometry step can optimise a biased fixed-volume objective. |
| Full laminography axis/roll/theta recovery across the original synthetic plan | `research_blocker` | The original laminography case still needs stronger axis/roll/theta and det-v observability evidence before it can be production-supported. |
| Object-frame thermal drift recovery | `unsupported_model` | The system can flag object-motion suspicion, but object-frame drift parameters are not a supported production solve path yet. |
| Combined nuisance, jumps, and laminography setup recovery | `research_blocker` | Bad-view and nuisance diagnostics exist, but hard-case setup/axis/theta recovery under nuisance is not production-supported. |
| Default Pallas alignment/reconstruction path | `diagnostic_only` | Reference and Pallas comparison tests exist, but the default production solver path is still JAX/reference-oriented for correctness. |

## Claiming Rules

- Do not call a fixed-truth or synthetic-truth geometry run a production pass.
- Do not call a run a production pass if hard views or unsupported metrics were
  excluded to make the summary green.
- Do not promote `tomojax dev` commands into public docs as normal user workflows.
- Do not claim support for object-frame motion, nuisance fitting, broad
  laminography, or truth-free stopped detector-centre recovery until the
  corresponding report contains production evidence.
- Reports may include developer diagnostics, but their status labels must say
  `oracle_geometry_pass`, `diagnostic_only`, `research_blocker`, or
  `unsupported_model` as appropriate.
