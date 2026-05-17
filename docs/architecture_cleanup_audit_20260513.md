# TomoJAX v2 CLI and Deep-Module Cleanup Audit

Date: 2026-05-13

## Objective

Clean up the TomoJAX v2 CLI and deep-module architecture so the repo is
production-ready according to the original v2 plan:

- consolidate public CLI entrypoints;
- move real data loading behind `tomojax.io`;
- clarify production vs diagnostic surfaces;
- tighten module boundaries;
- update docs, tests, and import rules;
- remove or quarantine obsolete/transitional code without weakening verified
  functionality.

## Evidence Checklist

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Canonical v2 plan exists | `docs/tomojax-v2/01_high_level_architecture.md` through `07_synthetic_generator_pseudocode.md` are present. | Done |
| Single public CLI entrypoint | `pyproject.toml` exposes only `tomojax = "tomojax.cli.main:main"`. `uv run tomojax --help` shows only product commands: `inspect`, `validate`, `preprocess`, `ingest`, `convert`, `recon`, `align`, and `simulate`. `uv run tomojax dev --help` separately exposes diagnostics. | Done |
| Public align CLI is product-shaped | `uv run tomojax align --help` shows `--mode {cor,pose,auto,max}` and `--quality`, not schedules, losses, optimizers, or active DOF internals. | Done |
| Developer diagnostics separated | `uv run tomojax dev --help` owns benchmark and diagnostic commands. `tomojax.align.api` no longer exports alignment smoke runners; synthetic alignment diagnostic runners now live under `tomojax.bench`. | Done |
| Measured data loading owned by IO | `tomojax.io` owns `ProjectionDataset`, load/save/validate/convert/inspect/preprocess facades. Production modules are forbidden from importing lower-level `tomojax.data` directly by `.importlinter`. | Done |
| Synthetic data owned separately | `tomojax.datasets` owns synthetic benchmark generation and phantom helpers. `tomojax.data` remains a lower-level implementation package used through IO/datasets. | Done |
| Calibration helper leakage removed | Detector-grid, axis-direction, calibration-state, calibrated metadata, and detector-pixel value helpers are owned and re-exported by `tomojax.geometry`; the old `tomojax.calibration` package has been deleted. | Done |
| Private implementation boundaries checked | `tools/check_public_imports.py` rejects cross-owner private imports. `just production-surface-check` runs it. | Done |
| Import direction checked | `.importlinter` keeps layer direction, production no-data, production no-bench, CLI alignment-facade, CLI core-geometry, and CLI dev-dispatch contracts. | Done |
| Public docs avoid development-era wording | `tests/test_cli_public_surface.py` guards selected public docs against `legacy`, `transitional`, `pre-v2`, `mvp`, `v1`, `parity`, and `smoke`. | Done |
| Focused production-surface gate | `just production-surface-check` passed with 74 focused tests after the latest boundary changes. | Done |
| Whole-repo `just check` | Not rerun after the aggressive align namespace cut. The current focused gate is clean; remaining full-repo debt should be measured in a final sweep rather than used to block every architectural deletion. | Deferred |
| Nested non-deep packages cleaned up | `src/tomojax/calibration` has been deleted. `src/tomojax/align/model`, `src/tomojax/align/objectives`, and `src/tomojax/align/geometry` were renamed to private implementation namespaces: `src/tomojax/align/_model`, `src/tomojax/align/_objectives`, and `src/tomojax/align/_geometry`. Product surfaces are guarded from importing them directly. | Done |
| Obsolete/transitional code removed or quarantined | Retired console scripts and public wording are cleaned up; diagnostics are under `tomojax dev`. Some old names remain in developer command internals, historical docs, benchmark docs, and tests. | Partial |

## Current Blockers

The biggest public-module ambiguity has been removed: the nested alignment model,
objective, and geometry packages are no longer named as public packages. The
remaining cleanup blockers are now narrower:

- broader scripts/tests may still have repo-wide lint debt unrelated to the
  production surface;
- `tomojax.data` and `tomojax.bench` are still retained/developer packages, but
  they are now explicitly guarded behind IO/datasets and developer command
  surfaces;
- a final whole-repo `just check` pass is still needed after the remaining
  architectural deletions are batched.

## Next Work

1. Continue deleting or assimilating old script/diagnostic surfaces now that the
   alignment implementation packages are private.
2. Collapse any remaining product imports onto `tomojax.align`, `tomojax.io`,
   `tomojax.datasets`, `tomojax.geometry`, and `tomojax.verify`.
3. Run the final whole-repo `just check` pass after the remaining large cleanup
   batch, then fix only the failures that survive that sweep.
