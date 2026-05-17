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
| Single public CLI entrypoint | `pyproject.toml` exposes only `tomojax = "tomojax.cli.main:main"`. `uv run tomojax --help` shows `inspect`, `validate`, `preprocess`, `ingest`, `convert`, `recon`, `align`, `simulate`, and `dev`. | Done |
| Public align CLI is product-shaped | `uv run tomojax align --help` shows `--mode {cor,pose,auto,max}` and `--quality`, not schedules, losses, optimizers, or active DOF internals. | Done |
| Developer diagnostics separated | `uv run tomojax dev --help` owns benchmark and diagnostic commands. `tomojax.align.api` no longer exports alignment smoke runners; synthetic alignment diagnostic runners now live under `tomojax.bench`. | Done |
| Measured data loading owned by IO | `tomojax.io` owns `ProjectionDataset`, load/save/validate/convert/inspect/preprocess facades. Production modules are forbidden from importing lower-level `tomojax.data` directly by `.importlinter`. | Done |
| Synthetic data owned separately | `tomojax.datasets` owns synthetic benchmark generation and phantom helpers. `tomojax.data` remains a lower-level implementation package used through IO/datasets. | Done |
| Calibration helper leakage reduced | Detector-grid, axis-direction, calibration-state, and calibrated metadata helpers are re-exported by `tomojax.geometry`; production modules are forbidden from importing `tomojax.calibration` directly by `.importlinter`. | Done |
| Private implementation boundaries checked | `tools/check_public_imports.py` rejects cross-owner private imports. `just production-surface-check` runs it. | Done |
| Import direction checked | `.importlinter` now keeps three contracts: layer direction, production no-data, and production no-calibration. | Done |
| Public docs avoid development-era wording | `tests/test_cli_public_surface.py` guards selected public docs against `legacy`, `transitional`, `pre-v2`, `mvp`, `v1`, `parity`, and `smoke`. | Done |
| Focused production-surface gate | `just production-surface-check` passed with 74 focused tests after the latest boundary changes. | Done |
| Whole-repo `just check` | `just check` is still red. It fails in repo-wide Ruff lint, mainly in nested `tomojax.align.model`, `tomojax.align.objectives`, internal alignment modules, scripts, and broad tests. | Not done |
| Nested non-deep packages cleaned up | `src/tomojax/align/model`, `src/tomojax/align/objectives`, `src/tomojax/align/geometry`, `src/tomojax/data`, `src/tomojax/bench`, and `src/tomojax/calibration` still exist. Some are quarantined; the align nested packages still behave like public modules for lint/import purposes. | Incomplete |
| Obsolete/transitional code removed or quarantined | Retired console scripts and public wording are cleaned up; diagnostics are under `tomojax dev`. Some old names remain in developer command internals, historical docs, benchmark docs, and tests. | Partial |

## Current Blockers

The cleanup cannot be called complete while `just check` is red. The broad Ruff
failure is not caused by the public CLI surface; it exposes remaining
architecture debt:

- nested alignment packages (`tomojax.align.model`, `tomojax.align.objectives`,
  `tomojax.align.geometry`) are implementation internals but are named and linted
  like public modules;
- some CLI code still reaches into those nested implementation packages instead
  of consuming a compact public `tomojax.align` facade;
- broad tests and scripts still contain lint debt unrelated to the focused
  production-surface gate.

## Next Work

1. Move production CLI dependencies on alignment schedules/loss parsing behind
   the public `tomojax.align` facade.
2. Convert nested alignment implementation packages into private implementation
   namespaces, or otherwise make their internal status executable and lint-clean.
3. Continue reducing repo-wide Ruff failures until `just check` either passes or
   the remaining debt is explicitly quarantined outside production code with
   tests/import rules.
