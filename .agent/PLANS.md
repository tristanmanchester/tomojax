# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Skeleton bridge before canonical Phase 0/Phase 1 implementation
- Goal: make the canonical v2 top-level deep-module skeleton importable and
  enforceable without migrating numerical behavior yet.

### Scope

- In scope:
  - Add missing top-level v2 package facades: `nuisance`, `forward`, `verify`,
    and `datasets`.
  - Add missing `api.py`/`README.md` public-boundary files for existing
    top-level owners where practical.
  - Update import-linter to include the newly importable top-level modules.
  - Add a small contract test that every canonical v2 top-level module has the
    required facade files and imports cleanly.
- Out of scope:
  - Implementing synthetic datasets, projectors, residuals, reconstruction, or
    optimisers.
  - Deleting old `data`, `calibration`, `bench`, or nested `align/*`
    transitional owners.
  - Changing old CLI behavior or public command compatibility.
- Deep module owner: repository-level v2 skeleton and import-boundary
  guardrails.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Finish and commit the Milestone 0 utility cleanup.
- [x] Read the repo-layout deep-module skeleton requirements.
- [x] Add missing top-level skeleton packages and facade files.
- [x] Add or update tests where practical before implementation.
- [x] Update import-linter and public-import checks if module boundaries changed.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.
- [ ] Commit the skeleton bridge slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/nuisance src/tomojax/forward
  src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py
  src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py
  tests/test_v2_module_skeleton.py` passes.
- `uv run basedpyright src/tomojax/nuisance src/tomojax/forward
  src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py
  src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py
  tests/test_v2_module_skeleton.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_v2_module_skeleton.py -q` passes with 2 tests.
- `just imports` passes.
- `uv run ruff format --check src/tomojax/nuisance src/tomojax/forward
  src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py
  src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py
  tests/test_v2_module_skeleton.py` passes.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py
  tests/test_v2_module_skeleton.py -q` passes with 107 tests.
- `just check` is not rerun for this skeleton-only slice because the immediately
  preceding Milestone 0 run already stopped in broad transitional legacy Ruff
  debt before reaching this new code. Next fix remains owner-by-owner migration
  or deletion of old transitional modules.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep this as a skeleton-only bridge so the repo shape converges on
  `docs/tomojax-v2/03_repo_layout.md` before Phase 0 benchmark and Phase 1
  geometry implementations add behavior.
- Decision: empty new facades should expose `__all__: tuple[str, ...] = ()`
  until the owning milestone introduces a typed public API. This avoids
  placeholder classes that would become compatibility debt.
- Deviation from canonical docs: old transitional owners remain importable for
  now. They will be deleted or migrated owner-by-owner in later milestones.
- Rationale: executable package boundaries help future migrations land in the
  correct owner without pretending the numerical implementation already exists.

### Risks

- Risk: adding empty modules can look like completed implementation.
- Mitigation: READMEs and the implementation log must mark these as skeleton
  facades only.
- Risk: old `tomojax.data` and future `tomojax.datasets` may temporarily
  coexist.
- Mitigation: keep `datasets` empty until the synthetic benchmark foundation
  owns deterministic generators and manifests.
