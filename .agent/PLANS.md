# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 bridge before Phase 1
- Goal: finish the guardrail/architecture-smell audit and define the smallest
  deletion/migration path into the v2 deep-module skeleton.

### Scope

- In scope:
  - Record current guardrail status against the live tree.
  - Identify pre-v2 architecture smells that block Phase 1.
  - Pick the first deep-module migration target and keep checks strict.
- Out of scope:
  - Implementing benchmark datasets, projectors, reconstruction, or optimisers.
  - Preserving old CLI/API compatibility surfaces unless explicitly needed for a
    benchmark/reference primitive.
- Deep module owner: repository-level architecture guardrails; first migration
  candidate is `tomojax.core`/`tomojax.io` ownership of former `tomojax.utils`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Read `AGENTS.md` and the canonical phased plan.
- [x] Verify the v2 design docs and guardrail files exist.
- [x] Run current import-boundary guardrails.
- [x] Run current typecheck to capture the transitional failure shape.
- [x] Identify public API/private implementation gaps.
- [x] Migrate or delete the first shallow utility surface without weakening
  checks.
- [x] Add or update tests where practical before implementation.
- [x] Delete superseded code introduced or made obsolete by this milestone.
- [x] Update import-linter and public-import checks if module boundaries changed.
- [x] Update `docs/implementation_log.md`.
- [x] Run validation commands.

### Validation

- `just imports` passes.
- `uv run ruff check src/tomojax/io src/tomojax/calibration/_json.py
  tests/test_json_utils.py` passes.
- `uv run basedpyright src/tomojax/io` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py -q` passes with 18 tests.
- `uv run ruff check src/tomojax/geometry tests/test_regression_geometry_io.py
  tests/test_axes_io.py tests/test_issue_fix_pr.py` passes.
- `uv run basedpyright src/tomojax/geometry` passes with 0 errors and 0
  warnings.
- `uv run pytest tests/test_axes_io.py tests/test_regression_geometry_io.py
  tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py
  tests/test_align_roi.py -q` passes with 66 tests.
- `uv run ruff check src/tomojax/motion tests/test_phasecorr.py` passes.
- `uv run basedpyright src/tomojax/motion` passes with 0 errors and 0
  warnings.
- `uv run pytest tests/test_phasecorr.py -q` passes with 5 tests.
- `uv run ruff check src/tomojax/backends tests/test_memory.py` passes.
- `uv run pytest tests/test_memory.py tests/test_cli_geometry_build.py
  tests/test_small_module_coverage.py -q` passes with 40 tests.
- `uv run ruff check src/tomojax/core/_logging.py src/tomojax/core/api.py
  src/tomojax/core/__init__.py tests/test_logging.py
  tests/test_small_module_coverage.py` passes.
- `uv run basedpyright src/tomojax/core/_logging.py src/tomojax/core/api.py
  src/tomojax/core/__init__.py` passes with 0 errors and 0 warnings.
- `uv run pytest tests/test_logging.py tests/test_small_module_coverage.py -q`
  passes with 9 tests.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py
  tests/test_align_checkpoint.py tests/test_axes_io.py
  tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py
  tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py
  tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py -q`
  passes with 105 tests.
- `uv run basedpyright` currently fails on the transitional tree with 4456
  errors and 9341 warnings. First reported failures are in `align/_config.py`,
  `align/_observer.py`, `align/_pose_stage.py`, and `tests/test_views.py`.
- `just check` currently fails during `uv run ruff check --fix src tests tools`
  after formatting. The first current failures are broad pre-existing
  transitional lint debt in old modules such as `src/tomojax/__init__.py`,
  `src/tomojax/align/_config.py`, and `src/tomojax/align/_pose_stage.py`;
  the command reported 2065 errors total, 448 fixed, and 1617 remaining.
  Proposed next fix: continue deleting or migrating old transitional owners
  into v2 deep modules rather than weakening Ruff or bulk-fixing legacy code
  outside the active milestone.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep the existing strict gates and treat failures as migration work,
  not as reasons to loosen configuration.
- Decision: make `tomojax.utils` the first cleanup target because production
  imports remain in `align/io/checkpoint.py`, `calibration/_json.py`, and
  `cli/manifest.py`, while `utils` is explicitly forbidden by the v2
  architecture.
- Decision: move the shared JSON normalization contract into `tomojax.io`
  because manifests, checkpoint metadata, and calibration JSON all serialize
  artifacts rather than perform numerical core work.
- Decision: move axis-order and detector-FOV helpers into `tomojax.geometry`
  because they describe geometry metadata and reconstruction domains, not
  generic utilities.
- Decision: move phase-correlation translation estimation into
  `tomojax.motion` because it estimates per-view motion used by alignment
  initialization.
- Decision: move backend memory and gather-dtype probes into
  `tomojax.backends` because they are runtime backend policy, not generic
  utilities.
- Decision: move logging/progress formatting into `tomojax.core` as shared
  runtime instrumentation so the forbidden `tomojax.utils` package can be
  deleted.
- Deviation from canonical docs: this is a Milestone 0 bridge before canonical
  Phase 1, because the tree still has old module owners and no top-level
  `api.py`/`README.md` deep-module skeletons.
- Rationale: migrating the shallow utility surface first removes a forbidden
  abstraction layer without committing to the full Phase 1 public API shape.

### Risks

- Risk: old tests are white-box coupled to private implementation and
  transitional module names.
- Mitigation: update tests only as the corresponding public/deep-module owner is
  introduced, keeping explicit `check-public-imports: allow-private` exceptions
  temporary and visible.
- Risk: large typecheck failure volume can hide new regressions.
- Mitigation: record current failure shape and run narrower validation after
  each scoped migration.
- Risk: `tomojax.backends._memory` still contains dynamic JAX/device probes that
  do not pass narrow basedpyright yet.
- Mitigation: keep its public facade typed and cover behavior with focused
  memory tests until backend policy is redesigned in a later milestone.
