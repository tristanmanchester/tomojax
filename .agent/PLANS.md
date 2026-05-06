# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: persist deterministic preview-slice artifacts for the Phase 7 smoke
  bundle.

### Scope

- In scope:
  - Write central truth, final, and error preview slices under
    `preview_slices/`.
  - Record a preview-slice summary JSON with shape, axis, index, and aggregate
    stats.
  - Index nested preview-slice artifacts and keep focused validation passing.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Full production dataset loading through the new command.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add nested preview-slice artifact paths.
- [x] Write central truth/final/error preview slices and summary.
- [x] Extend smoke artifact/index tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the preview-slice artifact slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Keep preview slices as `.npy` arrays for the smoke path so tests can validate
  deterministic numeric content without image rendering dependencies.
- Store nested artifact paths relative to the run directory in
  `artifact_index.json`.

### Risks

- Risk: preview image and plot artifacts are still absent.
- Mitigation: this slice records deterministic preview data first and leaves
  human-facing rendering for a separate UI/reporting pass.
