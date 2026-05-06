# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: persist truth and corrupted-input provenance with the Phase 7
  deterministic smoke run.

### Scope

- In scope:
  - Persist `ground_truth_volume.npy`.
  - Emit `geometry_true.json` and explicit `geometry_corrupted.json`.
  - Add those artifacts to `artifact_index.json` and focused tests.
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

- [x] Write truth volume and true/corrupted geometry artifacts.
- [x] Extend artifact index/test coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the smoke truth provenance slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- `geometry_initial.json` remains the alignment initial geometry; add
  `geometry_corrupted.json` as the explicit synthetic corrupted-input name.
- `geometry_true.json` should be the zero-gauge geometry used before synthetic
  corruption.

### Risks

- Risk: this does not replace the full synthetic dataset generator.
- Mitigation: mirror the key artifact names from the benchmark contract in the
  smoke run.
