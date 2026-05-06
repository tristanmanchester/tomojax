# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: expand the Phase 7 smoke run manifest toward the artifact contract.

### Scope

- In scope:
  - Add version, git commit, started/finished timestamps, backend requested,
    and geometry model details to `run_manifest.json`.
  - Keep the manifest deterministic where practical for smoke tests.
  - Extend smoke and artifact-validation tests.
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

- [x] Expand run manifest payload.
- [x] Extend smoke and validator tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the run manifest contract slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use deterministic timestamp placeholders for the smoke path, while recording
  the current git commit when available.

### Risks

- Risk: runtime timestamps are not measured in this smoke manifest.
- Mitigation: keep deterministic placeholders and add real timing when the
  production command owns run orchestration.
