# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: add schema validation for the deterministic Phase 7 smoke artifact
  bundle.

### Scope

- In scope:
  - Give `tomojax.verify` a typed public artifact validation API.
  - Validate required smoke JSON artifacts and indexed artifact existence.
  - Make the smoke run fail loudly before returning if required artifacts are
    missing or malformed.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Full production dataset loading through the new command.
- Deep module owner: `tomojax.verify`, integrated by `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add a typed `tomojax.verify` artifact validation API.
- [x] Wire Phase 7 smoke artifact validation into the run.
- [x] Add focused positive and negative validation tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the artifact validation slice.

### Validation

- `uv run ruff format src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use a lightweight stdlib validator rather than adding a runtime dependency
  for the smoke contract.
- Validate only the core JSON artifact schemas and artifact-index existence in
  this slice.

### Risks

- Risk: CSV and array semantic validation is still shallow.
- Mitigation: keep this API extensible and cover required JSON/index failures
  first.
