# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: add deterministic plot-summary artifacts for the Phase 7 smoke bundle.

### Scope

- In scope:
  - Add a `plots/summary.json` artifact with convergence values suitable for
    later rendering.
  - Include FISTA loss trace and per-level geometry loss trace summaries.
  - Index the nested plots artifact and extend focused smoke tests.
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

- [x] Add nested plots summary artifact path.
- [x] Write deterministic convergence summary payload.
- [x] Extend smoke artifact/index tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the plots summary slice.

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

- Store numeric plot-ready data first; leave PNG/SVG rendering for a later
  reporting/UI pass.

### Risks

- Risk: no rendered plot images are emitted yet.
- Mitigation: persist the deterministic data needed to render convergence plots
  without adding plotting dependencies to the smoke path.
