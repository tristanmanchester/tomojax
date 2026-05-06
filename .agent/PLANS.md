# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 diagnostic pause
- Goal: record the current benchmark/memory-regression diagnosis and pause
  before starting another feature/report/refactor slice.

### Scope

- In scope:
  - Summarise the five-case 32^3 benchmark failures as wiring-only evidence.
  - Summarise fixed-truth versus stopped-reconstruction evidence.
  - Summarise the GPU memory-regression source and fix.
  - Record commands, artifacts, and remaining open questions.
  - Commit the documentation-only pause point.
- Out of scope:
  - Starting new feature, report-field, or refactor slices.
  - Tolerance relaxation, solver tuning, nuisance fitting, or reconstruction
    changes.
  - Further legacy Ruff cleanup.
- Deep module owner: documentation only; implementation remains as of commit
  `dc2aa74`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Record current best diagnosis in `docs/implementation_log.md`.
- [x] Commit the pause/diagnostic summary.

### Validation

- Documentation-only pause update; no new validation required beyond clean git
  status after commit.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Pause before any new implementation slice. Next work should diagnose
  setup/pose/theta coupling or geometry convention mapping because fixed-truth
  fails at 64^3/64 views.

### Risks

- Risk: broad `just check` still exposes legacy Ruff debt outside this slice.
- Mitigation: run focused validation plus `just imports`, and avoid legacy Ruff
  cleanup unless explicitly requested.
