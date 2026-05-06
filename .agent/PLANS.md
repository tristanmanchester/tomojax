# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan:
- Phase:
- Goal:

### Scope

- In scope:
- Out of scope:
- Deep module owner:

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [ ] Read the relevant design docs.
- [ ] Identify public APIs and private implementation files.
- [ ] Add or update tests where practical before implementation.
- [ ] Implement the smallest vertical slice for this milestone.
- [ ] Delete superseded code introduced or made obsolete by this milestone.
- [ ] Update import-linter and public-import checks if module boundaries changed.
- [ ] Update `docs/implementation_log.md`.
- [ ] Run validation commands.

### Validation

- `just check`

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision:
- Deviation from canonical docs:
- Rationale:

### Risks

- Risk:
- Mitigation:
