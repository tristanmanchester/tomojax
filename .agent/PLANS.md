# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 alternating solver and continuation vertical slice
- Goal: neutralise the stopped-volume detector-shift gauge enough for sidecar
  stopped-reconstruction recovery to meet the existing geometry tolerances.

### Scope

- In scope:
  - Use the `stopped_volume_gauge` diagnostics to guide the smallest
    reconstruction/volume gauge correction.
  - Preserve fixed-truth sidecar Schur recovery as an isolating solver check.
  - Convert the stopped-reconstruction sidecar contract from explicit recovery
    gap to passing recovery when the gauge correction works.
- Out of scope:
  - Stripe/ring bias fields.
  - Larger 128^3 benchmark runtime.
  - New placeholder artifact/report polish.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Prototype a detector-shift gauge correction for the stopped volume.
- [ ] Add or update focused stopped-reconstruction sidecar recovery assertions.
- [ ] Run focused validation and `just imports`.
- [ ] Update `docs/implementation_log.md`.
- [ ] Commit the stopped-volume gauge correction slice.

### Validation

- Pending.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Do not relax recovery tolerances or switch default behavior to fixed truth.
- Prototype result: integer volume rolls and projection-COM det_u initialisation
  improve det_u in isolation but either leave det_u outside tolerance or damage
  theta recovery. Do not commit these as implementation.

### Risks

- Risk: a detector-shift correction can become synthetic-only registration.
- Mitigation: constrain changes to geometry/volume gauge handling and verify
  with projection-domain residuals, not truth-volume registration.
