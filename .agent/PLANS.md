# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: introduce the first typed per-view gain/offset nuisance model.

### Scope

- In scope:
  - Add a deep-module-owned `GainOffsetModel` public API in `tomojax.nuisance`.
  - Implement closed-form per-view gain/offset estimation for
    predicted/observed projection pairs.
  - Support masks and ridge stabilisation.
  - Add focused nuisance tests.
- Out of scope:
  - Coupling nuisance fitting into the alternating solver.
  - Low-frequency background fields.
  - Weak DOF auto-activation rules.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.nuisance`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add private gain/offset implementation.
- [x] Export typed public nuisance API.
- [x] Update nuisance README with behavior and tests.
- [x] Add focused gain/offset tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the gain/offset nuisance slice.

### Validation

- `uv run ruff format src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `uv run ruff check src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_nuisance_gain_offset.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Start with variable-projection closed-form gain/offset fitting because Phase 8
  allows that path and it creates a deterministic public nuisance primitive
  without changing the solver loop yet.

### Risks

- Risk: this does not yet prevent geometry from absorbing nuisance drift in the
  Phase 7 alternating loop.
- Mitigation: the next Phase 8 slice can thread this tested model into
  projection residual evaluation.
