# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Nuisance models and weak DOF handling
- Goal: add the first typed low-frequency background nuisance model.

### Scope

- In scope:
  - Add a public `BackgroundOffsetModel` to `tomojax.nuisance`.
  - Implement per-view constant plus vertical-gradient background application.
  - Implement masked closed-form fitting against residual backgrounds.
  - Add focused nuisance tests and README/API updates.
- Out of scope:
  - Coupling background fitting into Schur or alternating solver.
  - Stripe/ring bias fields.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.nuisance`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add private background model implementation.
- [x] Export typed public nuisance API.
- [x] Update nuisance README.
- [x] Add focused background tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the background nuisance slice.

### Validation

- `uv run ruff format src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `uv run ruff check src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_nuisance_background.py tests/test_nuisance_gain_offset.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Start with a low-frequency vertical-gradient basis because the hardest
  synthetic nuisance spec names a low-frequency vertical background drift.

### Risks

- Risk: this background model is not yet wired into solver residual evaluation.
- Mitigation: keep the model tested and public so the next slice can integrate
  it behind an opt-in solver/config flag.
