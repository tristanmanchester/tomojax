# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: fix the first `just check` Ruff failure cluster without weakening
  checks.

### Scope

- In scope:
  - Fix ambiguous Unicode punctuation in `src/tomojax/__init__.py`.
  - Fix `src/tomojax/align/_config.py` import style/type-checking/type-alias
    Ruff findings.
  - Split `AlignConfig.__post_init__` into smaller validation helpers to remove
    the current PLR0912/PLR0915 failures.
  - Run focused config tests and import gates.
- Out of scope:
  - Repository-wide legacy Ruff cleanup beyond this first cluster.
  - Behavioral changes to alignment config normalization.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Fix package docstring punctuation.
- [x] Update `_config.py` imports and type aliases.
- [x] Split `AlignConfig.__post_init__` into focused helpers.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run basedpyright src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run pytest tests/test_align_profiles.py tests/test_align_motion_models.py tests/test_align_gauge.py tests/test_align_optimizers.py tests/test_v2_module_skeleton.py -q`
  passed: 30 tests.
- `uv run pytest tests/test_cli_config.py tests/test_align_contracts.py tests/test_alignment_schedules.py -q`
  failed only on `test_alignment_public_facade_stays_narrow`, which expects the
  old three-symbol `tomojax.align.__all__` while the current committed facade
  exposes LM symbols.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `uv run pytest tests/test_cli_config.py tests/test_align_contracts.py tests/test_alignment_schedules.py tests/test_align_chunking.py -q`
  encountered a native JAX/Optax segmentation fault inside the existing LBFGS
  chunking path after the public-facade test failure.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; the first remaining failures now begin
  in `src/tomojax/align/_pose_stage.py` with `TID252` parent-relative imports,
  `TC001` type-only imports, annotation findings, and complexity findings. The
  formatter churn from this command was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep the `AlignConfig.__post_init__` helper order identical to the
  previous in-method normalization order.
- Deviation: none from the cleanup scope.

### Risks

- Risk: refactoring `__post_init__` could accidentally change config
  normalization order.
- Mitigation: keep helper calls in the same order as the existing method and
  run focused config/profile tests.
- Proposed next fix for `just check`: start a dedicated cleanup slice for
  `src/tomojax/align/_pose_stage.py` import hygiene and local annotation
  findings before addressing its larger complexity findings.
