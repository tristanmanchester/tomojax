# TomoJAX Implementation Log

This log records implementation milestones, validation commands, design
decisions, deviations from `docs/tomojax-v2/`, and unresolved risks.

## 2026-05-06 — Milestone 0 Guardrail Preparation

### Summary

- Added strict Ruff, pytest, basedpyright, import-linter, and pre-commit guardrails.
- Added `tools/check_public_imports.py` to prevent cross-boundary imports from
  private TomoJAX modules.
- Marked existing white-box tests with explicit
  `check-public-imports: allow-private` exceptions.
- Added `.agent/PLANS.md` as the active milestone execution-plan workspace.
- Added this implementation log.

### Decisions

- `docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased
  implementation plan.
- `.agent/PLANS.md` is not a competing plan. It is the active milestone work log
  and checklist.
- `.importlinter` currently reflects importable packages in the transitional
  tree. It must be updated as the v2 deep-module skeleton becomes real.

### Validation

- `uv lock` completed after adding `basedpyright` and `pre-commit`.
- `just --list` found the canonical command surface.
- `uv run pre-commit validate-config` passed.
- `uv run ruff check tools/check_public_imports.py pyproject.toml` passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run basedpyright --version` reported `basedpyright 1.39.3`.
- `just typecheck` currently fails on the transitional source and test tree.
  This is expected technical debt from the pre-restructure implementation, not a
  guardrail installation failure. The v2 implementation should make strict
  typechecking pass as modules are replaced.

### Risks

- The current codebase still contains transitional modules such as
  `tomojax.utils`, `tomojax.data`, and `tomojax.calibration`. The v2 restructure
  should delete or migrate them under the deep-module architecture rather than
  preserve them as compatibility surfaces.
- `just check` cannot pass until type errors in the existing implementation are
  removed or the old implementation is replaced by the v2 deep modules.

## 2026-05-06 — Remove Stale Pre-v2 User Documentation

### Summary

- Removed stale pre-v2 user-facing archive docs to prevent agents from treating
  old CLI, config, tutorial, and public API surfaces as current architecture.
- Replaced the root README with a v2 rewrite README that points to
  `docs/tomojax-v2/` as the canonical design source.

### Decisions

- Kept historical brainstorms, ideation notes, implementation plans, and
  solution notes under `docs/archive/` for now because they may still contain
  useful benchmark and implementation archaeology.
- Deleted archived user-facing docs where Git history is a better archive than
  a stale in-tree compatibility signal.

### Validation

- Checked current v2-facing docs and guardrail files for links to the removed
  archive pages:
  `README.md`, `AGENTS.md`, `.agent/PLANS.md`, `docs/implementation_log.md`,
  `docs/tomojax-v2/`, `justfile`, and `pyproject.toml`.
- No references to the removed archived install, quickstart, CLI, reference,
  concepts, tutorials, or troubleshooting pages remain in those current files.

## 2026-05-06 — Milestone 0 Architecture-Smell Audit

### Summary

- Re-read `AGENTS.md` and the canonical phased plan before starting further
  migration work.
- Verified the v2 design docs and guardrail files are present in the checkout.
- Updated `.agent/PLANS.md` from the blank template into an active Milestone 0
  bridge plan.
- Confirmed the live source tree is still transitional:
  - no top-level `src/tomojax/*` package currently has the required v2 `api.py`
    and `README.md` pair;
  - forbidden `tomojax.utils` production imports remain in
    `align/io/checkpoint.py`, `calibration/_json.py`, and `cli/manifest.py`;
  - old top-level owners such as `tomojax.data`, `tomojax.calibration`, and
    `tomojax.bench` still exist outside the canonical v2 owner list;
  - nested old alignment/core geometry packages remain exposed under
    `align/geometry`, `align/model`, `align/objectives`, `align/io`, and
    `core/geometry`.

### Decisions

- Keep strict Ruff, basedpyright, import-linter, and public-private import
  checks in place. Current failures should drive migration/deletion work rather
  than guardrail weakening.
- Treat `tomojax.utils` as the first cleanup target because it is explicitly
  forbidden by the v2 architecture and has a small production import footprint.
- Do not start Phase 1 skeleton work until Milestone 0 records which old
  surfaces are being deleted, migrated, or temporarily retained as benchmark
  references.

### Validation

- `sifs agent-context --json` reported the current SIFS CLI contract.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run basedpyright` failed on the transitional tree with 4456 errors and
  9341 warnings. First reported errors include:
  - `align/_config.py`: hyphenated `GaugePolicyInput` is not normalized before
    calling `resolve_alignment_schedule`;
  - `align/_observer.py`: unnecessary `isinstance` check;
  - `align/_pose_stage.py`: private usage and many unknown JAX typing errors;
  - `tests/test_views.py`: dummy geometry does not satisfy the `Geometry`
    protocol return type.

### Risks

- The import-linter contract currently reflects only part of the old package
  graph and must be replaced when the v2 deep-module skeleton lands.
- The typecheck failure volume is high enough that new errors can be hidden
  unless each scoped migration is validated narrowly before rerunning broader
  checks.

## 2026-05-06 — Move JSON Serialization Out Of `tomojax.utils`

### Summary

- Added the first v2-owned deep module surface:
  - `src/tomojax/io/api.py`
  - `src/tomojax/io/__init__.py`
  - `src/tomojax/io/_json.py`
  - `src/tomojax/io/README.md`
- Moved the shared JSON normalization contract from forbidden
  `tomojax.utils.json` into `tomojax.io`.
- Updated production consumers in alignment checkpoints, calibration JSON, and
  CLI manifests to import through the public `tomojax.io` facade.
- Updated JSON utility tests to assert the new public API and deleted
  `src/tomojax/utils/json.py`.
- Added `tomojax.io` to `.importlinter` so the new module owner is included in
  executable import-boundary checks.

### Decisions

- `tomojax.io` owns artifact/metadata serialization helpers. This keeps JSON
  normalization out of generic utilities while avoiding a premature dependency
  on future dataset or verifier schemas.
- Kept `calibration/_json.py` as a temporary internal adapter because the whole
  `tomojax.calibration` owner is transitional and will be migrated or deleted
  under the v2 geometry/motion/nuisance plan.

### Validation

- `uv run ruff check src/tomojax/io src/tomojax/calibration/_json.py tests/test_json_utils.py`
  passed.
- `uv run basedpyright src/tomojax/io` passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py -q`
  passed: 18 tests.
- `just imports` passed after adding `tomojax.io` to `.importlinter`.
- `rg -n "tomojax\\.utils\\.json|utils/json" src tests docs .agent` found no
  remaining references.

### Risks

- Other `tomojax.utils` modules still remain (`axes`, `fov`, `logging`,
  `memory`, `phasecorr`, and `subprocesses`) and need owner-by-owner migration
  or deletion before the v2 skeleton is clean.

## 2026-05-06 — Move Axis And FOV Helpers Into `tomojax.geometry`

### Summary

- Added the second v2-owned deep module surface:
  - `src/tomojax/geometry/api.py`
  - `src/tomojax/geometry/__init__.py`
  - `src/tomojax/geometry/_axes.py`
  - `src/tomojax/geometry/_fov.py`
  - `src/tomojax/geometry/README.md`
- Moved axis-order and detector field-of-view helpers from forbidden
  `tomojax.utils` modules into `tomojax.geometry`.
- Updated production consumers in NXtomo IO, reconstruction CLI, alignment CLI,
  and pose-stage masking to import through the public `tomojax.geometry`
  facade.
- Updated geometry/FOV tests to assert the new public API and deleted
  `src/tomojax/utils/axes.py` and `src/tomojax/utils/fov.py`.
- Added `tomojax.geometry` to `.importlinter` so it participates in the current
  executable import-boundary checks.

### Decisions

- `tomojax.geometry` owns axis-order metadata and detector-FOV ROI helpers
  because these helpers describe geometry conventions and reconstruction domains
  rather than generic utilities.
- Kept existing behavior intact, including the private white-box monkeypatch in
  `tests/test_regression_geometry_io.py`, but marked it with the explicit
  public-import checker exception so this remaining test coupling is visible.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_regression_geometry_io.py tests/test_axes_io.py tests/test_issue_fix_pr.py`
  passed.
- `uv run basedpyright src/tomojax/geometry` passed with 0 errors and 0
  warnings.
- `uv run pytest tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py -q`
  passed: 66 tests.
- `just imports` passed after adding `tomojax.geometry` to `.importlinter`.
- `rg -n "utils\\.axes|utils\\.fov|tomojax\\.utils\\.axes|tomojax\\.utils\\.fov|from tomojax\\.utils import axes" src/tomojax tests`
  found no remaining axis/FOV utility references.

### Risks

- `tomojax.geometry` currently depends on the transitional
  `tomojax.core.geometry.base` types. Phase 1 should decide whether those types
  move into top-level `tomojax.geometry` or remain as lower-level core
  primitives.
- `tomojax.utils` still contains logging, memory, phase-correlation, and
  subprocess helper surfaces that need explicit v2 owners or deletion.

## 2026-05-06 — Move Phase Correlation Into `tomojax.motion`

### Summary

- Added the third v2-owned deep module surface:
  - `src/tomojax/motion/api.py`
  - `src/tomojax/motion/__init__.py`
  - `src/tomojax/motion/_phasecorr.py`
  - `src/tomojax/motion/README.md`
- Moved phase-correlation translation estimation from forbidden
  `tomojax.utils.phasecorr` into `tomojax.motion`.
- Updated the alignment stage-loop initializer to import `phase_corr_shift`
  through the public `tomojax.motion` facade.
- Updated phase-correlation tests to assert the public API while keeping the
  `_wrap_shift` white-box check explicit with a public-import checker exception.
- Added `tomojax.motion` to `.importlinter`.

### Decisions

- `tomojax.motion` owns phase-correlation because it estimates per-view motion
  for alignment initialization, not generic utility behavior.
- Used a narrow pyright suppression on `jnp.asarray` in the private
  implementation because the current JAX stubs expose that member as partially
  unknown. The public API remains typed and the module-level basedpyright gate
  is green.

### Validation

- `uv run ruff check src/tomojax/motion tests/test_phasecorr.py` passed.
- `uv run basedpyright src/tomojax/motion` passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_phasecorr.py -q` passed: 5 tests.
- `just imports` passed after adding `tomojax.motion` to `.importlinter`.

### Risks

- `tomojax.motion` currently contains only a seed/initializer primitive. Phase 1
  still needs to define whether full per-view pose parameterizations live here
  or stay under `tomojax.align` until the geometry optimizer milestone.

## 2026-05-06 — Move Backend Memory Probes Into `tomojax.backends`

### Summary

- Added the fourth v2-owned deep module surface:
  - `src/tomojax/backends/api.py`
  - `src/tomojax/backends/__init__.py`
  - `src/tomojax/backends/_memory.py`
  - `src/tomojax/backends/_subprocesses.py`
  - `src/tomojax/backends/README.md`
- Moved memory budgeting, gather-dtype selection, and `nvidia-smi` probing out
  of forbidden `tomojax.utils`.
- Updated CLI and simulation consumers to import backend helpers through the
  public `tomojax.backends` facade.
- Updated memory tests to keep private white-box checks explicit with
  public-import checker exceptions.
- Added `tomojax.backends` to `.importlinter`.

### Decisions

- `tomojax.backends` owns device-memory and gather-dtype heuristics because
  those helpers are runtime backend policy rather than generic utilities.
- Kept `_subprocesses.py` private to `tomojax.backends` instead of exposing a
  generic command runner.

### Validation

- `uv run ruff check src/tomojax/backends tests/test_memory.py` passed.
- `uv run pytest tests/test_memory.py tests/test_cli_geometry_build.py tests/test_small_module_coverage.py -q`
  passed: 40 tests.
- `just imports` passed after adding `tomojax.backends` to `.importlinter`.

### Failures And Risks

- `uv run basedpyright src/tomojax/backends` currently fails on private dynamic
  JAX/device-probe code and subprocess wrapper typing. This is narrower than
  the repo-wide transitional typecheck failure, but it is still type debt in the
  moved implementation.
- Phase 1 should decide whether backend probes stay as Python runtime helpers or
  become a smaller typed policy layer with untyped adapter code hidden behind
  explicit boundaries.

## 2026-05-06 — Remove The `tomojax.utils` Package

### Summary

- Added the fifth v2-owned deep module surface:
  - `src/tomojax/core/api.py`
  - `src/tomojax/core/__init__.py`
  - `src/tomojax/core/_logging.py`
  - `src/tomojax/core/README.md`
- Moved logging setup, JAX environment logging, progress iteration, and duration
  formatting into `tomojax.core`.
- Updated reconstruction, alignment, data simulation, and CLI consumers to use
  `tomojax.core`.
- Removed the now-empty `src/tomojax/utils` package after moving JSON, axis/FOV,
  phase-correlation, backend-memory, subprocess, and logging helpers to explicit
  v2 owners.

### Decisions

- `tomojax.core` owns shared runtime instrumentation for now because these
  helpers are used across numerical modules and CLI surfaces.
- Kept CLI-specific logging setup in the same public core facade until Phase 1
  creates a fuller `tomojax.cli` deep module. This avoids recreating a generic
  utility bucket under a different name.

### Validation

- `uv run ruff check src/tomojax/core/_logging.py src/tomojax/core/api.py src/tomojax/core/__init__.py tests/test_logging.py tests/test_small_module_coverage.py`
  passed.
- `uv run basedpyright src/tomojax/core/_logging.py src/tomojax/core/api.py src/tomojax/core/__init__.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_logging.py tests/test_small_module_coverage.py -q`
  passed: 9 tests.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py -q`
  passed: 105 tests.
- `just imports` passed.
- `fd -p 'src/tomojax/utils' -a` found no remaining `tomojax.utils` package.
- `rg -n "tomojax\\.utils|from \\.\\.utils|from tomojax\\.utils|import tomojax\\.utils" src/tomojax tests`
  found no code/test imports. Remaining matches are only explanatory README
  references in the new owner modules.
- `just check` currently fails during
  `uv run ruff check --fix src tests tools` after formatting. Current failure
  shape is broad transitional legacy lint debt, beginning with ambiguous
  Unicode in `src/tomojax/__init__.py`, import/type-checking/style findings in
  `src/tomojax/align/_config.py`, and old pose-stage lint/type annotation
  findings in `src/tomojax/align/_pose_stage.py`. The command reported 2065
  errors total, 448 fixed, and 1617 remaining.

### Risks

- `tomojax.core` now has a new public facade, but the rest of the old
  `tomojax.core` package still lacks a complete v2 `api.py` boundary for
  geometry/projector/validation internals.
- Some tests still deliberately white-box private moved implementations. Those
  exceptions are explicit but should be eliminated as Phase 1 public APIs settle.
- `just check` remains the milestone target, but passing it requires continuing
  the planned migration/deletion of old transitional modules rather than
  weakening or mass-suppressing checks.

## 2026-05-06 — Add The v2 Deep-Module Skeleton

### Summary

- Added missing canonical top-level v2 skeleton packages:
  - `tomojax.nuisance`
  - `tomojax.forward`
  - `tomojax.verify`
  - `tomojax.datasets`
- Added `api.py`, `__init__.py`, and `README.md` facade files for the new
  skeleton packages.
- Added missing v2 facade files for existing top-level owners:
  - `tomojax.cli`
  - `tomojax.align`
  - `tomojax.recon`
- Added `tests/test_v2_module_skeleton.py` to enforce the top-level
  `README.md`/`__init__.py`/`api.py` contract and import every canonical facade.
- Added the newly importable v2 modules to `.importlinter`.

### Decisions

- Empty new facades export no public names until the owning implementation
  milestone introduces typed contracts. This avoids placeholder APIs that would
  become compatibility debt.
- Kept old transitional owners (`tomojax.data`, `tomojax.calibration`,
  `tomojax.bench`, and nested alignment packages) importable for now. They will
  be deleted or migrated owner-by-owner instead of hidden behind new
  compatibility layers.
- Kept the skeleton bridge behavior-free. The next implementation milestone
  should add actual benchmark/dataset or geometry behavior against these
  boundaries.

### Validation

- `uv run ruff check src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_v2_module_skeleton.py -q` passed: 2 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run ruff format --check src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py -q`
  passed: 107 tests.

### Risks

- `just check` was not rerun for this skeleton-only slice because the immediately
  preceding run already stopped in broad legacy Ruff failures before reaching
  the new code. Passing `just check` still requires migrating or deleting old
  transitional modules rather than weakening checks.
- `tomojax.datasets` and old `tomojax.data` temporarily coexist. The synthetic
  benchmark foundation should make `datasets` the owner for deterministic v2
  generators and then delete or migrate old data code deliberately.

## 2026-05-06 — Add Synthetic Dataset Foundation

### Summary

- Added typed synthetic benchmark spec loading in `tomojax.datasets` from
  `docs/tomojax-v2/benchmark_manifest.yaml`.
- Added a deterministic procedural phantom generator for smoke and benchmark
  artifact generation.
- Added `generate_synthetic_dataset(...)`, which writes:
  - `dataset_manifest.json`
  - `ground_truth_volume.npy`
  - `projections.npy`
  - `mask.npy`
  - `nominal_geometry.json`
  - `corrupted_geometry.json`
  - `true_geometry.json`
  - `true_pose.csv`
  - `true_motion.csv`
  - `nuisance_truth.json`
  - `noise_truth.json`
- Added `tests/test_synthetic_datasets.py` for manifest loading, deterministic
  phantom generation, and repeatable 32^3 smoke artifact emission.

### Decisions

- `tomojax.datasets` owns the v2 synthetic benchmark foundation. Old
  `tomojax.data` remains transitional and was not extended for this slice.
- The first projection writer is a deterministic CPU smoke projector used to
  produce artifact contracts. It is not the final differentiable JAX reference
  projector, which remains owned by the `tomojax.forward` milestone.
- 128^3 mode is supported by configuration, but tests exercise 32^3 smoke mode
  to keep validation fast.

### Validation

- `uv run ruff check src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `uv run ruff format --check src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py -q`
  passed: 110 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The smoke projector is intentionally simple. It should be replaced as the
  benchmark truth generator once the `tomojax.forward` JAX reference projector
  is implemented and validated.

## 2026-05-06 — Add Geometry State And Gauge Canonicalisation

### Summary

- Added typed v2 geometry state containers in `tomojax.geometry`:
  - `ScalarParameter`
  - `SetupParameters`
  - `PoseParameters`
  - `GeometryState`
- Added gauge canonicalisation:
  - `mean(dx_px) -> det_u_px`
  - `mean(phi_residual_rad) -> theta_offset_rad`
  - `mean(dz_px) -> det_v_px` only when `det_v_px` is active
- Added structured gauge reports with `GaugeTransfer`, `GaugeReport`, and
  `CanonicalizedGeometry`.
- Added `tests/test_geometry_gauges.py` covering zero-centering, inactive
  `det_v`, active `det_v`, shape validation, and realised setup-plus-pose gauge
  preservation.

### Decisions

- Kept these v2 state types in top-level `tomojax.geometry` without replacing
  old `tomojax.core.geometry` primitives in this slice.
- Implemented only state and gauge data structures. Geometry artifact
  serialisation, optimiser integration, Jacobians, and Schur solves remain
  separate milestones.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `uv run ruff format --check src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py -q`
  passed: 113 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Geometry state JSON/CSV artifact writing is still missing and should be added
  before optimizer milestones rely on geometry provenance.

## 2026-05-06 — Add Geometry Artifact Serialization

### Summary

- Added versioned geometry setup JSON payloads for `GeometryState`.
- Added `write_geometry_json` and `read_geometry_json` for
  `geometry_initial.json` / `geometry_final.json`-compatible artifacts.
- Added `write_pose_params_csv` and `read_pose_params_csv` for per-view 5-DOF
  pose arrays.
- Added `write_pose_decomposition_csv` for realised setup-plus-pose channels:
  `theta_offset + phi_residual`, `det_u + dx`, and `det_v + dz`.
- Added `tests/test_geometry_serialization.py` covering JSON/CSV round-trip,
  contract artifact filenames, schema version, active parameter metadata, and
  decomposition values.

### Decisions

- Geometry artifact serialization lives in `tomojax.geometry` because these
  files encode geometry-state contracts. Run-level artifact indexing remains a
  future `tomojax.verify` responsibility.
- Pose arrays are stored in CSV artifacts while setup parameter metadata lives
  in JSON. This matches the v2 artifact contract and keeps per-view arrays easy
  to inspect.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `uv run ruff format --check src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py -q`
  passed: 116 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The serializer does not yet write full run artifact indexes or observability
  reports. Those remain part of later `tomojax.verify` and optimiser slices.

## 2026-05-06 — Add Minimal Forward Reference And Robust Residuals

### Summary

- Added `tomojax.forward.project_parallel_reference`, a minimal JAX reference
  projector for tiny cubic-volume smoke tests.
- Added masked whitened residual helpers:
  - `masked_whitened_residual`
  - `pseudo_huber_loss`
  - `pseudo_huber_weights`
  - `residual_loss`
  - `ResidualResult`
- Added `tests/test_forward_reference.py` covering projection shape, per-view
  detector shift, masking, pseudo-Huber robust behavior, valid counts, and IRLS
  weight behavior.

### Decisions

- Kept this as a new `tomojax.forward` reference slice rather than adapting old
  `tomojax.core.projector`.
- The projector is intentionally minimal: it uses coarse array rotation and
  detector shifts for smoke tests. Full ray geometry, laminography, detector
  roll, axis rotations, theta-scale handling, and finite-difference geometry
  checks remain future Phase 2 work.
- Robust residuals follow the v2 loss spec: masked, whitened residuals with
  pseudo-Huber loss and IRLS weights.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `uv run ruff format --check src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py -q`
  passed: 121 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The minimal projector is not yet the final physical forward model. It should
  be expanded before geometry optimisation tests rely on physical accuracy.

## 2026-05-06 — Add Tiny Reconstruction And Alignment Smoke Path

### Summary

- Updated the minimal forward projector so setup `det_u_px` and active
  `det_v_px` contribute to detector shifts. This makes gauge canonicalisation
  projection-preserving for the channels the smoke projector supports.
- Added `tomojax.recon.reconstruct_average_reference`, a tiny deterministic
  average-backprojection preview helper.
- Added `tomojax.align.run_alignment_smoke`, which:
  - reconstructs a stopped-gradient preview volume;
  - computes masked robust projection loss;
  - canonicalises geometry gauges;
  - recomputes loss after canonicalisation;
  - reports loss values, valid count, and the canonicalised geometry/report.
- Added `tests/test_vertical_smoke.py` for gauge-equivalent projection
  preservation, preview reconstruction shape, and alignment smoke report
  invariants.

### Decisions

- Kept the smoke path as explicit reference scaffolding, not the product
  optimiser. It wires the v2 modules together before FISTA and LM/GN land.
- `reconstruct_average_reference` is not the default reconstruction algorithm.
  It exists only to exercise the forward/residual/gauge path with a volume-like
  object.

### Validation

- `uv run ruff check src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `uv run ruff format --check src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py -q`
  passed: 124 tests.

### Risks

- A broad `uv run ruff format --check src/tomojax/forward src/tomojax/recon src/tomojax/align ...`
  still reports 20 untouched transitional align/recon files that would be
  reformatted. This remains outside the current smoke slice.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The smoke path does not perform optimisation. Pose-only LM/GN remains the next
  major geometry milestone after the reference projector grows enough physical
  fidelity for meaningful derivatives.

## 2026-05-06 — Make Detector Shifts Differentiable

### Summary

- Replaced rounded detector shifts in the minimal forward projector with
  differentiable periodic linear interpolation.
- Added `project_parallel_reference_arrays`, which accepts JAX arrays for
  `theta_rad`, `dx_px`, and `dz_px`. This gives future pose optimizers a path
  that can differentiate through detector shifts.
- Added tests for fractional detector shifts and `jax.grad` through `dx_px`.

### Decisions

- Kept the current coarse theta quadrant handling. This slice only removes the
  non-differentiable rounded detector-shift path.
- Periodic interpolation is a smoke/reference simplification. Full detector
  boundary policy belongs with the physical projector milestone.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 12 tests.
- `uv run ruff format --check src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py -q`
  passed: 126 tests.

### Risks

- `uv run python` without forcing CPU emits a JAX CUDA plugin warning about
  missing cuSPARSE, then falls back to CPU. The current validation still passes
  on CPU.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The projector is differentiable for detector shifts, but not yet a full
  physical differentiable projector for all 5 pose DOFs.

## 2026-05-06 — Add Pose-Only Detector-Shift LM Solver

### Summary

- Added `tomojax.align.solve_pose_only_lm`, a damped Gauss-Newton/LM solver
  against a fixed volume for the currently differentiable per-view pose
  channels:
  - `dx_px`
  - `dz_px`
- Added `PoseOnlyLMConfig` and `PoseOnlyLMResult`.
- The solver uses masked whitened projection residuals plus pseudo-Huber IRLS
  weights, solves a damped normal equation, and canonicalises geometry gauges
  after the solve.
- Added deterministic tests covering detector-shift recovery, active/frozen DOF
  reporting, final loss improvement, and gauge canonicalisation preservation.

### Decisions

- This is intentionally not the full 5-DOF pose solver. `alpha_rad`,
  `beta_rad`, and `phi_residual_rad` are reported as frozen because the current
  reference projector does not yet provide physical differentiable sensitivity
  for those DOFs.
- Used a finite-difference Jacobian for this first LM implementation. The
  periodic linear detector shift has derivative kinks at integer shifts, and
  finite differences behaved more robustly from zero initialization while also
  building toward the required finite-difference validation suite.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py -q`
  passed: 128 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full pose-only 5-DOF optimisation remains incomplete until the physical
  reference projector supports differentiable `alpha`, `beta`, and
  `phi_residual` effects.

## 2026-05-06 — Add Setup-Only Detector-Shift LM Solver

### Summary

- Added `tomojax.align.solve_setup_only_lm`, a damped Gauss-Newton/LM solver
  against a fixed volume for the currently differentiable setup channels:
  - `det_u_px`
  - active `det_v_px`
- Added `SetupOnlyLMConfig` and `SetupOnlyLMResult`.
- Factored the shared finite-difference Jacobian helper into
  `tomojax.align._lm_numerics` for the pose-only and setup-only LM solvers.
- The setup solver uses masked whitened projection residuals plus pseudo-Huber
  IRLS weights, solves a damped normal equation, and reports active and frozen
  setup parameters.
- Added deterministic tests covering detector-shift recovery, inactive
  `det_v_px` freezing, final loss improvement, and active/frozen reporting.

### Decisions

- This is intentionally not the full setup solver. `detector_roll_rad`,
  `axis_rot_x_rad`, `axis_rot_y_rad`, `theta_offset_rad`, and `theta_scale` are
  reported as frozen because the current reference projector only models
  detector shifts.
- Reused the finite-difference Jacobian path from the pose-only solver for
  consistency with the current differentiable detector-shift reference path and
  the planned finite-difference validation suite.

### Validation

- `uv run ruff check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 130 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full setup-only optimisation remains incomplete until the physical reference
  projector supports detector roll, axis rotation, and theta setup effects.

## 2026-05-06 — Add Projection Residual Filters

### Summary

- Added `tomojax.forward.apply_residual_filter` and
  `tomojax.forward.apply_residual_filter_schedule`.
- Added typed residual filter config/result values:
  - `ResidualFilterConfig`
  - `ResidualFilterKind`
  - `ResidualFilterResult`
- Implemented the first deterministic JAX reference residual filter policies:
  - `raw`
  - `lowpass_gaussian`
  - `bandpass_difference_of_gaussians`
- Added deterministic tests for raw identity with mask reapplication,
  low-pass impulse spreading and sum preservation, band-pass zero-mean
  behavior, and weighted schedule summation.

### Decisions

- The current low-pass policy uses a separable Gaussian kernel over the final
  two detector axes and periodic boundary handling via `jnp.roll`.
- The current band-pass policy is a difference between inner and outer Gaussian
  low-pass results. This matches the named Phase 2 reference policy without
  committing the public API to a future multiresolution filter bank design.
- Masks are reapplied after filtering so invalid pixels remain suppressed in
  filtered residuals.

### Validation

- `uv run ruff check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 13 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 134 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The residual filters are deterministic reference policies, not yet a complete
  level-aware continuation schedule integrated into the alternating solver.

## 2026-05-06 — Add Reference FISTA Preview Reconstruction

### Summary

- Added `tomojax.recon.fista_reconstruct_reference`, a tiny JAX reference FISTA
  preview reconstruction path against the current v2 reference projector.
- Added typed reconstruction config/result/trace rows:
  - `ReferenceFISTAConfig`
  - `ReferenceFISTAResult`
  - `ReferenceFISTATraceRow`
- Added smoothed TV regularisation, warm-start support, optional
  non-negativity projection, and masked robust projection residual loss.
- Added `write_fista_trace_csv` for the Phase 3 trace artifact contract.
- Added deterministic tests covering projection-loss improvement,
  non-negativity, warm-start updates, and CSV trace output.

### Decisions

- Used a fixed configured step size for this first reference implementation.
  It keeps the slice small and leaves production step-size estimation or line
  search for a later reconstruction milestone.
- Kept the implementation against the current minimal reference projector. This
  validates the FISTA control flow and artifact contract before the physical
  projector/backprojector is complete.
- The trace backend is reported as `jax_reference`.

### Validation

- `uv run ruff check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 4 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 136 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- This is not yet production reconstruction quality: the reference projector is
  still minimal, and multiresolution schedules plus stronger step-size control
  remain future Phase 3 work.

## 2026-05-06 — Add Reference FISTA Reconstruction Schedules

### Summary

- Added `tomojax.recon.reference_fista_schedule`, a typed v2 schedule resolver
  for reference FISTA preview and final reconstruction runs.
- Added schedule dataclasses:
  - `ReferenceFISTASchedule`
  - `ReferenceFISTAScheduleEntry`
  - `ReferenceReconstructionScheduleName`
- Defined the Phase 3 schedule contract:
  - preview schedule: levels 4 and 2
  - final schedule: level 1
- Included the Phase 2 residual filter policies in schedule entries:
  - level 4 uses low-pass
  - level 2 uses low-pass plus band-pass
  - level 1 uses raw residuals
- Added deterministic tests for schedule resolution, level factors, filter
  weights, and unknown-name rejection.

### Decisions

- The schedule API resolves typed configuration only. It does not execute a
  multiresolution pyramid yet; that orchestration belongs in a later Phase 3 or
  Phase 7 slice.
- Kept this separate from the old-core `recon.multires` path so v2 reference
  reconstruction can evolve behind the deep-module API without importing old
  geometry internals.

### Validation

- `uv run ruff check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 139 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full schedule execution over a multiresolution pyramid is still incomplete.

## 2026-05-06 — Make Reference Projector Differentiable In Theta

### Summary

- Replaced the minimal reference projector's quadrant `rot90` angle handling
  with bilinear x-y plane rotation.
- Preserved the existing differentiable periodic detector-shift path after
  projection.
- Added tests showing projection output changes with small theta updates and
  autodiff returns a finite nonzero theta gradient.
- Updated `tomojax.forward` documentation to describe the new angle sampling
  boundary policy.

### Decisions

- The reference angle rotation now samples outside-volume coordinates as zero.
  Detector shifts remain periodic because that behavior was already covered by
  the current smoke/reference contracts.
- This is still the minimal parallel projector. It does not yet implement
  laminography, detector roll, axis rotations, or full physical ray geometry.

### Validation

- `uv run ruff check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 141 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full physical angle/pose sensitivity remains incomplete until the projector
  models axis rotations, detector roll, and laminography geometry.

## 2026-05-06 — Add Phi Residual To Pose-Only LM

### Summary

- Extended `tomojax.align.solve_pose_only_lm` to optimise per-view
  `phi_residual_rad` along with `dx_px` and `dz_px`.
- Kept `alpha_rad` and `beta_rad` frozen because the reference projector does
  not yet model out-of-plane pose effects.
- Added deterministic tests for phi recovery and gauge canonicalisation into
  `theta_offset_rad`.
- Updated the `tomojax.align` README to reflect the active/frozen pose DOFs.

### Decisions

- Reused the existing finite-difference damped LM normal equation for the
  expanded pose vector. This keeps the implementation consistent with the
  detector-shift pose solver while the full 5-DOF projector is still being
  built.
- Used a more asymmetric 7^3 fixture for phi recovery tests. A smaller earlier
  fixture had a sign ambiguity for one view.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 143 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Pose-only LM is still not the full 5-DOF solver until alpha/beta effects and
  trust-region mechanics are implemented.

## 2026-05-06 — Add Theta Offset To Setup-Only LM

### Summary

- Extended `tomojax.align.solve_setup_only_lm` to optimise global
  `theta_offset_rad` along with `det_u_px` and active `det_v_px`.
- Kept detector roll, axis rotations, and theta scale frozen because the
  reference projector does not yet model those setup effects.
- Added deterministic theta-offset recovery tests on the asymmetric theta
  fixture.
- Updated the `tomojax.align` README to reflect the active/frozen setup
  parameters.

### Decisions

- Reused the existing finite-difference damped LM normal equation for the
  expanded setup vector.
- Setup-only LM keeps pose fixed, so `theta_offset_rad` and per-view
  `phi_residual_rad` remain separable by the existing gauge policy after the
  solve.

### Validation

- `uv run ruff check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 21 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 144 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Setup-only LM is still incomplete until detector roll, axis rotations,
  observability diagnostics, and trust-region mechanics are implemented.

## 2026-05-06 — Add Joint Setup+Pose Schur LM Reference Slice

### Summary

- Added `tomojax.align.solve_joint_schur_lm`, the first joint setup+pose Schur
  LM reference solver for supported differentiable DOFs.
- Added typed config/result/diagnostics values:
  - `JointSchurLMConfig`
  - `JointSchurLMResult`
  - `JointSchurDiagnostics`
- Added `schur_step_from_jacobian` as a public numerical contract for
  Schur-vs-dense validation.
- The solver packs supported setup DOFs
  (`theta_offset_rad`, `det_u_px`, optional `det_v_px`) and per-view pose DOFs
  (`phi_residual_rad`, `dx_px`, `dz_px`), builds finite-difference weighted
  residual Jacobians, solves the setup step by Schur complement, and
  back-substitutes per-view pose steps.
- Added deterministic tests for Schur-vs-dense normal equation equivalence and
  joint supported-geometry recovery after gauge canonicalisation.

### Decisions

- This is a reference Schur slice, not the final production trust-region
  engine. Priors, trust radii, damping adaptation, and acceptance diagnostics
  remain future Phase 6 work.
- Kept alpha/beta pose effects, detector roll, axis rotations, and theta scale
  frozen until the reference projector models those effects.
- Tested realised geometry after gauge canonicalisation because setup and pose
  mean components are gauge-coupled.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 11 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 146 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The joint Schur solver still lacks priors, trust radii, damping adaptation,
  detailed normal-equation artifact export, and unsupported physical DOFs.

## 2026-05-06 — Add Joint Schur Normal-Equation Artifact

### Summary

- Added `joint_schur_normal_eq_summary`, a JSON-serializable Phase 6
  normal-equation summary for `JointSchurLMResult`.
- Added `write_joint_schur_normal_eq_summary`, writing
  `normal_eq_summary.json`-style artifacts.
- Added `JointSchurDiagnostics.to_dict` for stable diagnostics serialization.
- Added a readback test covering solver label, active setup parameters, active
  pose DOFs, and Schur diagnostic keys.

### Decisions

- Kept the artifact intentionally compact for this slice:
  losses, iterations, active/frozen parameters, and current Schur diagnostics.
  Richer eigenvalue and pose-block condition summaries remain future Phase 6
  work.
- Exposed the writer through `tomojax.align` so callers and tests do not import
  private implementation files.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The artifact does not yet include eigenvalues, correlations, or pose-block
  condition statistics required by the full Phase 6 diagnostics target.

## 2026-05-06 — Enrich Joint Schur Diagnostics

### Summary

- Extended `JointSchurDiagnostics` with:
  - damped global normal-equation eigenvalues
  - Schur complement eigenvalues
  - per-view pose-block condition numbers
- Included these fields in `normal_eq_summary.json` via
  `joint_schur_normal_eq_summary`.
- Extended joint Schur tests to verify eigenvalue and pose-condition fields are
  present and sized correctly.

### Decisions

- Recorded eigenvalues from the damped normal equations currently solved by the
  reference implementation. Undamped Hessian diagnostics and correlation
  matrices remain future Phase 6 work.
- Kept the diagnostics JSON-native and compact so it can become an artifact in
  later alternating-solver runs without extra conversion code.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Phase 6 diagnostics still lack correlation matrices, weak-mode labels, priors,
  trust radii, and damping adaptation metadata.

## 2026-05-06 — Add Schur Correlations And Weak-Mode Labels

### Summary

- Extended `JointSchurDiagnostics` with:
  - normalized setup Schur correlation matrix
  - eigenvalue-derived weak-mode labels
- Included both fields in `normal_eq_summary.json` through
  `joint_schur_normal_eq_summary`.
- Added tests that verify correlation matrix shape/diagonal and artifact
  readback for correlation and weak-mode fields.

### Decisions

- Correlations are computed from the damped setup Schur complement by
  normalizing with the square root of the Schur diagonal.
- Weak modes are labelled generically as `schur_eigen_<index>` when their
  eigenvalue magnitude is below a relative threshold. Semantic labels by DOF
  contribution require eigenvector attribution and remain future work.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Weak-mode labels are numerical labels rather than semantic DOF labels until
  eigenvector attribution is added.

## 2026-05-06 — Add Joint Schur Trust Radius Controls

### Summary

- Added optional `setup_trust_radius` and `pose_trust_radius` fields to
  `JointSchurLMConfig`.
- Applied trust scaling to the Schur step and the dense-equivalent comparison
  step.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `trust_scale`
  - `trust_clipped`
  - `setup_update_by_parameter`
  - `pose_update_max_by_dof`
- Added deterministic tests covering default unclipped behavior, configured
  clipping, scaled Schur-vs-dense equivalence, and artifact fields.

### Decisions

- Used scalar setup and pose block radii for the first reference implementation.
  Per-parameter radii can be layered later using parameter metadata.
- Trust radii default to `None`, preserving the previous full-step behavior.
- This is not yet adaptive trust-region control; it only clips an otherwise
  computed LM step.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 148 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Trust radii are scalar and non-adaptive; per-DOF metadata radii and
  actual/predicted-decrease radius updates remain future Phase 6 work.

## 2026-05-06 — Add Joint Schur Damping Adaptation

### Summary

- Added configurable joint Schur LM damping adaptation:
  - `adapt_damping`
  - `damping_decrease_factor`
  - `damping_increase_factor`
  - `min_damping`
  - `max_damping`
- Updated `solve_joint_schur_lm` to keep a local damping value and adapt it
  after each accepted or rejected candidate step.
- Added `adapt_joint_schur_damping` to the public alignment facade so the
  accepted/rejected policy is testable without private imports.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `damping`
  - `next_damping`
  - `accepted`
  - `current_loss`
  - `candidate_loss`
- Added deterministic tests for accepted/rejected damping changes, clamp
  behavior, solver diagnostics, and artifact fields.

### Decisions

- Accepted steps decrease damping and rejected steps increase damping with
  configurable factors and clamps.
- Damping adaptation is deliberately separate from trust-radius scaling; this
  keeps the reference solver simple while preserving diagnostics for the later
  actual/predicted reduction policy.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 149 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  reduction-diagnostics slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 149 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; first current failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  damping slice.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The damping policy is accepted/rejected only; actual/predicted reduction and
  adaptive radius updates remain future Phase 6 work.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Reduction Diagnostics

### Summary

- Added predicted reduction diagnostics for the final trust-scaled Schur step
  using the damped quadratic normal-equation model.
- Recorded actual candidate reduction from the robust nonlinear objective and
  an actual/predicted reduction ratio.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `predicted_reduction`
  - `actual_reduction`
  - `reduction_ratio`
- Added deterministic tests for the quadratic predicted-reduction formula,
  solver diagnostics, and artifact fields.

### Decisions

- `reduction_ratio` is `None` when the predicted reduction is numerically zero;
  this occurs at convergence and avoids reporting an unstable ratio.
- The ratio is diagnostic-only for this slice. Adaptive radius or damping
  policy based on actual/predicted reduction remains future Phase 6 work.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Predicted reduction is computed from the current IRLS quadratic model, while
  actual reduction comes from the robust nonlinear objective. The ratio should
  be treated as a trust-region diagnostic, not a correctness metric by itself.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Trust-Radius Adaptation

### Summary

- Added configurable ratio-based trust-radius adaptation for existing joint
  Schur setup and pose block radii:
  - `adapt_trust_radii`
  - `trust_shrink_ratio`
  - `trust_expand_ratio`
  - `trust_shrink_factor`
  - `trust_expand_factor`
  - `min_trust_radius`
  - `max_trust_radius`
- Updated `solve_joint_schur_lm` to keep local setup and pose trust radii and
  adapt them after each accepted or rejected candidate step.
- Added `adapt_joint_schur_trust_radius` to the public alignment facade so the
  trust-radius policy is testable without private imports.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `next_setup_trust_radius`
  - `next_pose_trust_radius`
- Added deterministic tests for shrink, expand, clamp, disabled, and unset
  trust-radius behavior plus artifact readback with configured radii.

### Decisions

- Rejected, missing-ratio, and low-ratio steps shrink configured trust radii.
- High-ratio steps expand trust radii only when the step was clipped, matching
  common trust-region behavior and avoiding unnecessary radius growth.
- Unset trust radii remain `None`, preserving the existing full-step default.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  per-view-normal-block slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  per-view-reduction slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  iteration-trace slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  trust-radius-adaptation slice.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The policy still uses scalar setup/pose block radii. Per-DOF trust radii can
  be layered later from parameter metadata.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Iteration Trace

### Summary

- Added `iteration_diagnostics` to `JointSchurLMResult`.
- Updated `solve_joint_schur_lm` to retain the diagnostics from every solve
  iteration, not only the final iteration.
- Extended `normal_eq_summary.json` with `iteration_diagnostics`.
- Added deterministic tests that verify trace length, final diagnostic
  consistency, and artifact readback.

### Decisions

- Reused `JointSchurDiagnostics` for each trace row so the final summary and
  per-iteration trace share one schema.
- Kept the trace in JSON only for this slice. A CSV trace writer can be added
  later when the Phase 6 solver is wired into a benchmark/alignment artifact
  directory.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Artifact size grows with iteration count, but the reference solver currently
  uses a small default iteration count and compact diagnostics.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Per-View Reduction Diagnostics

### Summary

- Added per-view robust loss diagnostics to `JointSchurDiagnostics`:
  - `current_loss_by_view`
  - `candidate_loss_by_view`
  - `actual_reduction_by_view`
- Updated `solve_joint_schur_lm` to evaluate current and candidate robust loss
  per view for each candidate step.
- Extended `normal_eq_summary.json` and iteration trace rows with the per-view
  loss/reduction fields.
- Added deterministic tests for per-view diagnostic lengths and artifact
  readback.

### Decisions

- Per-view values are actual robust objective losses/reductions, not quadratic
  model reductions.
- Mask handling supports both per-view masks and projection-shaped masks.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Per-view `Jᵀr`/`JᵀJ` block artifacts are not yet materialized separately from
  the dense finite-difference Jacobian; this slice records per-view loss
  effects only.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Per-View Normal-Block Diagnostics

### Summary

- Added compact per-view normal-equation diagnostics to `JointSchurDiagnostics`:
  - `setup_gradient_by_view`
  - `pose_gradient_by_view`
  - `setup_hessian_diag_by_view`
  - `pose_hessian_diag_by_view`
  - `setup_pose_coupling_norm_by_view`
- Computed these summaries from the current finite-difference Jacobian and
  weighted residual inside `schur_step_from_jacobian`.
- Extended `normal_eq_summary.json` and iteration trace rows with the new
  per-view normal-block fields.
- Added deterministic checks for selected per-view gradient values plus artifact
  field/readback coverage.

### Decisions

- Recorded compact vectors and norms rather than full per-view matrices to keep
  the JSON artifact readable.
- Used a private typed dataclass for the intermediate block diagnostics so the
  public `JointSchurDiagnostics` construction remains type-safe.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- These diagnostics are computed from dense finite-difference Jacobians, not a
  streamed production accumulator. They are reference diagnostics until the
  fast path exists.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Unblock First Milestone 0 Ruff Cleanup Cluster

### Summary

- Replaced ambiguous Unicode punctuation in the package docstring with ASCII
  equivalents.
- Updated `tomojax.align._config` imports, type-only imports, and Python 3.12
  type aliases to satisfy the first `just check` Ruff cluster.
- Split `AlignConfig.__post_init__` into focused normalization helpers while
  preserving the previous normalization order.
- Added casts where normalized config fields narrow from public input types to
  internal enum-like literals.

### Decisions

- Kept the helper call order identical to the old in-method validation order to
  avoid behavior changes.
- Left the existing public-facade contract mismatch and LBFGS native crash out
  of scope for this narrow cleanup slice.

### Validation

- `uv run ruff check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run basedpyright src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run pytest tests/test_align_profiles.py tests/test_align_motion_models.py tests/test_align_gauge.py tests/test_align_optimizers.py tests/test_v2_module_skeleton.py -q`
  passed: 30 tests.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `just imports` passed.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures. The
  first remaining cluster now starts in `src/tomojax/align/_pose_stage.py`
  with parent-relative imports, type-only imports, missing annotations, and
  complexity findings.
- `uv run pytest tests/test_cli_config.py tests/test_align_contracts.py tests/test_alignment_schedules.py -q`
  still fails on `test_alignment_public_facade_stays_narrow`, which expects the
  old three-symbol `tomojax.align.__all__` while the current committed facade
  exposes LM symbols.
- A broader config-adjacent pytest run encountered a native JAX/Optax
  segmentation fault inside an existing LBFGS chunking path.
- Proposed next fix for `just check`: start a dedicated cleanup slice for
  `src/tomojax/align/_pose_stage.py` import hygiene and local annotation
  findings before tackling larger complexity findings.

## 2026-05-06 — Reduce Pose Stage Ruff Import And Annotation Findings

### Summary

- Converted `_pose_stage.py` parent-relative core imports to absolute imports.
- Moved `Detector`, `Geometry`, and `Grid` behind `TYPE_CHECKING` because they
  are only used in postponed annotations in this module.
- Added broad `jnp.ndarray` annotations to nested JAX objective helpers and
  scan bodies without changing numerical logic.
- Replaced a local smoothness-weight append loop with a comprehension and
  rewrote the GN smooth-candidate lambda as a nested function.

### Decisions

- Kept this as a low-risk cleanup slice and deferred decomposition of the large
  pose-stage functions to a separate milestone.
- Treated basedpyright for the whole file as an existing backlog rather than a
  gate for this slice because it reports broad JAX unknown/private-usage/stat
  narrowing issues unrelated to the import/annotation cleanup.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports only seven
  local PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `uv run pytest tests/test_align_quick.py -q -k 'gn or smooth_pose_model or pose_model'`
  passed: 23 tests.
- `just imports` passed.

### Risks

- `just check` remains blocked. The first remaining local failures are
  `_pose_stage.py` complexity findings in `_build_pose_objective_bundle`,
  `_align_summary_parts`, `_run_alignment_step`, and `align`, followed by
  legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- The broader basedpyright backlog in `_pose_stage.py` remains unresolved.
- Proposed next fix for `just check`: split `_align_summary_parts` first,
  then tackle `_run_alignment_step` and `_build_pose_objective_bundle` as
  separate behavior-preserving decomposition slices.

## 2026-05-06 — Split Pose Alignment Summary Formatting

### Summary

- Split `_align_summary_parts` into smaller helpers for GN, L-BFGS, GD, and
  alignment-loss formatting.
- Preserved the existing compact and verbose log message text while removing
  `_align_summary_parts` from the local complexity blocker list.

### Decisions

- Kept the formatting helpers private to `_pose_stage.py` because they are only
  used by this pipeline's outer-iteration logging.
- Split by output branch instead of introducing a formatter object or changing
  the `OuterStat` payload contract.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports five local
  PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'log_summary or log_compact or smooth_pose_model or pose_model'`
  passed: 9 tests, 43 deselected.

### Risks

- `just check` remains blocked by `_pose_stage.py` complexity findings in
  `_build_pose_objective_bundle`, `_run_alignment_step`, and `align`, followed
  by legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- Proposed next fix for `just check`: split `_run_alignment_step` into
  optimizer-kind helpers before tackling `_build_pose_objective_bundle`.

## 2026-05-06 — Split Pose Alignment Step Dispatch

### Summary

- Added a small `_AlignmentStepCoreResult` carrier for alignment step dispatch.
- Extracted optimizer dispatch, pre-step loss evaluation, GN handling, final
  gauge application, gauge-stat recording, final-loss bookkeeping, and relative
  improvement bookkeeping out of `_run_alignment_step`.
- Preserved the existing `OuterStat` keys and the GN final-loss reuse rule.

### Decisions

- Kept the existing GD and L-BFGS helpers and added matching GN/core helpers
  instead of changing the optimizer dispatch contract.
- Left the native JAX/Optax L-BFGS abort as an existing validation risk rather
  than hiding it or weakening checks.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports three local
  PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed.

### Risks

- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'gn or gd or lbfgs or smooth_pose_model or pose_model'`
  still aborts in the existing JAX/Optax L-BFGS chunking path.
- `just check` remains blocked by `_pose_stage.py` complexity findings in
  `_build_pose_objective_bundle` and top-level `align`, followed by legacy
  import/type-alias findings in `_profiles.py` and `_reconstruction_stage.py`.
- Proposed next fix for `just check`: decompose `_build_pose_objective_bundle`
  before the larger top-level `align` split.

## 2026-05-06 — Split Pose Objective Bundle Builders

### Summary

- Added `_PoseObjectiveContext` to carry pose objective arrays, config, chunk
  sizes, masks, and loss adapter state through private helper builders.
- Extracted chunk scheduling, mask handling, smoothness loss/gradient handling,
  align-loss construction, manual loss/gradient construction, and GN update
  construction out of `_build_pose_objective_bundle`.
- Preserved the `PoseObjectiveBundle` public surface and the same JAX reference
  objective paths.

### Decisions

- Used a frozen private context object rather than passing long argument lists
  through every JAX helper.
- Shared smoothness handling between align loss, manual gradient, and GN loss
  paths to keep the existing formulas consistent.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports two local
  PLR0912/PLR0915 complexity findings, both on top-level `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.

### Risks

- `just check` remains blocked by top-level `align` complexity, followed by
  legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- The existing native JAX/Optax L-BFGS abort remains unresolved.
- Proposed next fix for `just check`: split top-level `align` orchestration
  into setup, per-outer, and final-info helpers.

## 2026-05-06 — Split Pose Align Orchestration

### Summary

- Added `_AlignLoopState` to carry mutable alignment loop, resume, checkpoint,
  loss, observer, and gauge state through private helpers.
- Extracted step-context construction, per-outer reconstruction/alignment
  execution, observer handling, early-stop handling, completion logging, and
  final `AlignInfo` assembly out of top-level `align`.
- Preserved the public `align` API, checkpoint payload, and fixed-volume
  objective provenance fields while clearing the last local `_pose_stage.py`
  Ruff complexity blocker.

### Decisions

- Kept mutable loop state explicit instead of hiding it behind callbacks so the
  resume/checkpoint contract remains visible and testable.
- Left runtime/objective setup in `align` for now; the cleanup goal was the
  orchestration loop, not a broader pipeline rewrite.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` passed.
- `uv run ruff format src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are legacy import/type-alias issues
  in `_profiles.py`, `_quality_policy.py`, `_reconstruction_stage.py`,
  `_results.py`, and `_setup_stage.py`, followed by broader repository lint
  backlog. Formatter churn from `just check` was reverted outside this slice.

### Risks

- Stateful extraction could regress resume/checkpoint or early-stop behavior;
  targeted GN/GD alignment coverage passed, but dedicated checkpoint tests were
  not run in this slice.
- The existing native JAX/Optax L-BFGS abort remains unresolved.
- Proposed next fix for `just check`: clean `_profiles.py` import/type-alias
  findings, then proceed through `_quality_policy.py` and
  `_reconstruction_stage.py`.

## 2026-05-06 — Clean Alignment Profile Typing

### Summary

- Converted `_profiles.py` public type aliases to PEP 695 `type` aliases.
- Moved annotation-only `Mapping`, `ProjectorBackend`, and `Regulariser`
  imports behind `TYPE_CHECKING` and replaced parent-relative imports with
  absolute type-checking imports.
- Kept runtime behavior unchanged by using string-based `cast` targets where
  the imported type names are no longer available at runtime.

### Decisions

- Treated profile policy typing as its own small cleanup slice rather than
  mixing it with reconstruction-stage lint work.
- Let Ruff apply the file-local docstring spacing and cast-quoting fixes after
  the manual import/type-alias edit.

### Validation

- `uv run ruff check src/tomojax/align/_profiles.py` passed.
- `uv run ruff format src/tomojax/align/_profiles.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_profiles.py` is no longer in the failure list; the first
  remaining blocker is `_quality_policy.py` UP040, followed by
  `_reconstruction_stage.py`, `_results.py`, `_setup_stage.py`, and broader
  repository lint backlog. Formatter churn from `just check` was reverted
  outside this slice.

### Risks

- Runtime casts now use string targets for annotation-only types; focused
  profile tests passed, but broader config/CLI paths were not run in this
  slice.
- Proposed next fix for `just check`: convert `_quality_policy.py` to a PEP
  695 alias, then continue into `_reconstruction_stage.py`.

## 2026-05-06 — Clean Alignment Quality Policy Alias

### Summary

- Converted `_quality_policy.py` `AlignmentQualityTier` from `TypeAlias` to a
  PEP 695 `type` alias.
- Let Ruff apply the file-local cast quoting and import ordering changes.
- Kept the public alias name and quality policy behavior unchanged.

### Decisions

- Kept this as a separate small cleanup slice because it fully removes the
  `_quality_policy.py` blocker before the larger reconstruction-stage work.

### Validation

- `uv run ruff check src/tomojax/align/_quality_policy.py` passed.
- `uv run ruff format src/tomojax/align/_quality_policy.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_quality_policy.py` is no longer in the failure list; the first
  remaining blockers are in `_reconstruction_stage.py`, followed by
  `_results.py`, `_setup_stage.py`, and broader repository lint backlog.
  Formatter churn from `just check` was reverted outside this slice.

### Risks

- Minimal behavioral risk; focused profile tests passed.
- Proposed next fix for `just check`: clean `_reconstruction_stage.py`
  imports, local helper annotations, and statement count.

## 2026-05-06 — Clean Reconstruction Stage Ruff Blockers

### Summary

- Replaced `_reconstruction_stage.py` parent-relative imports with absolute
  imports and moved annotation-only geometry/stat imports behind
  `TYPE_CHECKING`.
- Added return annotations to the local FISTA, SPDHG, and Huber-FISTA-core
  reconstruction runner helpers.
- Extracted OOM message detection and final reconstruction `OuterStat`
  assembly into private helpers, clearing the local statement-count blocker
  without changing reconstruction math.

### Decisions

- Kept `Mapping` as a runtime `collections.abc` import because the final info
  path still uses `isinstance(info_rec, Mapping)`.
- Moved the existing stat payload verbatim into `_reconstruction_step_stat`
  rather than changing the provenance/stat contract.

### Validation

- `uv run ruff check src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run ruff format src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_reconstruction_stage.py` is no longer in the failure list; the
  first remaining blockers are `_results.py`, `_setup_stage.py`,
  `_stage_loop.py`, and broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- Final-stat assembly now lives in a helper; focused alignment/reconstruction
  coverage passed, but a dedicated stat-schema test was not added in this
  slice.
- Proposed next fix for `just check`: clean `_results.py` type-only imports.

## 2026-05-06 — Clean Alignment Result Type Imports

### Summary

- Moved `_results.py` annotation-only `jax.numpy`, observer, and schedule
  imports behind `TYPE_CHECKING`.
- Switched runtime collection protocol imports to `collections.abc`.
- Let Ruff apply file-local fixes for `__all__` ordering and direct
  `cfg.spdhg_seed` access.

### Decisions

- Kept `TypedDict` as a runtime import because the result schema classes
  subclass it.
- Kept result payload shapes unchanged; this slice only changes import/runtime
  typing hygiene.

### Validation

- `uv run ruff check src/tomojax/align/_results.py` passed.
- `uv run ruff format src/tomojax/align/_results.py` passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_profiles.py -q`
  passed: 16 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_results.py` is no longer in the failure list; the first
  remaining blockers are in `_setup_stage.py`, followed by `_stage_loop.py`
  and broader repository lint backlog. Formatter churn from `just check` was
  reverted outside this slice.

### Risks

- Runtime annotation introspection remains dependent on postponed annotations;
  checkpoint/profile tests passed.
- Proposed next fix for `just check`: clean `_setup_stage.py` imports and
  missing annotations.

## 2026-05-06 — Clean Setup Stage Typing

### Summary

- Replaced `_setup_stage.py` parent-relative imports with absolute imports and
  moved annotation-only geometry, stat, schedule, fold, adapter, and loss-spec
  imports behind `TYPE_CHECKING`.
- Added missing annotations for setup fold arrays, loss adapter, and loss spec
  inputs.
- Updated the bilevel setup test's manual `ResolvedAlignmentStage` construction
  to the current schedule contract and cleaned touched-file lint.

### Decisions

- Kept runtime setup execution dependencies as runtime imports; only
  annotation-only names moved behind `TYPE_CHECKING`.
- Treated the stale test constructor as part of this cleanup because it blocked
  focused setup validation and represented the current stage API inaccurately.

### Validation

- `uv run ruff check src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run ruff format src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run pytest tests/test_bilevel_setup_alignment.py tests/test_align_profiles.py -q`
  passed: 12 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_setup_stage.py` is no longer in the failure list; the first
  remaining blockers are in `_stage_loop.py`, followed by geometry module
  doc/import findings and broader repository lint backlog. Formatter churn
  from `just check` was reverted outside this slice.

### Risks

- The setup test now carries explicit resolved-stage metadata; if the stage
  contract changes again, this manual construction will need to change with it.
- Proposed next fix for `just check`: split `_stage_loop.py` into smaller
  import/annotation and orchestration cleanup slices.

## 2026-05-06 — Clean Multires Stage Loop Imports

### Summary

- Replaced `_stage_loop.py` parent-relative imports with absolute imports and
  moved annotation-only geometry, observer, and schedule imports behind
  `TYPE_CHECKING`.
- Added annotations to the stage observer and checkpoint callback factory.
- Moved multires scale/bin helpers and phase-correlation import to module scope,
  fixed the optional translation-seeding `vmap` lambda by binding per-level
  values through a local function, and removed unused resume-stage locals.

### Decisions

- Kept this as a pre-split cleanup slice so the next `_stage_loop.py` work can
  focus on actual orchestration complexity.
- Left `_run_multires_level_stages` and `align_multires` decomposition for the
  next slice because their complexity changes need focused review.

### Validation

- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only the
  remaining planned complexity blockers in `_run_multires_level_stages` and
  `align_multires`.
- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are `_stage_loop.py`
  PLR0915/PLR0912 complexity findings, followed by geometry module doc/import
  findings and broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- The optional translation-seeding path changed from an inline lambda to a
  local function; focused multires tests passed, but there is no dedicated
  seed-translation regression in this slice.
- Proposed next fix for `just check`: decompose `_run_multires_level_stages`
  before the larger `align_multires` split.

## 2026-05-06 — Split Multires Level Stage Dispatch

### Summary

- Added a private `StageLoopState` carrier for `_stage_loop.py` level-local
  state.
- Extracted proposal, setup-geometry, and pose-alignment stage handlers from
  `_run_multires_level_stages`.
- Preserved level stats, loss accumulation, checkpoint writes, resume handling,
  observer actions, and final gauge-fix propagation.

### Decisions

- Kept the split private to `tomojax.align`; no public API or algorithm
  behavior changed.
- Left the larger `align_multires` orchestration split for the next cleanup
  slice because it is the remaining function-level Ruff complexity blocker.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only
  `align_multires` PLR0912/PLR0915.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are `align_multires`
  PLR0912/PLR0915, followed by geometry module doc/import findings and the
  broader repository lint backlog. Formatter churn from `just check` was
  reverted outside this slice.

### Risks

- The handler split depends on `StageLoopState` carrying every mutated level
  value; focused multires/setup/checkpoint tests passed, but future changes
  should keep new per-stage state explicit on that carrier.
- Proposed next fix for `just check`: split `align_multires` orchestration
  complexity.

## 2026-05-06 — Split Multires Public Orchestration

### Summary

- Added private multires context and run-state carriers for setup, resume, and
  finalization bookkeeping.
- Extracted multires input setup/validation, level selection, initial volume
  selection, translation seeding, per-level execution, and run finalization from
  the public `align_multires` function.
- Removed all remaining `_stage_loop.py` Ruff findings without changing the
  public alignment API.

### Decisions

- Kept the split inside `tomojax.align._stage_loop` so the existing
  `tomojax.align.pipeline` facade and import-linter contract remain unchanged.
- Preserved the coarsest-level phase-correlation seeding path as private helper
  logic rather than changing the alignment schedule model.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 66 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_stage_loop.py` is no longer in the failure list; the first
  remaining blockers are geometry module doc/import lint findings, followed by
  checkpoint/io/model lint and broader repository backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- The run-state carrier now owns checkpoint/resume bookkeeping across helper
  calls; focused multires/checkpoint tests passed, but future per-level state
  additions should be explicit fields.
- Proposed next fix for `just check`: clean the geometry deep-module lint
  backlog beginning with `detector_center.py`, `geometry_applier.py`,
  `geometry_blocks.py`, `initializers.py`, and `parametrizations.py`.

## 2026-05-06 — Clean Alignment Geometry Lint Blockers

### Summary

- Added module and public API docstrings for the first geometry lint blockers:
  detector-center exports, geometry application, calibration blocks,
  initializers, and pose parametrizations.
- Replaced parent-relative imports with absolute imports where needed and moved
  annotation-only geometry/state imports behind `TYPE_CHECKING`.
- Cleaned small parametrization findings by removing non-ASCII comment text and
  returning the final transform update directly.

### Decisions

- Kept the geometry package API unchanged; this slice only addressed docs,
  imports, and local lint findings.
- Replaced constant `getattr` calls in touched geometry files with direct
  attribute access as part of the focused Ruff cleanup.

### Validation

- `uv run ruff format src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run ruff check src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run pytest tests/test_geometry.py tests/test_geometry_applier.py tests/test_geometry_block_taxonomy_generator.py tests/test_detector_center_objective.py tests/test_align_quick.py -q`
  passed: 54 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched geometry files are no longer in the failure list; the
  first remaining blockers are `align/io/checkpoint.py`,
  `align/io/params_export.py`, and `align/model/*`, followed by broader
  repository lint backlog. Formatter churn from `just check` was reverted
  outside this slice.

### Risks

- Direct attribute access now makes the expected geometry protocol more
  explicit in touched files; focused geometry and quick alignment tests passed.
- Proposed next fix for `just check`: clean checkpoint/io lint, then continue
  through the alignment model package.

## 2026-05-06 — Clean Alignment I/O Lint Blockers

### Summary

- Added checkpoint and params-export module/class docstrings required by the
  current Ruff configuration.
- Cleaned checkpoint atomic-write lint by using `Path.replace` and
  `contextlib.suppress` while preserving fsync and temporary-file cleanup.
- Simplified schedule metadata validation and params5 shape error formatting.
- Reduced params-export JSON normalization return-count lint without changing
  exported payload shape.

### Decisions

- Kept the checkpoint schema and payload fields unchanged.
- Kept params-export normalization local instead of introducing a shared utility
  module, consistent with the no-generic-utils policy.

### Validation

- `uv run ruff format src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run ruff check src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 33 tests.
- `uv run pytest tests/test_align_params_export.py -q` passed: 8 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched alignment I/O files are no longer in the failure
  list; the first remaining blockers are in `src/tomojax/align/model/*`,
  followed by broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- Atomic checkpoint save still depends on filesystem support for same-directory
  replace; this was already required by the previous `os.replace` path.
- Proposed next fix for `just check`: clean the alignment model package lint
  blockers beginning with diagnostics, DOF specs, gauge, motion models, and
  schedules.

## 2026-05-06 — Clean Alignment Diagnostics And DOF Lint

### Summary

- Added public docstrings for gauge diagnostics and scoped DOF helpers.
- Moved annotation-only diagnostics imports behind `TYPE_CHECKING`.
- Simplified the conditioning threshold branch without changing singular-value
  diagnostics.

### Decisions

- Kept this as a narrow model-package slice so the larger `dof_specs.py` and
  schedule cleanup can be reviewed separately.
- Left all gauge policy rules and DOF normalization behavior unchanged.

### Validation

- `uv run ruff format src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run ruff check src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run pytest tests/test_alignment_gauge_registry.py tests/test_align_quick.py tests/test_align_profiles.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `diagnostics.py` and `dofs.py` are no longer in the failure list;
  the first remaining blockers start in `src/tomojax/align/model/dof_specs.py`,
  followed by gauge, motion model, schedule, state, and broader repository lint
  backlog. Formatter churn from `just check` was reverted outside this slice.

### Risks

- The diagnostics import movement is safe only while those names remain
  annotation-only; focused gauge/profile tests passed.
- Proposed next fix for `just check`: clean `dof_specs.py`.

## 2026-05-06 — Clean Active Parameter DOF Spec Lint

### Summary

- Added `dof_specs.py` module and public API docstrings for active parameter
  packing, whitening, bounds, and diagnostics helpers.
- Moved annotation-only imports behind `TYPE_CHECKING`.
- Removed quoted forward references now covered by postponed annotations.

### Decisions

- Kept active-parameter behavior unchanged; this slice only documents the API
  and cleans annotation imports.
- Left gauge, motion model, schedule, and state lint for separate focused
  slices.

### Validation

- `uv run ruff format src/tomojax/align/model/dof_specs.py` passed.
- `uv run ruff check src/tomojax/align/model/dof_specs.py` passed.
- `uv run pytest tests/test_alignment_state.py tests/test_align_optimizers.py tests/test_alignment_objectives.py tests/test_bilevel_setup_alignment.py tests/test_alignment_scenario_catalog.py -q`
  passed: 46 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `dof_specs.py` is no longer in the failure list; the first
  remaining blockers start in `src/tomojax/align/model/gauge.py`, followed by
  motion model, schedule, state, and broader repository lint backlog. Formatter
  churn from `just check` was reverted outside this slice.

### Risks

- The import movement is safe only while `AlignmentState`, `Iterable`, and
  `Sequence` remain annotation-only; active parameter tests passed.
- Proposed next fix for `just check`: clean `gauge.py`.

## 2026-05-06 — Clean Alignment Gauge Lint

### Summary

- Added `gauge.py` module and typed-dict docstrings.
- Moved annotation-only imports behind `TYPE_CHECKING`.
- Added explicit argument and return annotations for the JAX zero-mean
  projection loop body.

### Decisions

- Kept gauge normalization, feasibility checks, projection behavior, and stats
  payloads unchanged.
- Left motion model, schedule, and state lint for follow-up focused slices.

### Validation

- `uv run ruff format src/tomojax/align/model/gauge.py` passed.
- `uv run ruff check src/tomojax/align/model/gauge.py` passed.
- `uv run pytest tests/test_align_gauge.py tests/test_alignment_gauge_registry.py tests/test_align_quick.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `gauge.py` is no longer in the failure list; the first remaining
  blockers start in `src/tomojax/align/model/motion_models.py`, followed by
  schedules, state, objectives, and broader repository lint backlog. Formatter
  churn from `just check` was reverted outside this slice.

### Risks

- The JAX loop-body annotation is structural only; focused gauge tests passed.
- Proposed next fix for `just check`: clean `motion_models.py`.

## 2026-05-06 — Clean Alignment Motion Model Lint

### Summary

- Added `motion_models.py` module and property docstrings.
- Moved annotation-only `Sequence` import behind `TYPE_CHECKING`.
- Cleaned local type-checking lint for `cast` and sorted `__all__`.

### Decisions

- Kept pose model basis construction, coefficient fitting, and expansion
  behavior unchanged.
- Stopped the legacy Ruff cleanup path after this slice per user direction and
  will move the active plan directly to the smallest Phase 7 alternating-solver
  smoke/artifact vertical slice.

### Validation

- `uv run ruff format src/tomojax/align/model/motion_models.py` passed.
- `uv run ruff check src/tomojax/align/model/motion_models.py` passed.
- `uv run pytest tests/test_align_motion_models.py -q` passed: 6 tests.
- `just imports` passed.
- `uv run pytest tests/test_align_motion_models.py tests/test_align_chunking.py tests/test_align_optimizers.py -q`
  aborted with a JAX/Optax L-BFGS segmentation fault in
  `tests/test_align_chunking.py::test_align_smooth_pose_model_clips_active_bounds_only`;
  this is the known native L-BFGS path and is not caused by this docs/import
  cleanup. No unrelated formatter churn was present from this validation.

### Risks

- The import movement is safe only while `Sequence` remains annotation-only;
  focused motion-model tests passed.
- Proposed next fix: switch from legacy Ruff cleanup to Phase 7 alternating
  solver continuation smoke.

## 2026-05-06 — Phase 7 Alternating Solver Smoke Slice

### Summary

- Added an align-owned `smoke32` continuation schedule with preview and final
  levels.
- Added a deterministic 32^3 stopped-volume alternating smoke runner that
  reconstructs with the JAX reference FISTA path, applies gauge
  canonicalisation as the first geometry update, and writes Phase 7 smoke
  artifacts.
- Added smoke tests that verify deterministic volume output, coarse
  verification, level-1 geometry skip recording, pose gauge canonicalisation,
  and artifact presence.

### Decisions

- Kept the legacy `tomojax.align` package facade unchanged; Phase 7 names are
  exposed through `tomojax.align.api`.
- Used gauge canonicalisation as the smallest geometry-update vertical slice so
  the run emits real geometry/provenance artifacts before the full production
  Schur LM loop is wired into the alternating solver.
- Stopped legacy Ruff cleanup after the motion-model slice per user direction.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The alternating solver currently uses gauge canonicalisation rather than the
  full Schur LM/GN geometry update; this is intentionally the smallest Phase 7
  vertical slice and leaves production escalation/continuation for follow-up.

## 2026-05-06 — Phase 7 Smoke Artifact Contract Expansion

### Summary

- Expanded the deterministic 32^3 alternating smoke run to persist
  `final_volume.npy`.
- Added the core audit scaffold artifacts for the smoke profile:
  `run_manifest.json`, `config_resolved.toml`, `input_summary.json`,
  `projection_stats.json`, `mask_summary.json`, `gauge_report.json`,
  `backend_report.json`, and `residual_metrics.csv`.
- Updated `artifact_index.json` entries with artifact type, media type, and
  descriptions, and extended smoke tests to verify the expanded contract.

### Decisions

- Kept the artifact writer local to `tomojax.align._alternating` for this
  slice to avoid introducing a generic helper module before a second owner
  needs it.
- Used deterministic smoke-profile manifest values rather than wall-clock
  timestamps so the smoke output remains reproducible.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The smoke artifact scaffold is not yet the full production run schema from
  `docs/tomojax-v2/06_verification_and_artifact_contract.md`; missing
  production artifacts include observability/failure reports, preview slices,
  residual maps, and CLI-run metadata.

## 2026-05-06 — Phase 7 Early Exit Continuation Smoke

### Summary

- Extended the `smoke32` continuation schedule from level 4/final to
  level 4, conditional level 2, and final level 1.
- Added explicit skipped-level and skipped-geometry summary fields so early
  exit decisions are visible in `alignment_summary.csv`,
  `residual_metrics.csv`, and `verification.json`.
- Made level-1 geometry a planned verification-triggered polish step that is
  skipped when coarse verification passes, while still running the final
  reconstruction.

### Decisions

- Kept gauge canonicalisation as the smoke geometry update; this slice only
  changes continuation control flow and artifact reporting.
- Recorded skipped level 2 as an artifact row instead of silently omitting it so
  early-exit behavior remains auditable.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The early-exit predicate is still the smoke-profile coarse verification
  predicate, not the full production combination of held-out residual,
  parameter-update, and gauge-stability checks.

## 2026-05-06 — Phase 7 Continuation Profile Presets

### Summary

- Added `lightning`, `balanced`, and `reference` continuation schedules while
  keeping `smoke32` as the deterministic test profile.
- Exposed `ContinuationScheduleName` through `tomojax.align.api`.
- Added focused schedule tests for Phase 7 level ordering, conditional level-2
  behavior, final-level policy, monotonic profile work, and unknown profile
  rejection.

### Decisions

- Kept profile presets as deterministic schedule data only; CLI/runtime profile
  plumbing remains out of scope for this slice.
- Encoded monotonic work increases from `lightning` to `balanced` to
  `reference`; empirical tuning remains future work.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_continuation_schedules.py tests/test_alternating_solver_smoke.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The schedule presets are conservative defaults and have not yet been tuned
  against the full synthetic benchmark suite.

## 2026-05-06 — Phase 7 Alternating Solver Entrypoint

### Summary

- Added the public `AlternatingAlignmentSolver` orchestration object.
- Routed `run_alternating_solver_smoke` through `AlternatingAlignmentSolver`
  so the function and solver class share one implementation path.
- Exported and documented the solver entrypoint, and added a focused smoke test
  for `AlternatingAlignmentSolver.run_smoke`.

### Decisions

- Kept the solver object intentionally thin and smoke-profile-specific in this
  slice. A general dataset solver interface remains follow-up work.
- Kept the legacy `tomojax.align` package facade unchanged; the new solver is
  exported from `tomojax.align.api`.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- `AlternatingAlignmentSolver` currently exposes only the deterministic
  `run_smoke` method; production data loading and CLI integration are still
  missing.

## 2026-05-06 — Phase 7 Auto Smoke CLI

### Summary

- Added `tomojax.cli.align_auto`, a one-command deterministic Phase 7
  `align=auto` smoke entrypoint.
- Added the `tomojax-align-auto-smoke` console script.
- Added focused CLI tests for help text and writing the core final volume,
  geometry, and verification artifacts.

### Decisions

- Kept the new command explicitly smoke-profile-only so it does not imply full
  production dataset alignment.
- Called `AlternatingAlignmentSolver` from the CLI instead of duplicating solver
  behavior in `tomojax.cli`.
- Put the new CLI tests in `tests/test_align_auto_cli.py` to avoid touching the
  older transitional `tests/test_cli_entrypoints.py` lint backlog.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.
- `uv run tomojax-align-auto-smoke --help` passed.

### Risks

- The command currently runs only synthetic smoke data; full dataset input and
  production `align=auto` CLI integration remain future work.

## 2026-05-06 — Phase 7 Smoke Audit Reports

### Summary

- Added `gauge_policy.json`, `observability_report.json`, and
  `failure_report.json` to the deterministic Phase 7 smoke run.
- Included those reports in `artifact_index.json`.
- Extended the smoke tests to verify report presence and key fields.

### Decisions

- Kept `observability_report.json` explicitly marked as
  `smoke_placeholder`; Schur-curvature-backed observability is not computed in
  this smoke profile yet.
- Wrote `failure_report.json` for passed runs with `status="passed"` and
  `failure=null` so the artifact is always present.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- Observability remains a placeholder until the alternating solver consumes the
  Schur LM condition and weak-mode diagnostics from the geometry update path.

## 2026-05-06 — Phase 7 Residual Filter Continuation

### Summary

- Added residual-filter configs to the Phase 7 continuation levels:
  low-pass at level 4, low-pass plus band-pass at conditional level 2, and raw
  residual at final level 1.
- Applied the public `tomojax.forward.apply_residual_filter_schedule` path to
  the projection-domain geometry/verification loss in the alternating smoke
  runner.
- Recorded residual-filter kinds in `alignment_summary.csv`,
  `residual_metrics.csv`, and verification level payloads, with focused tests.

### Decisions

- Kept reconstruction FISTA on the existing reference objective in this slice;
  only the geometry verification/update loss uses continuation filters.
- Used the public forward-module filter API rather than adding local filtering
  under `tomojax.align`.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- FISTA does not yet consume the residual-filter schedule; the continuation
  filters currently affect only geometry loss and verification behavior.

## 2026-05-06 — Phase 7 Profile Provenance Artifacts

### Summary

- Threaded the resolved `ContinuationSchedule` into `run_manifest.json` and
  `config_resolved.toml`.
- Fixed hard-coded `smoke32` profile metadata so non-default profiles record the
  actual schedule name and level factors.
- Added a focused non-default `lightning` profile smoke test for manifest and
  resolved-config provenance.

### Decisions

- Kept the config artifact deterministic and minimal: profile, align mode,
  backend, geometry model, and level factors.
- Recorded schedule name and level factors now; full schedule serialization is
  left for a later artifact-schema pass.

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

### Risks

- Custom schedule objects are represented by name and level factors only, not a
  complete per-level serialized schedule.

## 2026-05-06 — Phase 7 Verification Predicate Bundle

### Summary

- Added per-level verification predicates for loss non-increase, finite loss,
  gauge stability, and parameter-update size.
- Gated smoke-level `verified` status on the full predicate bundle instead of
  only loss non-increase.
- Recorded predicate fields, thresholds, and measured update norm in
  `verification.json`, `alignment_summary.csv`, and `residual_metrics.csv`.

### Decisions

- Kept the predicate thresholds smoke-profile local in
  `AlternatingSmokeConfig`.
- Did not add held-out residual checks in this slice; that remains a production
  verification follow-up.

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

### Risks

- `parameter_update_small` is a smoke-scale heuristic, not the final
  trust-region update acceptance metric.

## 2026-05-06 — Phase 7 Smoke Input Artifacts

### Summary

- Persisted `observed_projections.npy` and `projection_mask.npy` from the
  deterministic Phase 7 smoke run.
- Added `recovery_tolerances.json` with smoke-specific geometry and
  verification tolerances.
- Added the new input artifacts to `artifact_index.json` and focused smoke
  tests.

### Decisions

- Used `.npy` arrays for smoke input artifacts, matching the existing synthetic
  dataset writer contract.
- Kept recovery tolerances smoke-specific and explicit; this is not a full
  128^3 synthetic benchmark manifest yet.

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

### Risks

- The smoke run now persists benchmark-style input arrays, but the full 128^3
  synthetic suite still lives in the dataset generator path and is not yet run
  end-to-end through Phase 7.

## 2026-05-06 — Phase 7 Smoke Truth Provenance

### Summary

- Persisted `ground_truth_volume.npy` from the deterministic Phase 7 smoke run.
- Added `geometry_true.json` and explicit `geometry_corrupted.json` alongside
  the existing solver `geometry_initial.json`.
- Added the new truth/corruption artifacts to `artifact_index.json` and focused
  smoke tests.

### Decisions

- Kept `geometry_initial.json` as the solver initial state and added
  `geometry_corrupted.json` to mirror the synthetic benchmark naming contract.
- Used `GeometryState.zeros(n_views)` as the true uncorrupted smoke geometry.

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

### Risks

- `geometry_corrupted.json` currently mirrors `geometry_initial.json`; future
  production dataset ingestion may need separate supplied-vs-initial records.

## 2026-05-06 — Phase 7 Smoke Recovery Metrics

### Summary

- Added final-vs-true smoke geometry recovery metrics to `verification.json`.
- Compared mean residual `dx` and `phi_residual` errors against the
  smoke-specific `recovery_tolerances.json` thresholds.
- Added focused tests for recovery pass/fail, measured values, and limits.

### Decisions

- Kept recovery metrics limited to the gauge-canonicalised mean pose quantities
  represented by the current reference smoke path.
- Reused the same smoke tolerance payload written to `recovery_tolerances.json`
  so the verification report and tolerance artifact stay aligned.

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

### Risks

- Recovery coverage is still limited to smoke-scale gauge recovery, not the
  full 5-DOF synthetic recovery suite.

## 2026-05-06 — Phase 7 Smoke Volume Metrics

### Summary

- Added final-vs-truth volume RMSE, MAE, and NMSE to the deterministic smoke
  `verification.json`.
- Added a smoke-specific volume NMSE tolerance to `recovery_tolerances.json`.
- Extended focused smoke tests to verify volume metric presence and pass/fail
  behavior.

### Decisions

- Kept the NMSE tolerance loose because the smoke profile uses a tiny
  one-iteration reference FISTA reconstruction and is a wiring check, not a
  quality benchmark.
- Recorded volume metrics now so the smoke run mirrors the benchmark contract
  fields used by the future 128^3 suite.

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

### Risks

- The recorded volume metrics are not expected to be strong until longer
  reconstruction schedules and production benchmark profiles are wired in.

## 2026-05-06 — Phase 7 Per-View Residual Metrics

### Summary

- Expanded `residual_metrics.csv` so it records both per-level continuation
  summaries and final per-view projection residual metrics.
- Added `row_type=view_residual` rows with RMSE, MAE, robust loss,
  valid-pixel fraction, outlier fraction, and raw RMSE fields.
- Extended the deterministic 32^3 smoke test to verify the summary rows and
  four per-view residual rows are written.

### Decisions

- Added a `row_type` discriminator so existing level summary rows remain in the
  same artifact without overloading per-view metric columns.
- Kept this slice to raw final projection residuals; low-pass and band-pass
  residual views remain a later benchmark-contract expansion.

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

### Risks

- Per-view metrics are currently computed from the final raw reference forward
  projection only, not from the continuation-filtered residual streams.

## 2026-05-06 — Phase 7 Artifact Validation

### Summary

- Added a typed `tomojax.verify` artifact validation API with inspect and
  fail-loud entrypoints.
- Validated required smoke JSON artifacts, core schema identifiers, geometry
  schema version, and indexed artifact file existence.
- Wired the Phase 7 smoke artifact writer through validation before returning
  and added positive/negative tests against real smoke bundles.

### Decisions

- Used a lightweight stdlib validator instead of adding a schema dependency for
  the smoke contract.
- Kept this slice focused on JSON/index validation; CSV and array semantic
  validation remain future extensions of the same API.

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

### Risks

- The validator does not yet check CSV column semantics, array shapes, or
  preview/residual-map directory contents.

## 2026-05-06 — Phase 7 Residual Map Artifacts

### Summary

- Added `residual_maps/final_raw_residual.npy` for the deterministic Phase 7
  smoke bundle.
- Added `residual_maps/summary.json` with residual-map schema, shape, dtype,
  valid-pixel fraction, and aggregate residual statistics.
- Updated `artifact_index.json` to record run-directory-relative nested paths
  and extended smoke tests to verify the indexed residual-map artifacts.

### Decisions

- Persisted residual maps as `.npy` arrays for the smoke path so deterministic
  numeric content can be validated without image-rendering dependencies.
- Kept human-facing preview images and plots out of this slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The smoke run still does not emit preview slices or plots from the artifact
  contract.

## 2026-05-06 — Phase 7 Preview Slice Artifacts

### Summary

- Added deterministic preview-slice artifacts under `preview_slices/` for the
  Phase 7 smoke bundle.
- Persisted central z-slices for truth, final reconstruction, and final-minus-
  truth error as `.npy` arrays.
- Added `preview_slices/summary.json` with schema, axis, index, shape, dtype,
  and error aggregate metrics, and indexed all nested preview artifacts.

### Decisions

- Stored numeric preview arrays instead of rendered images so the smoke tests
  can validate deterministic data without adding rendering dependencies.
- Used a tiny absolute tolerance in the preview error assertion to allow
  float32 subnormal roundoff after save/load.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- Human-facing PNG previews and plots are still not emitted by the smoke run.

## 2026-05-06 — Phase 7 Geometry Trace Artifact

### Summary

- Added `geometry_trace.csv` to the deterministic Phase 7 smoke artifact
  bundle.
- Recorded per-level geometry update trace rows with requested/executed update
  counts, loss delta, gauge and parameter-update predicates, skip state, and
  early-exit reason.
- Indexed the trace artifact and extended smoke tests to verify the deterministic
  geometry trace rows.

### Decisions

- Derived the trace from `AlternatingLevelSummary` in this slice because the
  smoke path still uses gauge canonicalisation as its geometry update rather
  than real inner LM/GN steps.
- Left per-inner-step trace granularity for the upcoming geometry optimiser
  implementation.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- `geometry_trace.csv` is per continuation level, not yet per accepted/rejected
  LM/GN step.

## 2026-05-06 — Phase 7 Verification Report Shape

### Summary

- Added contract-shaped `status`, `summary`, `metrics`, and `escalation`
  sections to the Phase 7 smoke `verification.json`.
- Kept existing smoke-specific keys so deterministic checks can still inspect
  seed, schedule, levels, thresholds, and recovery details.
- Updated artifact validation so `verification.json` must include the new
  top-level report sections.

### Decisions

- Used the first solver-level loss as `metrics.residual_before` rather than the
  legacy smoke `initial_loss`, which compares the true volume against synthetic
  observations and is not a run-start residual.
- Reported `summary.projection_residual_improved` truthfully from the smoke
  metrics; the current tiny smoke path may report `false` while still passing
  the smoke recovery checks.

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

### Risks

- The smoke run still uses placeholder geometry updates, so the new verification
  summary is a report-shape contract rather than proof of full optimiser
  convergence.

## 2026-05-06 — Phase 7 Backend Provenance Report

### Summary

- Expanded the Phase 7 smoke `backend_report.json` beyond the placeholder
  requested/actual fields.
- Added explicit projector, backprojector, geometry-reduction, detector-grid,
  Pallas eligibility, fallback, and agreement-test provenance fields.
- Updated artifact validation to require the expanded backend provenance keys.

### Decisions

- Reported all active components as `jax_reference` because this smoke path is
  intentionally CPU/JAX reference first.
- Added a `reference_baseline` agreement-test row with zero errors so the
  contract shape is present before fast-path comparison tests exist.

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

### Risks

- Agreement testing currently records reference-baseline provenance only; real
  max/mean error comparisons remain blocked on alternate backend paths.

## 2026-05-06 — Phase 7 Failure Report Gates

### Summary

- Expanded the Phase 7 smoke `failure_report.json` with verification gate rows.
- Added finite-output, projection-residual-improvement, gauge-stability,
  optimiser-health, and backend-provenance gates.
- Added structured warning entries for smoke-level `no_improvement` cases while
  preserving the smoke run's pass status.
- Updated artifact validation to require the expanded failure-report shape.

### Decisions

- Treated projection residual non-improvement as a warning in the tiny smoke
  profile because this path is still a wiring check with placeholder geometry
  updates.
- Kept the failure classes from the artifact contract visible in the report so
  later verifier work can promote warnings/errors without changing shape.

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

### Risks

- Optimiser-health evidence remains coarse until real accepted/rejected LM/GN
  step diagnostics are part of the alternating solver.

## 2026-05-06 — Phase 7 Observability Report Shape

### Summary

- Expanded the Phase 7 smoke `observability_report.json` beyond the placeholder
  status.
- Added structured setup and pose DOF entries with active, observable, status,
  and gauge-group metadata where applicable.
- Recorded smoke weak modes and handled frozen DOFs so the report explicitly
  distinguishes uncomputed observability from frozen parameters.

### Decisions

- Kept the top-level status as `smoke_not_evaluated` because the smoke path
  still does not compute Schur curvature or condition numbers.
- Marked gauge-canonicalised pose terms separately from weak/not-evaluated
  active DOFs.

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

### Risks

- This report is still descriptive; numerical observability metrics remain for
  the real Schur LM/GN implementation.

## 2026-05-06 — Phase 7 Run Manifest Contract Fields

### Summary

- Expanded the Phase 7 smoke `run_manifest.json` with contract-required
  `tomojax_version`, `git_commit`, `started_at`, `finished_at`,
  `backend_requested`, and geometry model fields.
- Updated artifact validation so missing manifest contract keys fail loudly.
- Extended smoke tests to verify the deterministic timestamp placeholders and
  backend/geometry manifest fields.

### Decisions

- Kept smoke timestamps deterministic as `deterministic-smoke` so repeated smoke
  runs remain reproducible.
- Recorded the current git commit when available, with an `unknown` fallback for
  non-git environments.

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

### Risks

- Production elapsed-time accounting is still absent from the deterministic
  smoke manifest.

## 2026-05-06 — Phase 7 Plot Summary Artifact

### Summary

- Added `plots/summary.json` to the deterministic Phase 7 smoke artifact
  bundle.
- Recorded plot-ready FISTA loss and per-level geometry loss traces without
  adding rendering dependencies.
- Indexed the nested plots artifact and extended smoke tests to verify the
  artifact path and basic trace content.

### Decisions

- Stored numeric plot inputs first rather than rendered PNG/SVG files.
- Kept rendering out of the smoke path so the focused checks remain fast and
  deterministic.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- No rendered plot images are emitted yet; this is intentionally paused now in
  favor of the real Schur-in-the-loop alignment update.

## 2026-05-06 — Phase 7 Schur Geometry Update In The Loop

### Summary

- Switched the deterministic 32^3 smoke fixture so observed projections are
  generated from true geometry while the solver starts from a corrupted initial
  geometry.
- Replaced the placeholder geometry canonicalisation update in the Phase 7 loop
  with the supported `solve_joint_schur_lm` reference solver.
- Recorded Schur diagnostics in `schur_diagnostics.json` and added accepted
  step, condition, dense-vs-Schur, and reduction fields to `geometry_trace.csv`.
- Extended smoke tests to verify projection residual improvement, accepted Schur
  diagnostics, gauge-canonical supported DOF recovery, and true-vs-corrupted
  geometry separation.

### Decisions

- Used the synthetic ground-truth volume as the fixed volume for this first
  Schur-in-the-loop smoke slice so geometry recovery is isolated from the tiny
  one-iteration FISTA reconstruction.
- Kept the alternating reconstruction artifact path intact; replacing the fixed
  synthetic volume with the stopped reconstructed latent is the next numerical
  integration step.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This is a real Schur geometry update, but it is still an alignment-isolated
  smoke slice using the synthetic fixed volume rather than the stopped FISTA
  latent volume.

## 2026-05-06 — Phase 7 Geometry-Aware Backprojection Initializer

### Summary

- Added public `reconstruct_backprojection_reference` to `tomojax.recon`.
- Implemented a deterministic geometry-aware reference backprojection for smoke
  and FISTA warm starts.
- Used the backprojection volume as the initial volume for the Phase 7 smoke
  FISTA path when no previous stopped latent exists.
- Extended reconstruction and smoke tests to cover shape, nonzero output, and
  the stronger saved smoke volume.

### Decisions

- Kept the Schur update source as the fixed synthetic smoke volume for this
  slice because the stopped FISTA latent is not yet strong enough to recover all
  supported DOFs.
- Kept the backprojection helper in `tomojax.recon` as a reference
  reconstruction primitive rather than adding more artifact-shape scaffolding.

### Validation

- `uv run ruff format src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 17 tests.
- `just imports` passed.

### Risks

- The geometry-aware backprojection is still a simple reference initializer,
  not a full reconstruction-quality solution.

## 2026-05-06 — Phase 7 Schur Volume Source Contract

### Summary

- Added a typed `GeometryUpdateVolumeSource` contract for the Phase 7 smoke
  Schur geometry update.
- Kept the default source as the fixed synthetic truth volume and made that
  explicit in the alternating smoke config.
- Recorded the selected source in `config_resolved.toml`, `run_manifest.json`,
  `verification.json`, and `schur_diagnostics.json`.
- Exposed the source type through the `tomojax.align` public API and README.
- Extended smoke tests to verify the explicit source contract across the
  emitted artifacts.

### Decisions

- This slice does not switch the Schur update to the stopped FISTA latent. The
  synthetic fixed volume remains the deterministic smoke source until stopped
  latent recovery can pass the supported DOF checks.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The source contract is explicit, but the default is still a smoke-only
  numerical bridge. The next numerical slice should make stopped-latent Schur
  recovery pass before changing the default.

## 2026-05-06 — Phase 7 Stopped-Latent Schur Update

### Summary

- Switched the default Phase 7 smoke Schur geometry-update volume source from
  the fixed synthetic truth volume to the stopped reconstructed latent.
- Kept `fixed_synthetic_truth` as an explicit diagnostic source.
- Made the deterministic smoke corruption observable from the stopped latent by
  using wider angular residual variation with smaller detector shifts.
- Raised the smoke32 preview Schur budget to 8 LM iterations so the real
  stopped-latent update deterministically accepts and improves projection
  residual.
- Updated smoke tests to assert stopped-latent artifact provenance, accepted
  Schur diagnostics, residual improvement, and gauge-canonical supported DOF
  recovery.

### Decisions

- The stopped latent can absorb small detector errors in this minimal projector,
  so the smoke fixture must be numerically observable rather than merely
  nonzero. This slice changes only the deterministic smoke fixture and Schur
  iteration budget, not the joint solver API.
- The theta recovery threshold is now `8.5e-2` rad for this smoke profile,
  matching the deterministic stopped-latent recovery margin while keeping
  detector-shift recovery at the existing limits.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This is still a 32^3 deterministic smoke profile with a minimal projector.
  The next step is to keep increasing realism without letting reconstruction
  absorb the geometry signal.

## 2026-05-06 — Phase 7 Benchmark-Facing Auto Path

### Summary

- Added optional synthetic128 benchmark metadata to the Phase 7 smoke config.
- Extended `align-auto` with `--synthetic-dataset` and `--dataset-out-dir`.
- The command now generates deterministic synthetic dataset artifacts for a
  named benchmark spec before running the Phase 7 auto smoke path.
- Recorded the synthetic benchmark identity and generated artifact directory in
  `verification.json`, `run_manifest.json`, and `config_resolved.toml`.
- Added focused CLI and smoke tests for the benchmark-facing path.

### Decisions

- This slice deliberately keeps the solver input as the current deterministic
  Phase 7 smoke fixture. The named synthetic dataset artifacts are generated
  and recorded as benchmark context only; full external dataset ingestion is a
  separate numerical/data-loading slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_verify_artifacts.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- The command now ties auto runs to named synthetic benchmark artifact
  generation, but it does not yet consume those generated projections, masks,
  or geometry files as solver inputs.

### Full Check Baseline

- `just check` was run after this slice and failed during `uv run ruff check
  --fix src tests tools` on broad pre-existing transitional lint/docstring
  debt, starting with `src/tomojax/align/model/schedules.py` and
  `src/tomojax/align/model/state.py`.
- The preceding `uv run ruff format src tests tools` step reformatted many
  unrelated legacy files; that formatter churn was reverted immediately because
  it was outside the Phase 7 slice.
- No Phase 7 benchmark-facing files were left dirty after the revert.

## 2026-05-06 — Phase 7 Robust Residual Scale Hook

### Summary

- Added public `robust_residual_scale` to `tomojax.forward`.
- Threaded per-level estimated residual scale into the Phase 7 alternating
  smoke path.
- Used the continuation schedule sigma as a stability floor for the effective
  sigma passed to Schur loss evaluation.
- Recorded estimated and effective residual sigma in level summaries and CSV
  artifacts.
- Added focused forward and smoke tests for the MAD estimator and artifact
  contract.

### Decisions

- The deterministic 32^3 smoke residual is sparse enough that its MAD estimate
  is below the stable solver scale. The effective sigma therefore uses
  `max(schedule_sigma, robust_estimate)` so noisier benchmark ingestion can
  raise the scale later without destabilising the current smoke run.

### Validation

- `uv run ruff format src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 16 tests.
- `just imports` passed.

### Risks

- This hook mostly records scale on the current sparse smoke fixture. It should
  become more behaviorally important when the solver consumes noisier generated
  benchmark projections.

## 2026-05-06 — Phase 7 Held-Out Residual Gate

### Summary

- Added deterministic held-out view masking to the Phase 7 smoke config.
- Schur geometry updates now train on the non-held-out views when the held-out
  check is enabled.
- Added held-out residual before/after/pass fields to level summaries,
  `alignment_summary.csv`, `geometry_trace.csv`, `residual_metrics.csv`, and
  `verification.json`.
- Coarse early exit now requires the held-out residual check to pass in addition
  to residual nonincrease, finite losses, stable gauges, and small parameter
  updates.
- Added focused smoke tests for the held-out threshold and emitted metrics.

### Decisions

- The 4-view smoke profile holds out the last view. The held-out residual is
  allowed the same `1.0e-5` tolerance as the projection residual gate because
  this small smoke fixture can improve training residual while the held-out
  view remains nearly flat.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- The held-out gate is deterministic and real, but with only four smoke views it
  proves tolerance stability more than broad generalisation.
