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
