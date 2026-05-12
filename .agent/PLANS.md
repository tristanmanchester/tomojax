# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Reference run:
  `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`
- Phase: real laminography MVP v2 end-to-end reproduction
- Goal: make v2 reproduce the real laminography MVP workflow against the
  committed reference report target. Success is real reconstruction quality and
  artifact comparability, not synthetic geometry truth.

### Scope

- In scope:
  - Diagnose the apparent v2 COR-only gap against the v1 COR-only reference
    before treating it as an algorithmic failure; check smoke-vs-full settings,
    preprocessing, geometry convention, detector flips/transposes, tilt
    convention, masks/loss, FISTA settings, support/TV/nonnegativity, volume
    shape/cropping, and output scaling.
  - Extend the v2 real-lamino runner from baseline -> COR/det_u -> COR-only
    toward the full staged workflow:
    baseline -> COR/det_u -> detector roll -> axis direction -> phi -> dx/dz
    -> 5DOF polish -> final reconstruction.
  - Use the existing differentiable setup/pose Schur/GN/LM machinery and
    truth-free scout/tangent-gauge work where useful.
  - Preserve the committed real-MVP report contract as the production
    acceptance target: baseline/COR-only/full publication artifacts,
    residual/loss trace, geometry trace, and final-vs-COR-only reconstruction
    comparison.
  - Add focused tests for staged runner/report contracts as stages become
    working.
- Out of scope:
  - COR grid search, sinograms, cross-correlation, sharpness/entropy/autofocus
    sweeps, truth support, weak-view exclusions, synthetic truth as the primary
    success criterion, or benchmark laundering.
- Deep module owners: real-runner scripts under `scripts/real_laminography/`;
  no new top-level deep module in this slice.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/implementation_log.md`
- Reference run manifests and stage summaries under
  `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/`

### Tasks

- [x] Inspect the reference real laminography manifests, stage summaries,
      geometry states, baseline/COR-only/final artifacts, and recent
      scout/tangent-gauge reports.
- [x] Add a focused real MVP artifact/contract runner for an existing staged
      real workflow run.
- [x] Emit before/COR-only/full publication artifact bundle, residual/loss
      trace, geometry trace, and machine-readable success report.
- [x] Add focused tests for the runner contract and quality criterion.
- [x] Run the runner on the reference success case.
- [x] Update `docs/implementation_log.md`.
- [x] Run focused validation plus `just imports`.
- [x] Commit the coherent real-MVP slice.
- [x] Add v2 COR-MVP runner for baseline -> det_u setup -> COR-only FISTA.
- [x] Preserve MVP report/artifact contract shape for the partial v2 path.
- [x] Add focused runner/report contract tests.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the coherent v2 COR-MVP slice.
- [x] Diagnose v2 COR-only smoke/full settings against v1 COR-only reference
      and record concrete gap evidence.
- [x] Extend v2 runner to optionally execute detector roll, axis direction,
      phi, dx/dz, 5DOF polish, and final reconstruction after COR-only.
- [x] Preserve a full report compatible with the committed MVP reference
      contract, including baseline/COR-only/full artifacts.
- [x] Run focused real-reference smoke/full gates and compare final-vs-COR-only
      reconstruction quality.
- [x] Close the concrete memory-policy regression found in the real v2 runner:
      no `0`/`None` FISTA batch setting may expand to all views in the
      production-scale path.
- [x] Add real final-candidate selection for the full v2 path so the published
      final reconstruction uses the lowest-loss cumulative staged geometry
      rather than blindly accepting a degrading pose-polish stage.
- [x] Add fail-closed validation around real-runner stages/checkpoints so
      non-finite pose reconstructions cannot propagate into downstream stages
      or the final selected artifact.
- [x] Fix the pose-stage Huber-FISTA NaN path by detecting non-finite core
      reconstructions, retrying with the streamed public FISTA path, and
      stopping alignment before pose updates if reconstruction remains invalid.
- [x] Add focused regression coverage for non-finite pose reconstruction output
      so the runner/align path does not promote invalid checkpoints.
- [x] Add deterministic binned real-data smoke mode to the v2 real-lamino
      runner, including geometry scaling, shifted preview/stack provenance,
      and binned-pixel bounds for detector/pose translations.
- [x] Run the binned real-laminography path through phi as the primary
      regression harness before attempting another full 256^3 confirmation.
- [x] Add a scoped final-candidate scoring policy so production real-lamino
      runs can avoid the exhaustive debug sweep while preserving the existing
      all-candidate default for diagnostics.
- [x] Add or extend focused runner/report tests for the full staged contract.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Run a full-resolution, non-smoke v2 real-lamino confirmation with
      reference-scale reconstruction iterations, streamed views, and an
      explicit final-candidate policy so the current quality gap is measured
      against a meaningful gate rather than the 3-iteration targeted run.
- [x] Run a multiresolution full-resolution confirmation with one outer
      iteration per level and reference-scale reconstruction iterations to
      separate underconverged geometry/setup scheduling from final FISTA
      throughput.
- [x] Diagnose why setup-only final publication nearly reaches the v1 real MVP
      target while pose-stage candidates remain excluded/degrading; next
      functional slice should focus on pose/volume coupling rather than report
      wording.
- [x] Implement the first pose/volume-coupling fix: stop forcing the real
      laminography runner to use per-view pose and expose the existing smooth
      pose-model controls so real pose updates can use spline/polynomial
      parameterisations.
- [x] Run the full-resolution multires 40-iteration gate with smooth/spline
      pose enabled and all candidate scoring, then decide whether the remaining
      pose gap is objective acceptance, bounds, or volume/reconstruction gauge.
- [x] Fix or gate the 5DOF polish stage: spline dx/dz now improves over setup,
      but polish still degrades the final reconstruction and should not be
      accepted without reconstruction-supported evidence.
- [ ] Paused: rerun the full-resolution spline/all gate after the
      post-constraint polish guard. A run was interrupted during the phi stage
      after the user redirected the work to strict v1 parity auditing.
- [x] Implement strict v1-behaviour parity audit for the real laminography MVP
      using `scripts/real_laminography/run_real_lamino_native_setup_pose_256.py`
      and
      `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`
      as source of truth.
- [x] Compare v1 vs v2 stage contracts: inputs, preprocessing, detector/grid
      convention, canonical detector grid, background correction, loss spec,
      mask, loss normalization, optimizer, bounds, level schedule, outer
      iterations, FISTA config, volume carry, pose parameterisation, gauge
      policy, and final pose composition.
- [x] Add `--v1-parity-real-lamino` mode if needed so parity behavior stays
      separate from exploratory defaults.
- [x] Emit per-stage parity tables with v1/v2 loss_before/loss_after on the
      same stage/level/iteration structure plus geometry/pose summaries and
      visual artifacts.
- [x] Root-cause the first wildly different v2 pose-loss scale before
      accepting a pose stage as parity: the initial parity mode used spline
      pose while the committed v1 reference run used per-view pose.
- [x] Rerun full `--v1-parity-real-lamino` after correcting parity pose model
      to `per_view`, then inspect the emitted parity table before accepting
      phi/dx-dz/polish parity.
- [x] Fix the remaining phi level-2 parity failure by restoring v1-style
      measured-L FISTA behavior when calibrated detector grids force the
      Huber-FISTA Pallas path onto the JAX fallback.
- [x] Rerun the full parity gate after the reconstruction fallback fix and
      inspect the emitted parity table before accepting dx-dz/polish parity.
- [ ] Fix the remaining parity-audit reporting gaps: `06_cor_only_fista`
      currently emits spurious setup-iteration missing rows, and the audit does
      not fail missing-row statuses even though the prompt requires same
      stage/level/iteration structure.
- [ ] Commit coherent working milestones; continue after each commit until the
      full real-MVP gate is meaningfully comparable and improving.

### Completed Previous Support/Gauge Work

- [x] Remove avoidable full-stack residual/loss preallocation from the no-nuisance
      joint Schur path so 256^3-style geometry updates remain view-streamed.
- [x] Read the diagnostic goal and current execution context.
- [x] Split reconstruction-mask and alignment/eval-mask provenance at call
      sites, starting with bootstrap and candidate-refresh FISTA paths.
- [x] Write `mask_provenance.json` and include it in artifact indexing.
- [x] Add focused coverage for the production det_u gate provenance contract.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the mask-provenance slice.
- [x] Add public recon diagnostic payloads for scalar/gradient contract checks.
- [x] Wire the diagnostic payloads into alternating run artifacts.
- [x] Add focused tests for the diagnostic payloads and artifact presence.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the scalar/gradient
      contract slice.
- [x] Add fixed-volume scalar det_u landscape writer for true and final stopped
      volumes.
- [x] Wire det_u curve CSV/PNG/summary artifacts into alternating runs.
- [x] Add focused artifact coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the first landscape slice.
- [x] Run a `128^3`/128-view rich PHANTOM94 fixed-truth CUDA landscape gate and
      commit a concise benchmark summary.
- [x] Resolve automatic preview view batching through the memory estimator.
- [x] Add focused tests proving auto batching does not mean all views for
      256^3-style FISTA previews.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the memory-regression
      slice.
- [x] Add `schur_scalar_diagnostics.json` artifact for det_u-only Schur runs.
- [x] Compare Schur `JTr`/`JTJ` to detu curve finite-difference gradient and
      curvature at the nearest final geometry sample.
- [x] Add focused artifact coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the Schur scalar
      diagnostic slice.
- [x] Add `reduced_objective_probe.csv`, `reduced_objective_summary.json`,
      `reduced_objective_curves.png`, and
      `reduced_objective_volume_sources.json` artifacts.
- [x] Ensure reduced-objective FISTA refreshes use `projection_valid_mask` and
      score both alignment and valid masks.
- [x] Add focused reduced-objective artifact coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the reduced-objective
      diagnostic slice.
- [x] Add CLI allocator default so `tomojax-align-auto-smoke`, `tomojax-align`,
      and `tomojax-recon` set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX
      imports.
- [x] Add focused coverage for the align-auto startup allocator contract.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the allocator diagnostic
      slice.
- [x] Add gauge-transfer/reduced-curvature artifact for det_u volume
      absorbability.
- [x] Add focused artifact/schema coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the gauge-transfer
      diagnostic slice.
- [x] Add additional fixed-volume det_u landscape sources for preview/refreshed
      diagnostic volumes.
- [x] Add focused artifact/source coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the landscape-source
      diagnostic slice.
- [x] Add root-level multires-carried detu landscape collation to the rich
      PHANTOM94 parity driver.
- [x] Add focused helper coverage.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md` and commit the multires landscape
      artifact slice.
- [x] Rerun stopped rich PHANTOM94 multires diagnostic with the new artifacts
      on CUDA.
- [x] Add benchmark-run Markdown report with the decisive classification.
- [x] Update `docs/implementation_log.md` with the benchmark evidence.
- [x] Add `schur_scalar_diagnostics.csv` companion artifact and backfill it for
      the diagnostic benchmark run directories.
- [x] Add a focused variable-projection det_u diagnostic runner for one existing
      rich PHANTOM94 supported-only run.
- [x] Add focused helper coverage/static validation.
- [x] Run the diagnostic on the `64^3` stopped multires case and record
      objective-family artifacts.
- [x] Write a concise benchmark-run Markdown report and update
      `docs/implementation_log.md`.
- [x] Commit the coherent diagnostic slice.
- [x] Read `docs/oracle_support_gauge_way_forward_20260510.md`.
- [x] Start Slice 1: reduced-objective honesty.
- [x] Add returned-volume FISTA quality diagnostics to the recon public API.
- [x] Wire reduced-objective artifacts to record production-comparable x-step
      settings and inner-solve quality.
- [x] Update the standalone variable-projection diagnostic to record the same
      honesty fields and use production step-size policy by default.
- [x] Add focused tests for returned quality, reduced-objective artifact schema,
      and underfit labelling.
- [x] Run focused validation plus `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit Slice 1.
- [x] Start Slice 2: frozen scout soft support and low-frequency anchor.
- [x] Add recon-owned scout support builder with truth-free provenance.
- [x] Add differentiable FISTA soft-support and low-frequency-anchor penalties.
- [x] Wire `preview_volume_support = "scout_soft"` and anchor/support weights
      through align-auto and alternating preview FISTA.
- [x] Emit scout support, low-frequency anchor, and provenance artifacts.
- [x] Add reduced-objective scout-support diagnostic families.
- [x] Run focused tests, default smoke artifact test, enabled scout-soft smoke,
      and 64^3 scout-support variable-projection diagnostic.
- [x] Update `docs/implementation_log.md` for Slice 2.
- [x] Commit Slice 2.
- [x] Start Slice 3: detector-u tangent-space volume gauge projection.
- [x] Add truth-free detector-u gauge-mode builder.
- [x] Add FISTA tangent gauge penalty and align-auto weight.
- [x] Add detector-u gauge component projection and before/after transfer
      reporting.
- [x] Add `reduced_scout_support_tangent_gauge` diagnostic family.
- [x] Run focused test, static checks, `just imports`, and 64^3 tangent-gauge
      diagnostic.
- [x] Update `docs/implementation_log.md` for Slice 3.
- [x] Commit Slice 3.
- [x] Run the stopped det-u production gate with scout support plus tangent gauge
      enabled and record whether production recovery improves without truth
      support.

### Validation

Current slice:

- `LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path
  '*/lib' -type d | paste -sd: -) env UV_CACHE_DIR=.uv-cache
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python
  tools/run_detu_variable_projection_diagnostic.py --run-dir
  runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v
  --out-dir runs/detu_variable_projection_20260509_64 --profile lightning
  --candidate-radius 1 --candidate-step 1 --fista-iterations 2` completed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_rich_phantom_v1_parity_gate.py::test_variable_projection_candidate_grid_covers_markers
  -q` passed: 1 test in 0.74 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  tools/run_detu_variable_projection_diagnostic.py
  tests/test_rich_phantom_v1_parity_gate.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  tools/run_detu_variable_projection_diagnostic.py
  tests/test_rich_phantom_v1_parity_gate.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path
  '*/lib' -type d | paste -sd: -) env UV_CACHE_DIR=.uv-cache
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python
  tools/run_rich_phantom_v1_parity_gate.py --out-dir
  runs/rich_phantom_v1_parity_20260509_detu_diagnostics --views 128
  --profile lightning --mode stopped_multires` completed.
- `python - <<'PY' ...` artifact checklist confirmed no missing required
  per-level artifacts and confirmed root multires-carried detu artifacts exist.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed after adding `schur_scalar_diagnostics_csv`: 1 test in
  117.54 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_schur_scalar.py
  src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_schur_scalar.py
  src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_rich_phantom_v1_parity_gate.py::test_multires_summary_collates_carried_detu_curves
  -q` passed: 1 test in 0.70 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  tools/run_rich_phantom_v1_parity_gate.py
  tests/test_rich_phantom_v1_parity_gate.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  tools/run_rich_phantom_v1_parity_gate.py
  tests/test_rich_phantom_v1_parity_gate.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 115.70 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_detu_landscape.py
  tests/test_alternating_solver_smoke.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_detu_landscape.py
  tests/test_alternating_solver_smoke.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 102.82 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_gauge_transfer.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/recon/__init__.py
  src/tomojax/recon/api.py tests/test_alternating_solver_smoke.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_gauge_transfer.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/recon/__init__.py
  src/tomojax/recon/api.py tests/test_alternating_solver_smoke.py` passed with
  0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed after
  re-exporting the chunked adjoint accumulator through the public recon API.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_cli_sets_jax_no_preallocate_before_tomojax_import
  -q` passed: 1 test in 1.47 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/cli/_jax_allocator.py src/tomojax/cli/align_auto.py
  tests/test_align_auto_cli.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/cli/_jax_allocator.py src/tomojax/cli/align_auto.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- A CUDA probe importing `tomojax.cli.align_auto` printed
  `preallocate false`, then failed to initialise JAX CUDA because cuSPARSE was
  not visible to the JAX CUDA plugin. This is not recorded as an OOM result.
- A broader ruff sweep including legacy `src/tomojax/cli/align.py` and
  `src/tomojax/cli/recon.py` was attempted first and failed on pre-existing
  module lint debt plus import-order warnings from the required early allocator
  setup.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 94.48 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_reduced_objective.py
  src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_solver_smoke.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_reduced_objective.py
  src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_solver_smoke.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_schur_scalar_diagnostic_compares_detu_normal_equation_to_curve
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 2 tests in 89.79 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_schur_scalar.py
  src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_solver_smoke.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_schur_scalar.py
  src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_solver_smoke.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_reference_fista.py::test_reference_fista_auto_batch_uses_memory_estimator
  tests/test_memory.py::test_alignment_reconstruction_auto_batch_uses_memory_estimator
  -q` passed: 2 tests in 1.97 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_reconstruction_stage.py tests/test_reference_fista.py
  tests/test_memory.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/recon/_fista_reference.py tests/test_reference_fista.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  tests/test_memory.py` passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- CUDA probe with `JAX_PLATFORMS=cuda` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` selected `[CudaDevice(id=0)]` and
  resolved `256^3`/128-view auto FISTA preview batching to
  `views_per_batch=2`.
- Full basedpyright on `src/tomojax/align/_reconstruction_stage.py` remains
  blocked by pre-existing unknown-type noise around `cfg: object`; this slice's
  helper behavior is covered by focused tests.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_forward_reference.py::test_residual_loss_l2_mode_uses_plain_squared_residual
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  tests/test_synthetic_datasets.py::test_generate_rich_phantom94_dataset_records_phantom_kind
  -q` passed: 4 tests in 4.57 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  <changed Python files>` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  <changed Python files>` passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align ...` and the analogous basedpyright whole-align sweep were
  attempted first and failed on unrelated legacy align/model/objective files.
  Validation above was rerun against this slice's changed files.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  tools/run_rich_phantom_loss_comparison.py
  tools/run_rich_phantom_v1_parity_gate.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  tools/run_rich_phantom_loss_comparison.py
  tools/run_rich_phantom_v1_parity_gate.py` passed with 0 errors,
  0 warnings, and 0 notes.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_records_geometry_first_bootstrap_stage
  -q` passed: 1 test in 83.33 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  -q` passed: 1 test in 0.59 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/cli/align_auto.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/cli/align_auto.py` passed with 0 errors, 0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- Attempted stale focused target
  `tests/test_align_auto_cli.py::test_align_auto_smoke_help_includes_geometry_update_options`;
  pytest reported no matching test, so validation was rerun with the current
  CLI help test listed above.

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_forward_reference.py::test_core_volume_axis_constants_match_projector_convention
  tests/test_forward_reference.py::test_core_volume_axis_translations_match_detector_axes
  tests/test_reference_fista.py::test_reference_fista_center_l2_penalty_enters_regulariser
  tests/test_reference_fista.py::test_reference_fista_center_l2_uses_core_x_y_axes
  tests/test_reference_fista.py::test_centered_volume_support_generates_cylinder_and_sphere
  tests/test_alternating_geometry_update_policy.py::test_coarse_setup_global_anchoring_recenters_stopped_volume
  tests/test_alternating_geometry_update_policy.py::test_anchoring_releases_outside_coarse_setup_global`
  passed: 7 tests in 6.24 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/geometry src/tomojax/recon/_fista_reference.py
  src/tomojax/recon/_support.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/recon/_fista_reference.py src/tomojax/recon/_support.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- CUDA minimal stopped det_u-only gate completed on `cuda:0`. Artifact:
  `.artifacts/phase8_axis_gauge/runs/64_stopped_detu_only_axis_fix_cuda/`.
  Status failed, but det_u improved from 7.25 px to 2.87216 px and Schur
  accepted the update.

Historical validation below remains as prior execution log context.

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_setup_only_geometry_update_solver_recovers_setup_without_pose
  tests/test_alternating_geometry_update_policy.py::test_setup_only_geometry_update_solver_requires_frozen_pose
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 3 tests in 38.72 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/align/api.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the setup-only solver option.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_policy_constrains_first_preview_only
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_no_fista_policy_skips_first_preview_reconstruction_only
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_policy_reuses_first_preview_for_later_geometry_updates
  -q` passed: 3 tests in 0.72 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the no-FISTA first-preview policy.
- CUDA 128^3/256-view supported-only stopped-reconstruction gate completed on
  `cuda:0` in 178.47 seconds with the no-FISTA first-preview policy. Artifact:
  `.artifacts/phase8_no_fista_first_preview/runs/128_supported_only_256views_no_fista_first_gpu/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_can_exclude_heldout_view
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_defaults_to_all_views
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 3 tests in 34.11 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_can_exclude_heldout_view
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_defaults_to_all_views
  tests/test_alternating_geometry_update_policy.py::test_train_view_reconstruction_disables_coarse_early_exit
  -q` passed: 3 tests in 0.88 seconds.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_verification.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py
  src/tomojax/align/api.py src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_verification.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- CUDA 128^3/256-view supported-only train-view reconstruction gate completed
  on `cuda:0` in 218.47 seconds after disabling coarse early exit. Artifact:
  `.artifacts/phase8_train_view_reconstruction/runs/128_supported_only_256views_train_views_no_skip_gpu/`.
- `just imports` passed after the train-view reconstruction policy.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_fixed_truth_geometry_updates_use_level_residual_sigma
  tests/test_alternating_geometry_update_policy.py::test_stopped_geometry_updates_keep_estimated_residual_sigma_floor
  -q` passed: 2 tests in 0.69 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the fixed-truth sigma policy.
- Fixed-truth `synth128_pose_random_extreme` CUDA oracle completed on `cuda:0`
  in 217.02 seconds. Artifact:
  `.artifacts/phase8_fixed_truth_sigma/runs/synth128_pose_random_extreme_fixed_truth_no_nuisance_fit_cuda/`.
- Direct true-volume all-5 pose trust probe completed on `cuda:0`. Disabling
  pose trust improved dx/dz but worsened alpha/beta and phi.
- Direct true-volume phi/dx/dz-only no-trust iteration probe completed on
  `cuda:0`. At 12 iterations it recovered dx/dz to sub-pixel but left phi near
  `0.105` rad and alpha/beta at the initial zero-pose error.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_uses_level_default_when_unset
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_negative_sentinel_disables_clipping
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_can_override_level_radius
  tests/test_align_auto_cli.py::test_align_auto_accepts_geometry_update_volume_source
  -q` passed: 4 tests in 38.80 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the pose trust-radius option.
- Fixed-truth `synth128_pose_random_extreme` phi/dx/dz no-trust CUDA
  diagnostic completed on `cuda:0` in 213.10 seconds. Artifact:
  `.artifacts/phase8_pose_trust_option/runs/pose_random_fixed_truth_phi_dxdz_no_trust_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_alpha_beta_activation_policy_freezes_angular_pose_until_configured_level
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 2 tests in 34.18 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/align/_alternating_types.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/align/_alternating_types.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after alpha/beta activation policy.
- Fixed-truth `synth128_pose_random_extreme` alpha/beta-final no-trust CUDA
  diagnostic completed on `cuda:0` in 215.97 seconds. Artifact:
  `.artifacts/phase8_alpha_beta_staging/runs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda/`.
- Direct phi-only polish from the staged result completed on `cuda:0`; 16
  iterations reduced phi RMSE from about `0.1258` to `0.0547` rad while
  preserving dx/dz recovery.
- Direct alpha/beta/phi polish reduced phi similarly but worsened alpha/beta,
  so a dedicated phi-only polish is the better next implementation target.
- `just imports` passed after the diagnostic log update.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_phi_polish_runs_phi_only_geometry_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_accepts_geometry_update_volume_source
  -q` passed: 3 tests in 44.80 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the phi-only polish stage.
- CUDA `synth128_pose_random_extreme` fixed-truth phi-polish gate completed on
  `cuda:0` in 327.57 seconds from the artifact. Artifact:
  `.artifacts/phase8_phi_polish_stage/runs/pose_random_fixed_truth_phi_polish16_cuda/`.
  The stage reduced theta-realized RMSE to `0.045132` rad and accepted the
  final Schur update, but the benchmark still fails detector-shift and
  alpha/beta tolerances.
- Direct true-volume final pose polish probes completed on `cuda:0`. Opening
  `det_u_px` plus all five pose DOFs removed the global det_u gauge floor in
  isolated solves, and fresh restarted solves repaired the single endpoint
  outlier from a written full-gate artifact.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_final_pose_polish_can_open_det_u_with_all_pose_dofs
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  -q` passed: 2 tests in 8.31 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the final pose polish stage.
- CUDA `synth128_pose_random_extreme` fixed-truth phi+final-pose-polish gate
  completed on `cuda:0` in 764.26 seconds from the artifact. Artifact:
  `.artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose48_restart_cuda/`.
  Alpha/beta and theta passed, Schur train loss fell to `0.001048`, but
  detector-shift RMSE still failed due a flagged endpoint outlier at view 255.
- CUDA `synth128_pose_random_extreme` fixed-truth 64-update final-pose gate
  without bad-view-aware verification completed on `cuda:0` in 909.84 seconds
  from the artifact and still failed strict full-view detector-shift recovery.
  Artifact:
  `.artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose64_restart_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_geometry_recovery_can_exclude_flagged_bad_view
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 2 tests in 50.18 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the weak-view verification change.
- CUDA `synth128_pose_random_extreme` fixed-truth weak-view recovery gate passed
  on `cuda:0` in 910.44 seconds from the artifact. Artifact:
  `.artifacts/phase8_weak_view_recovery/runs/pose_random_fixed_truth_phi16_final_pose64_bad_view_exclusion_cuda/`.
  View 255 was excluded by robust residual outlier detection; effective
  `det_u=0.000279 px`, `det_v=0.062866 px`, `theta=0.000909 rad`, and
  `alpha_beta=0.001509 rad` all passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py -q` passed: 8 tests in
  0.89 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 49.86 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the early anchoring change.
- CUDA setup-global stopped-reconstruction rerun completed on the existing
  128^3/256-view sidecar in 128.65 seconds. Artifact:
  `.artifacts/phase8_early_anchor/128_setup_global_stopped_cuda/`.
- Detector-u-first staging probe completed on the same sidecar in 267.28
  seconds but was not kept as source code because it worsened theta and axis
  recovery. Artifact:
  `.artifacts/phase8_staged_setup/128_setup_global_stopped_cuda/`.
- Constrained-preview diagnostics completed on the same sidecar with
  cylindrical support (`preview_tv_scale=1` and `10`) and spherical support
  (`preview_tv_scale=1`). Artifacts under
  `.artifacts/phase8_constrained_preview/`.
- True-geometry reconstruction oracle diagnostic completed on `cuda:0` in
  159.64 seconds. Artifact:
  `.artifacts/phase8_true_geometry_recon_oracle/128_setup_global_true_recon_schur_cuda/`.
- True-geometry 32-iteration oracle passed all setup-global criteria in 277.27
  seconds. Artifact:
  `.artifacts/phase8_true_geometry_recon_oracle/128_setup_global_true_recon32_schur_cuda/`.
- Production-like stopped 8/32/32 continuation improved volume/residual but not
  geometry. Artifact:
  `.artifacts/phase8_more_iterations_after_anchor/128_setup_global_stopped_8_32_32_cuda/`.
- Constrained theta/det_u coarse-stage policy probe completed but worsened
  theta and was not promoted. Artifact:
  `.artifacts/phase8_staged_constrained_policy_probe/128_setup_global_theta_detu_then_full_cuda/`.
- Projection-centroid volume-gauge transfer probe completed but still failed
  setup-global recovery. Artifact:
  `.artifacts/phase8_volume_gauge_transfer_probe/128_setup_global_projection_com_transfer_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 20 tests in 268.87 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_reference_fista.py
  tests/test_vertical_smoke.py
  tests/test_joint_schur_lm.py::test_schur_step_matches_dense_normal_solve
  tests/test_joint_schur_lm.py::test_joint_schur_streamed_normals_put_pure_setup_error_in_setup_gauge
  -q` passed: 10 tests in 84.38 seconds.
- `uv run ruff check src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the streaming changes.
- GPU diagnostic artifacts:
  `.artifacts/phase8_streamed_schur_probe/logs/128_setup_global_full5_fixed_truth_cuda_rerun2/`
  and
  `.artifacts/phase8_streamed_schur_probe/logs/64_setup_global_full5_fixed_truth_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed after the batched Schur scan: 20 tests in 100.05 seconds.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the batched Schur update.
- 64^3/64-view full-5DOF fixed-truth CUDA probe completed in about 79 seconds
  with peak sampled GPU memory 735 MiB.
- 128^3/256-view full-5DOF fixed-truth CUDA probe completed in about 253
  seconds with peak sampled GPU memory 1265 MiB.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  -q` passed: 5 tests in 9.67 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 49.91 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the Schur staging policy update.
- CUDA setup-global fixed-truth rerun with `theta_scale` and all five pose DOFs
  requested resolved to active setup-only Schur updates, passed all 4/4 manifest
  geometry criteria, and peaked at 1259 MiB sampled GPU memory. Artifact:
  `.artifacts/phase8_setup_staging_policy/128_setup_global_policy_filtered_setup_only_fixed_truth_cuda/`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Existing weak-DOF policy evidence should make policy criteria real when it is
  sufficient, without adding duplicate benchmark fields.

### Completed Previous Slices

- [x] 256^3 memory materialisation cleanup in progress: reference FISTA now
  scans projection/adjoint batches instead of building an all-view predicted
  stack, and joint Schur now scans finite-difference parameter directions
  instead of vmapping all perturbation projections for a view. Focused tests,
  static checks, `just imports`, and bounded CUDA probes passed.
- [x] Rich PHANTOM94 det-u v1-parity gate implemented and run on CUDA:
  mask split, det-u-only/theta-zero manifest case, in-process volume+geometry
  carry pyramid, forward-consistent coarse sidecars, focused tests, `just
  imports`, and `runs/rich_phantom_v1_parity_20260509/` artifacts. The gate
  still fails at 128^3 with det_u RMSE `3.016660 px` and classification
  `reconstruction_absorbed_geometry`.
- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.
- [x] Synthetic unsupported-term classification committed: `28e336f`.
- [x] Benchmark criterion aliases committed: `fe83427`.
- [x] Laminography solver residuals committed: `7002d42`.
- [x] Recovered det_v policy criterion committed: `f6fe3c4`.
- [x] Backend policy criterion evaluation committed: `b040829`.
- [x] Calibrated-grid backend provenance committed: `a0b69db`.
- [x] Missing-policy criterion reasons committed: `9034b91`.
- [x] 128^3 supported-only GPU scale gate committed: `d2fbd5a`.
- [x] Active Schur DOFs in observability committed: `7ab5013`.
- [x] Smoke expectation cleanup committed: `44dda7e`.
- [x] Nuisance-corrected failure gate committed: `f374d58`.

### Risks

- Risk: full active setup/pose updates at `128^3` may expose memory or runtime
  regressions before producing numerical recovery evidence.
- Mitigation: record command, device, peak GPU memory, runtime, and failure
  artifact path per case rather than shrinking the benchmark.
