# TomoJAX v2 Production Hardening Goal

Date: 2026-05-12

## Objective

Turn the current TomoJAX v2 branch from a successful research/prototype state
into a production-shaped package that Tristan can review as a publishable v2:

- clean public CLI/API,
- no public `mvp`, `v1`, `parity`, `audit`, `cor_mvp`, or debugging-era naming,
- legacy/v1/debug code deleted, quarantined, or renamed as internal reference
  regression material,
- real laminography workflow documented and runnable from clean commands,
- synthetic tomography workflows documented and runnable from clean commands,
- original `128^3` synthetic scenarios actually exercised and honestly
  reported,
- tests and validation that protect the public contracts,
- coherent commits.

This is not a short report task. Work for hours if needed. Prefer deleting,
renaming, and simplifying over adding another layer of wrappers. The repo should
feel like TomoJAX v2, not a pile of historical debugging sessions.

## Current Starting Point

Recent completed work established:

- A k11 real-laminography v2 run that reproduces the useful original staged
  workflow and improves over COR-only.
- A cleaned real-lamino profile split in progress.
- Report artifacts and contact sheets for the real run.
- Bounded 32^3 synthetic tomography smoke reports for:
  - `synth128_setup_global_tomo`
  - `synth128_pose_random_extreme`
- Those 32^3 synthetic smoke gates currently fail as quality gates:
  - setup-global proves det_u/theta wiring but fails axis/roll and has poor
    reconstruction quality,
  - pose-random emits artifacts but pose recovery/reporting is incomplete.

That is useful, but it is not enough. This goal is to make the repository and
public surface production-shaped and to run the original `128^3` scenarios
instead of stopping at tiny smoke tests.

## Non-Negotiable Product Direction

Public users should not see development-history names.

Do not leave these as public CLI/API/profile names:

- `mvp`
- `v1`
- `parity`
- `audit`
- `cor_mvp`
- `full_mvp`
- `smoke` as a success profile
- `after_fista_fallback`
- debugging one-off names

Acceptable public naming should describe behavior, not history. Prefer names
like:

- `staged-lamino`
- `staged-tomo-setup`
- `staged-tomo-pose`
- `synthetic-setup-global`
- `synthetic-pose-random`
- `diagnostic-fast`
- `reference-regression` for internal regression-only paths

It is acceptable for docs to say that a production profile was validated against
the original TomoJAX reference run. It is not acceptable for the user-facing CLI
to expose `--v1-parity-real-lamino` as the normal way to run TomoJAX v2.

## Read First

Read these before changing code:

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/03_repo_layout.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`
- `docs/tomojax-v2/benchmark_manifest.yaml`
- `.agent/PLANS.md`
- `docs/implementation_log.md`
- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`
- `docs/benchmark_runs/2026-05-12-synthetic-tomo-mvp.md`
- `src/tomojax/cli/align_auto.py`
- `src/tomojax/cli/align.py`
- `src/tomojax/data/phantoms.py`
- `src/tomojax/data/simulate.py`
- `src/tomojax/data/contrast.py`
- `src/tomojax/data/artefacts.py`
- `scripts/real_laminography/run_real_lamino_v2_cor_mvp.py`
- `scripts/real_laminography/run_real_lamino_native_setup_pose_256.py`
- `tests/test_align_auto_cli.py`
- `tests/test_real_lamino_runner_contract.py`

Also inspect the run artifacts from:

- `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`
- `.artifacts/production_synthetic_tomo_mvp/runs/synth128_setup_global_tomo_32`
- `.artifacts/production_synthetic_tomo_mvp/runs/synth128_pose_random_extreme_32`

## Required Outcome

By the end, leave the repo with:

1. A clean public command/profile for the real laminography staged workflow.
2. Clean public commands/profiles for synthetic tomography setup-global and
   pose-random workflows.
3. No public product surface that depends on `mvp`, `v1`, `parity`, `audit`, or
   debugging-era names.
4. Internal reference-regression paths preserved only where useful and named as
   such.
5. Dead v1/debug code either deleted, quarantined behind internal fixtures, or
   documented as intentionally retained.
6. Original `128^3` synthetic scenarios run and reported, not merely 32^3
   smoke tests.
7. A morning-facing production readiness report that clearly says:
   - what is production-shaped now,
   - what works on real data,
   - which `128^3` synthetic scenarios pass/fail,
   - what remains research work,
   - what commands users should run.
8. Focused tests for CLI/profile names, synthetic scenario presets, report
   contracts, and absence of public legacy terminology.
9. Validation results recorded.
10. Coherent commits.

## Scope Boundary

This goal is production hardening through the core of the original plan:

- Phase 0: benchmark/report contract,
- Phase 1: geometry state and gauges,
- Phase 2: forward/residual paths,
- Phase 3: FISTA/reconstruction,
- Phase 4: pose updates,
- Phase 5: setup updates,
- Phase 6: Schur/LM/GN machinery where already present,
- Phase 7: staged/alternating workflow as the current production path.

Do not spend the night trying to complete research phases:

- nuisance as a default production path,
- object drift/deformation,
- full Pallas end-to-end default backend,
- experimental neural/RED/PnP/object-motion modules.

Those may appear in reports as future work or diagnostic-only surfaces, but they
should not be allowed to sprawl into the public product surface.

## Required 128^3 Synthetic Scenario Runs

The previous 32^3 synthetic report is not enough. The original v2 plan was a
`128^3` synthetic suite. Run actual `128^3` scenarios and report them honestly.

Required `128^3` runs:

1. `synth128_setup_global_tomo`
2. `synth128_pose_random_extreme`

These two are mandatory and should be treated as production-blocker evidence for
tomography.

If either mandatory `128^3` tomography gate fails, do not stop at reporting the
failure. Debug and fix it. Reporting a failure is only acceptable after you have:

- identified the smallest reproducible failing subcase,
- determined whether the cause is data generation, sidecar/truth wiring,
  metric/evaluation logic, active-DOF selection, gauge canonicalisation, solver
  update logic, reconstruction handoff, numerical conditioning, memory/runtime,
  or an actually unsupported model feature,
- implemented any low-risk fix that follows from that diagnosis,
- rerun the smallest useful gate to prove the fix,
- rerun the `128^3` case or clearly record the remaining blocker to doing so.

The goal is to make these tomography cases work, not merely to produce an
honest red report. A final red status is acceptable only if it is accompanied by
a precise diagnosis, committed tests/artifacts for the failing path, and a
concrete next implementation step.

Also exercise the remaining original cases at `128^3` if the code supports
generating/running them without inventing large new subsystems:

3. `synth128_lamino_axis_roll_pose`
4. `synth128_thermal_object_drift`
5. `synth128_combined_nuisance_jumps`

If cases 3-5 are not implemented enough to run honestly, do not fake them.
Instead, create explicit `not_implemented` or `diagnostic_only` report rows with
the concrete missing code paths. But cases 1 and 2 must be attempted at `128^3`.

Run policy:

- Use CUDA for `128^3` runs where available.
- Do not exceed `128^3` volume shape.
- Use the documented detector/view counts where feasible.
- If 256 views is too slow, first run a bounded `128^3` lower-view gate and
  record the exact blocker to running the full manifest view count.
- Do not silently replace `128^3` with `32^3` or `64^3`.
- Do not use weak-view exclusions, unsupported-DOF omissions, or metric
  laundering to mark a case green.
- Every run must emit artifacts and a machine-readable result.

Reports required:

- `docs/benchmark_runs/2026-05-13-synthetic128-production-gates.md`

The report must include:

- command for each case,
- device/backend,
- volume/detector/view shape,
- runtime and peak GPU memory if available,
- pass/fail/partial/not-implemented status,
- reconstruction metrics,
- geometry/pose recovery metrics where truth exists,
- artifact paths,
- direct explanation of failures,
- next fix for each failed case.

For cases 1 and 2, include explicit answer:

- Did `synth128_setup_global_tomo` recover setup/COR/roll/axis/theta at 128^3?
- Did `synth128_pose_random_extreme` recover per-view dx/dz/phi/alpha/beta at
  128^3?

## Production Naming Cleanup

Audit public code and docs for these strings:

```text
mvp
v1
parity
audit
cor_mvp
full_mvp
after_fista_fallback
smoke
```

Do not blindly delete every occurrence. Classify each occurrence:

- public product surface: rename,
- internal regression fixture: keep but rename to `reference-regression` style
  if practical,
- historical report: leave if it describes past evidence,
- dead code: delete,
- tests: update to match production naming while preserving internal regression
  coverage.

At minimum, public CLI help, README/quickstart docs, and public profile names
should not expose the historical debugging names.

Expected public commands should look like production commands. For example:

```bash
tomojax align <scan.nxs> --profile staged-lamino --out <run-dir>
tomojax align-auto <scan-or-sidecar> --profile staged-tomo-setup --out <run-dir>
tomojax align-auto <scan-or-sidecar> --profile staged-tomo-pose --out <run-dir>
tomojax align-auto --synthetic-case setup-global --size 128 --out <run-dir>
tomojax align-auto --synthetic-case pose-random --size 128 --out <run-dir>
```

Adapt to the actual CLI style in the repo, but keep the surface clean.

## Dead Code And Legacy Audit

Audit:

- `scripts/`
- `src/tomojax/cli/`
- `src/tomojax/align/`
- `src/tomojax/bench/`
- old runner scripts and one-off diagnostics,
- tests that preserve old behavior only accidentally.

For each suspicious file/path, decide:

1. production code,
2. internal regression fixture,
3. benchmark/diagnostic tool,
4. dead code.

Then act:

- delete dead code,
- rename/quarantine reference-only paths,
- document diagnostic-only tools,
- leave production paths clean.

Do not delete useful evidence artifacts. Git history is the archive for dead
code; docs are the archive for decisions.

## Public API And CLI Hardening

Make the public surface feel intentional:

- `tomojax` package imports should expose deliberate APIs through `api.py` and
  `__init__.py`, not private internals.
- CLI help should guide a user to the real staged laminography workflow and the
  synthetic tomography gates.
- Profile resolution should be typed/tested and not scattered through giant
  scripts.
- Reports should be generated by reusable report helpers where practical.
- Artifact paths should be portable and documented.

Add or update docs:

- `README.md` or the nearest repo-level user entrypoint,
- `docs/quickstart.md` if appropriate,
- `docs/real-laminography.md` or an equivalent real-data guide,
- `docs/synthetic-tomography.md` or an equivalent synthetic guide,
- `docs/architecture-status.md` or equivalent phase/status report,
- `docs/known-limitations.md` if the repo lacks a clear limitations page.

Docs should answer:

- how to run the real staged laminography workflow,
- how to run synthetic setup-global tomography,
- how to run synthetic pose-random tomography,
- what artifacts are produced,
- what current limitations are,
- how v2 relates to the original TomoJAX without making original-v1 terms the
  public product language.

## Real Laminography Production Workflow

Preserve the working k11 evidence, but make the public entrypoint clean.

Requirements:

- A clean profile/command for the staged laminography workflow.
- Internal reference-regression path can compare to the original TomoJAX run,
  but should not be the default user command.
- The real run report should link publication PNGs and contact sheets.
- The report should state this is a validated real-data workflow, not universal
  arbitrary-scan production proof.

Do not rerun the 6-hour full k11 workflow unless needed. The completed evidence
can be used for docs if the production command/profile is tested by focused
contract tests. If you do rerun, record why.

## Tomography Fix Workstream

Do not stop at "the 128^3 scenarios failed" if there is a clear fix available.
Spend meaningful time diagnosing and improving the two mandatory tomography
cases.

For `synth128_setup_global_tomo`:

- Check whether axis/roll failures are real solver failures, insufficient views,
  incorrect criteria conversion, unsupported active DOFs, or report/evaluation
  bugs.
- Verify det_u/theta/roll/axis truth values enter sidecars correctly.
- Verify criteria units and gauge canonicalisation.
- If a low-risk fix is found, implement it and rerun the smallest useful gate,
  then rerun 128^3 if feasible.
- Do not leave this as "axis/roll failed" without inspecting sidecar truth,
  active setup names, geometry traces, criteria units, and the actual Schur
  accepted step history.

For `synth128_pose_random_extreme`:

- Make dx/dz/phi metrics evaluable when those DOFs are active.
- Do not mark unsupported alpha/beta as success.
- Verify pose truth sidecars and pose decomposition output.
- If the solver is not recovering pose, identify whether the failure is
  evaluation, active-DOF selection, trust radii, loss scaling, or real solver
  weakness.
- Implement low-risk fixes and rerun.
- Do not leave this as "pose metrics not evaluated" without fixing the metrics
  path or proving the DOFs are genuinely unsupported in the current production
  profile.

Do not add grid search, sinogram correlation, sharpness/entropy sweeps,
benchmark-specific fake passes, or truth leakage.

## Testing Requirements

Add or update tests for:

- public profile names and aliases,
- old names not appearing in public CLI help,
- internal reference regression path remains available,
- synthetic case presets for setup-global and pose-random,
- `128^3` command construction without actually running expensive tests,
- report/artifact contract for synthetic and real runs,
- fail-closed candidate selection,
- phi-level loss-scale guard,
- no public imports from private deep modules,
- deleted/quarantined legacy paths not referenced from docs.

Validation commands to run at minimum:

```bash
uv run pytest tests/test_align_auto_cli.py tests/test_real_lamino_runner_contract.py -q
uv run ruff check <changed files>
uv run basedpyright <changed files>
just imports
```

Run broader `just check` if feasible. If it is too slow or fails for unrelated
known reasons, record the exact command and failure.

## Commit Strategy

Make coherent commits. Suggested slices:

1. public naming/profile cleanup,
2. dead-code/legacy quarantine cleanup,
3. synthetic 128^3 scenario runner/report work,
4. tomography metric/fix work,
5. docs/readme/quickstart,
6. validation/report finalization.

Do not leave important productionization changes uncommitted. Do not commit the
pre-existing oracle/support note files unless they are intentionally part of the
new documentation set.

## Final Morning Report

Create/update:

- `docs/benchmark_runs/2026-05-13-production-hardening.md`
- `docs/benchmark_runs/2026-05-13-synthetic128-production-gates.md`

The final report must answer:

1. What is the clean public API/CLI now?
2. What old public names were removed or renamed?
3. What dead/legacy code was deleted or quarantined?
4. How does a user run the real staged laminography workflow?
5. How does a user run synthetic setup-global and pose-random tomography?
6. Which original `128^3` scenarios were actually run?
7. Which passed, failed, partial, or not implemented?
8. What exact artifacts should Tristan inspect?
9. What tests and checks passed?
10. What remains before a real public release?

Be direct. If tomography still fails, say exactly that and why. The goal is a
production-shaped, honest repo, not green JSON through benchmark laundering.
