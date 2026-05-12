# TomoJAX v2 Overnight Productionization Goal

Date: 2026-05-12

## Objective

Turn the current successful real-laminography v1-parity breakthrough into a
clean, honest, production-ready TomoJAX v2 MVP slice that Tristan can inspect in
the morning and plausibly use as the basis for an article/demo.

The goal is not another small incremental probe. Work end-to-end for as long as
needed. Consolidate the working path, clean up the obvious code sprawl, run
meaningful validation, produce comparison artifacts, and leave a clear morning
report.

The current strongest evidence is:

- Reference v1-style real run:
  `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`
- Winning v2 parity run:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`
- Winning v2 result:
  - final selected candidate: `04_pose_polish`
  - final loss: `6378.63330078125`
  - COR-only loss: `6740.05126953125`
  - improvement over COR-only: about `5.36%`
  - v1 reference final loss: about `6438.1611328125`
- Winning v2 setup estimates:
  - `det_u_px ~= -3.7259`
  - `detector_roll_deg ~= 0.1374`
  - `axis_rot_x_deg ~= 0.4603`
  - `axis_rot_y_deg ~= 0.0077`
- Winning v2 pose summary:
  - `dx std ~= 2.404 px`
  - `dz std ~= 1.578 px`
  - `phi mean ~= -0.191 deg`
  - `alpha std ~= 0.280 deg`
  - `beta std ~= 0.319 deg`

That is the first real-data result that looks like v1 parity. Preserve it, make
it reproducible, and make the code/story clean enough to stand behind.

## Non-Negotiable Framing

Do not overclaim.

The productionized MVP should say:

- TomoJAX v2 now reproduces the successful v1 real laminography staged
  alignment workflow on the k11 reference scan.
- The pipeline improves final reconstruction quality over COR-only on real
  data, with visual artifacts and machine-readable provenance.
- The stage path includes COR/det_u, detector roll, axis direction, phi, dx/dz,
  5DOF polish, and final FISTA reconstruction.
- This is a real-data MVP and article/demo candidate.

The productionized MVP must not say:

- All synthetic stopped-reconstruction gauge issues are solved.
- All v1 practical capability has been proven across all datasets.
- The full system is generally production-ready for arbitrary tomography and
  laminography data.
- Pallas is the default end-to-end production alignment backend if the run still
  falls back to JAX for gradient-safe alignment objectives.

Keep the distinction between:

- `real_lamino_mvp`: the clean production/demo path.
- `v1_parity_audit`: the source-of-truth regression/audit mode.
- `synthetic_tomo_mvp`: the required synthetic tomography MVP gates from the
  original v2 plan.
- `synthetic_diagnostics`: separate research diagnostics that should not block
  the real or tomography MVP unless they expose a direct regression.

## Source Documents And Code To Read First

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
- `scripts/real_laminography/run_real_lamino_native_setup_pose_256.py`
- `scripts/real_laminography/run_real_lamino_v2_cor_mvp.py`
- `src/tomojax/data/phantoms.py`
- `src/tomojax/data/simulate.py`
- `src/tomojax/data/contrast.py`
- `src/tomojax/data/artefacts.py`
- `src/tomojax/align/_pose_stage.py`
- `src/tomojax/align/_reconstruction_stage.py`
- `src/tomojax/align/_setup_stage.py`
- `tests/test_real_lamino_runner_contract.py`

Also inspect the artifacts and reports from:

- `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`
- `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_setup_policy_20260511`

## Constraints

Use the existing TomoJAX v2 architecture and deep-module boundaries. Do not add
new generic helper/utils/misc modules.

Do not add or promote:

- COR grid search
- sinogram correlation
- cross-correlation
- sharpness/entropy/autofocus sweeps
- truth support
- true COM
- weak-view exclusions as a success criterion
- synthetic truth as the primary real-MVP success criterion
- benchmark-only knobs as product defaults

Do not hide failures. If a mode is only parity/audit, label it as such. If a
stage is finite but worsens reconstruction, report that plainly.

Synthetic validation during this overnight goal must be bounded. The original
`synthetic128` suite is relevant and may be used, but do not start larger or
open-ended synthetic runs:

- maximum synthetic volume shape: `128 x 128 x 128`
- maximum synthetic detector shape: `128 x 128` unless an existing documented
  `synthetic128` manifest explicitly requires a modest larger detector
- maximum synthetic views: `128` for overnight productionization checks unless
  there is already a completed artifact to summarize
- prefer `32^3`, `64^3`, or binned/smoke settings while iterating

Do not run `256^3` synthetic volumes, 256-view synthetic sweeps, or broad
multi-case benchmark grids tonight unless the run is already completed and only
needs summarizing. If a synthetic check is useful, make it targeted and
artifact-producing.

The required synthetic scope is the tomography MVP, not all five original
synthetic cases. The required gates are:

- `synth128_setup_global_tomo`
- `synth128_pose_random_extreme`

These are MVP gates alongside the real-laminography MVP. Make them runnable,
documented, artifact-producing, and honestly classified at up to `128^3` scale.
Use `32^3` or `64^3` scaled versions while iterating, but leave clear commands
and reports for the final `128^3` target when feasible.

The remaining original cases are not required overnight gates:

- `synth128_lamino_axis_roll_pose`: stretch/follow-up, partly covered by the
  real k11 laminography MVP.
- `synth128_thermal_object_drift`: escalation/future object-motion benchmark.
- `synth128_combined_nuisance_jumps`: hard stress/future benchmark.

The morning deliverable should be usable even if not every stretch task is done.
Prioritize a coherent, clean, validated MVP over adding more experiments.

## Current Known Issues To Address

1. The winning real path is currently exposed as `--v1-parity-real-lamino`.
   That is good as an audit mode but awkward as the public/demo path.

2. `scripts/real_laminography/run_real_lamino_v2_cor_mvp.py` is about 2000
   lines. It has too many responsibilities: data loading, preprocessing, stage
   profile definitions, stage execution, candidate selection, artifact writing,
   v1 parity auditing, and reporting.

3. `src/tomojax/align/_pose_stage.py` is about 2200 lines. It likely mixes pose
   parameterization, smooth constraints, objective construction, acceptance,
   failure guards, and artifact/reporting behavior.

4. `tests/test_real_lamino_runner_contract.py` is about 1200 lines. The tests
   likely need clearer separation between report-contract tests, profile tests,
   fail-closed tests, and parity-audit tests.

5. The branch may contain dirty or uncommitted work. At the start, inspect git
   status. Preserve user/unrelated untracked notes. Commit coherent milestones
   as you complete them.

6. The previous failure mode was severe: phi/dx/dz/polish looked visibly wrong
   until the v1-parity path restored the measured FISTA fallback. Add regression
   coverage so level-2 phi loss cannot silently explode again.

## Required Outcome

By the morning, leave the repository in a state where:

1. There is a clean real-lamino MVP command/profile that a user can run without
   knowing the historical debugging story.

2. There is still a strict v1-parity audit mode that can compare against the
   known reference run.

3. The winning k11 run has a concise, complete report with:
   - final-vs-COR-only metrics
   - v1-vs-v2 comparison
   - stage loss table
   - setup estimates
   - pose summaries
   - visual artifact contact sheets
   - backend/fallback provenance
   - memory/runtime summary
   - explicit caveats

4. The code has been cleaned up enough that the real MVP is not buried in a
   2000-line script as an accidental mode. Do a pragmatic refactor, not a
   months-long rewrite.

5. The synthetic tomography MVP has artifact-producing reports for:
   - `synth128_setup_global_tomo`
   - `synth128_pose_random_extreme`

   These reports must state pass/fail/partial status honestly, include commands,
   run sizes, phantom/projection settings, geometry/pose recovery metrics where
   truth exists, reconstruction-quality metrics, visual artifacts where useful,
   and what remains unproven.

6. Tests cover the public contract and the specific failure modes that mattered:
   - v1 parity profile uses the v1 stage bounds/schedules.
   - real MVP profile resolves to the winning settings.
   - synthetic tomography MVP profiles resolve to bounded setup-global and
     pose-random cases from the original plan.
   - phi level-2 loss scale is sane on the reference parity/audit path.
   - non-finite stage output fails closed.
   - final candidate selection cannot promote black/NaN or objectively worse
     pose-polish output.
   - report artifacts are repo-relative or otherwise portable where expected.

7. Validation has been run and recorded:
   - focused tests for the changed real-lamino/align code
   - focused tests for synthetic tomography MVP contracts
   - `ruff`
   - `basedpyright` for changed files
   - `just imports`
   - broader `just check` if feasible within time; if not feasible, state why.

8. The branch has coherent commits for coherent slices. Do not leave important
   productionization work uncommitted. Leave unrelated existing untracked oracle
   notes alone.

## Implementation Plan

### Slice 1: Snapshot And Morning-Report Baseline

Start by recording the current state:

- `git status --short`
- recent commits
- current `.agent/PLANS.md`
- winning run summary JSON
- current dirty files

Write a short plan update in `.agent/PLANS.md` naming this productionization
milestone. Keep it current as you work.

Create or update a report document:

- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`

This should become the morning-facing report. Initially seed it with the known
winning run and update it as validation proceeds.

### Slice 1A: Reconcile The Original v2 Plan Without Restarting It

The original v2 documents are still relevant. They are the north-star
architecture and article story, but they are not a literal overnight checklist.

Create a concise plan-status section in:

- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`
- `.agent/PLANS.md`

It must answer, phase by phase, how much of the original plan is implemented:

- Phase 0 baseline and benchmark contract
- Phase 1 geometry graph, parameters, and gauges
- Phase 2 JAX reference forward model and residual loss
- Phase 3 FISTA / Huber-TV preview reconstruction
- Phase 4 pose-only 5x5 LM
- Phase 5 setup-only LM
- Phase 6 joint setup+pose Schur LM
- Phase 7 alternating solver and continuation
- Phase 8 nuisance and weak DOF handling
- Phase 9 Pallas fast paths
- Phase 10 experimental modules

Use statuses like `implemented_for_real_mvp`, `partial`, `diagnostic_only`,
`not_proven`, and `out_of_scope_for_mvp`. Be explicit that the current overnight
goal is not to complete Phases 8-10 and not to make broad production claims.
The goal is to make the real-laminography MVP clean, reproducible, and honest
while preserving a clear roadmap back to the original v2 architecture.

Do not blindly implement every item in `04_phased_implementation_plan.md`.
Instead, reconcile the current implementation with that plan and pull forward
only items that directly support the real MVP, documentation honesty, artifact
quality, or bounded synthetic regression tests.

### Slice 1B: Inventory And Restore The Rich Synthetic Story

V1 had useful rich synthetic phantoms: displaced/misaligned objects, random
cubes/squares/spheres, realistic shapes, and Beer-Lambert-style intensity /
absorption handling. V2 already has some of this:

- `src/tomojax/data/phantoms.py` includes spheres, cubes, rotated cubes,
  blobs, Shepp-Logan-like phantoms, random cubes+spheres, and laminography disk
  helpers.
- `src/tomojax/bench/alignment_scenarios.py` includes the PHANTOM94
  random-shapes scenario.
- `src/tomojax/data/contrast.py` contains Beer-Lambert transmission /
  absorption conversion helpers.
- `src/tomojax/data/artefacts.py` contains Poisson/Gaussian noise, detector
  blur, stripes, dead/hot pixels, zingers, dropped views, and intensity drift.
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md` and
  `docs/tomojax-v2/07_synthetic_generator_pseudocode.md` specify a richer
  future generator with ellipsoids, rods/fibres, beads/fiducials, thin plates,
  voids, asymmetric marker clusters, and weak texture.

Do a lightweight inventory and cleanup so the story is coherent:

1. Document what rich phantom / Beer-Lambert / artefact functionality exists
   today and what is still only in the design docs.
2. Ensure the real MVP docs do not imply the full synthetic generator is already
   complete if it is not.
3. Add or update bounded smoke/regression tests proving that the public
   synthetic path can:
   - make a nontrivial random cubes+spheres phantom,
   - apply Beer-Lambert transmission/absorption roundtrip,
   - apply deterministic projection artefacts/noise,
   - run a targeted misaligned synthetic forward/reconstruction/alignment
     contract if such a fast path already exists.
4. If adding a richer phantom adapter is straightforward, add it behind the
   existing public data/simulation API and test it at no more than `128^3`. If
   it is not straightforward, write a clear TODO/roadmap entry instead of
   starting a large generator rewrite.

Do not turn synthetic diagnostics back into the main blocker for the real
laminography MVP, but do treat the two tomography MVP gates below as required
deliverables.

### Slice 1C: Implement The Synthetic Tomography MVP Gates

The original five-case plan includes two core tomography cases that should be
part of the MVP story:

1. `synth128_setup_global_tomo`
2. `synth128_pose_random_extreme`

Make these two cases runnable and reportable through a clean synthetic
tomography MVP path. This can be a script, CLI profile, or benchmark runner,
but it must not be a pile of one-off shell commands. It should emit the same
kind of provenance/report artifacts expected by the v2 verification contract.

For `synth128_setup_global_tomo`, test classic global setup errors:

- parallel tomography
- rich random-shapes phantom, preferably PHANTOM94 or the documented rich
  synthetic phantom if implemented
- detector/COR shift
- detector roll
- axis rotation
- theta offset
- no true per-view pose

Expected result: the solver should materially improve reconstruction quality
over unaligned/COR-only baselines and recover the dominant setup correction
honestly. At minimum, det_u/COR should move in the right direction and by a
plausible magnitude; if roll/axis/theta are gauge-limited or weak at this stage,
record that explicitly rather than hiding it.

For `synth128_pose_random_extreme`, test per-view pose recovery:

- parallel tomography
- setup nominal
- random per-view dx/dz
- random per-view phi
- alpha/beta if the current solver path supports them
- hard-ish but bounded noise/artefacts

Expected result: the solver should materially improve reconstruction quality
and reduce pose error. Report dx/dz/phi recovery directly. Report alpha/beta as
`recovered`, `weak`, `not_supported`, or `not_evaluated`; do not turn missing
DOFs into a green pass.

Required artifacts for each synthetic tomography MVP case:

- machine-readable summary JSON
- markdown report
- manifest with phantom, geometry, pose, noise/artefact, seed, and run-size
  provenance
- geometry/pose truth and recovered summaries
- reconstruction metrics against truth
- residual/loss trace
- final reconstruction PNG/contact sheet
- failure classification if not a pass

Required size policy:

- iterate at `32^3` or `64^3`
- final target may run at `128^3`
- do not exceed `128^3`
- do not exceed `128` views unless summarizing existing completed artifacts
- if `128^3` is too slow, produce a clear partial result at `64^3` plus a
  documented exact command for the `128^3` run and explain what blocked it

Create or update:

- `docs/benchmark_runs/2026-05-12-synthetic-tomo-mvp.md`

This report must answer:

1. Are the two tomography MVP cases implemented?
2. What exact commands were run?
3. What sizes were run?
4. Did setup-global tomography recover COR/setup well enough?
5. Did pose-random tomography recover per-view motion well enough?
6. What artifacts should Tristan inspect?
7. What failed or remains unproven?

Do not make the hard original cases overnight blockers. Mention
`synth128_lamino_axis_roll_pose`, `synth128_thermal_object_drift`, and
`synth128_combined_nuisance_jumps` in the plan-status matrix as follow-up or
stress/future cases.

### Slice 2: Separate Profiles From Execution

Refactor the real-lamino runner so the working behavior is expressed as explicit
profiles rather than scattered flags.

Introduce focused internal structures/functions owned by the real-laminography
script area. Keep them private to the script package unless a deep module
already owns the concept.

The profile separation should include at least:

- `real_lamino_mvp`
  - clean production/demo profile
  - uses the working v1-derived schedule and measured FISTA fallback where
    required
  - runs all stages through final reconstruction
  - emits complete report/artifacts
- `v1_parity_audit`
  - strict behavioral parity mode
  - compares against the reference v1 run
  - emits parity tables
  - should be allowed to be slow and verbose
- `diagnostic_fast` or equivalent
  - optional smoke/debug profile
  - clearly not a production-quality success criterion

Do not break existing tests or scripts unless you update them intentionally.

If retaining the old command-line flag `--v1-parity-real-lamino`, make it an
alias for the audit profile. Add a clean way to run the MVP profile, for
example:

```text
--profile real_lamino_mvp
```

or a dedicated documented command if that is already the repo pattern.

### Slice 3: Make The Winning Path Reproducible And Not Accidental

Ensure the real MVP profile uses the settings that actually worked:

- full 256 detector input
- slab shape `[256, 256, 96]`
- COR/det-u setup
- detector roll
- axis direction
- phi
- dx/dz
- 5DOF polish
- final FISTA
- reference-scale reconstruction iterations
- v1-like stage schedules and bounds
- measured FISTA fallback for real phi parity where needed
- streamed view batching
- fail-closed finite-volume checks
- final candidate selection that can select the genuine best real candidate,
  but does not hide the fact that a stage degraded if that happens

Make sure the profile records all of this in `run_manifest.json` and
`real_mvp_summary.json`.

### Slice 4: Refactor The Worst File Sprawl

Pragmatically reduce the worst concentration of behavior.

Targets:

- `scripts/real_laminography/run_real_lamino_v2_cor_mvp.py`
- `src/tomojax/align/_pose_stage.py`
- `tests/test_real_lamino_runner_contract.py`

Do not do a risky rewrite. Extract stable, obvious chunks:

For the real-lamino script, consider separating:

- profile/schedule construction
- input/preprocessing setup
- report/contact-sheet generation
- v1 parity table generation
- stage artifact validation
- final candidate policy

For `_pose_stage.py`, consider separating private helpers for:

- pose model/profile resolution
- smooth constraint application
- stage update guards
- objective/loss scale diagnostics
- artifact/stat row construction

For tests, separate logically if practical:

- real-lamino profile contract tests
- real-lamino report artifact tests
- v1 parity audit tests
- fail-closed/non-finite tests
- pose guard tests

Keep imports within deep-module boundaries. If new private files are added under
`src/tomojax/align`, keep them private `_*.py` and do not expose them unless
needed.

### Slice 5: Parity And Regression Tests

Add or update tests so the exact failures from this week cannot recur silently.

Required checks:

1. The v1 parity/audit profile uses v1-like schedules and bounds:
   - setup levels `8,4,2`
   - phi levels `4,2,1`
   - dx/dz levels `4,2,1`
   - polish levels `2,1`
   - phi bounds approximately `+-5 deg`
   - dx/dz bounds approximately `+-16 px` at full resolution
   - polish bounds approximately v1-style

2. The real MVP profile resolves to the same stage order:
   - `00_baseline`
   - `01_setup_geometry/01_cor`
   - `01_setup_geometry/02_detector_roll`
   - `01_setup_geometry/03_axis_direction`
   - `02_pose_phi`
   - `03_pose_dx_dz`
   - `04_pose_polish`
   - `05_final`
   - `06_cor_only_fista`

3. The phi level-2 loss scale regression is guarded. Use a small fixture or a
   parsed reference artifact if a full run is too expensive in unit tests. The
   test should fail if level-2 phi loss is absurdly larger than v1-like scale
   without an explicit failure label.

4. Non-finite reconstructions cannot propagate to downstream stages or final
   selection.

5. Final candidate selection:
   - selects the lowest valid final reconstruction loss
   - records the candidate source stage
   - does not promote invalid or missing artifacts
   - records when a later stage was rejected/degraded

6. Reports remain portable and artifact paths resolve correctly.

### Slice 6: Run The Real MVP Validation

Run the clean real MVP profile on the k11 reference input.

Suggested output:

```text
runs/real_lamino_v2_production_mvp_k11_54014_20260512
```

Use the same input and reference report:

```text
/home/tristan/projects/tomojax/runs/real-lamo-256/k11-54014_corrected_log_256cube.nxs
runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/real_mvp_summary.json
```

If full rerun time is too high, do not fake it. Use the completed winning run as
the evidence source, but make the production MVP profile and tests prove it
would run the same settings. Clearly record whether the new production-named run
was executed or whether the winning parity run was promoted as evidence.

If possible, also run one additional real scan or binned real scan as a
sanity-check. If another full real scan is unavailable or too expensive, run a
deterministic binned real smoke only as a smoke/regression test, not as a
quality claim.

### Slice 7: Visual Artifacts And Contact Sheets

Generate and save useful PNG comparison artifacts. Treat these as first-class
deliverables.

Required:

- v2 production MVP stage contact sheet for central slice
- v2 production MVP stage contact sheet for orthos
- v2 production MVP stage contact sheet for z-stack
- direct v1-vs-v2 final comparison
- direct v2 COR-only-vs-final comparison
- optional delta/contact sheets if already supported

Save them under the run report or a publication directory, and reference them
from:

- `real_mvp_summary.md`
- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`

### Slice 8: Documentation Cleanup

Update docs so a reader can understand the state without reading chat history.

Required docs:

- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`
- `docs/implementation_log.md`
- `.agent/PLANS.md`

The benchmark run doc should include:

- command(s)
- commit hash
- input dataset
- reference run
- stage schedule
- stage loss table
- final-vs-COR-only comparison
- v1-vs-v2 comparison
- setup and pose summaries
- artifact paths
- GPU/memory/runtime summary
- what worked
- what remains unproven
- exact validation commands and results

Also consider a concise top-level or `scripts/real_laminography/README.md`
update showing how to run the real-lamino MVP and how to run the parity audit.

### Slice 9: Final Validation And Commits

Run appropriate validation:

- focused pytest for changed tests
- focused pytest for real-lamino runner contracts
- focused pytest for pose-stage guard/profile logic
- `ruff check` on changed files
- `basedpyright` on changed Python files
- `just imports`
- `just check` if feasible

If `just check` is too slow or fails for unrelated known reasons, record that
plainly. Do not claim it passed unless it did.

Commit coherent slices:

1. profile/refactor slice
2. tests/regression slice
3. report/artifact/doc slice

Do not accidentally commit unrelated oracle notes unless they are explicitly
part of the productionization deliverable.

## Morning Report Requirements

Before marking the goal complete, write a final morning report in:

```text
docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md
```

It must answer:

1. Is TomoJAX v2 now at real-lamino MVP parity with v1?
2. What exact run proves it?
3. What are the final, COR-only, and v1 reference losses?
4. What stages improved the reconstruction?
5. What visual artifacts should Tristan inspect?
6. What code was cleaned up?
7. What tests and checks passed?
8. What remains messy or unproven?
9. What should be done next before a public article/release?

Use direct, honest wording. Avoid victory-lap language. The point is to make
the repo ready for Tristan to review and decide, not to pretend all remaining
research is done.

## Suggested Final Success Criteria

The goal is complete when all of these are true:

- A clean real-lamino MVP profile exists and is documented.
- Strict v1 parity audit mode still exists and is documented.
- The winning k11 result is preserved in a machine-readable and human-readable
  report.
- V2 final real reconstruction is shown to improve over COR-only and to match
  or beat the v1 reference on the reported loss.
- Visual comparison artifacts are generated and easy to find.
- The worst code sprawl has been reduced or at least isolated behind clearer
  profile/report/stage functions.
- Tests cover the critical profile, fail-closed, candidate-selection, and phi
  loss-scale regressions.
- Validation results are recorded.
- Coherent commits exist.
- `.agent/PLANS.md` and `docs/implementation_log.md` are current.

If time runs short, prioritize:

1. clean profile + docs + tests around the winning behavior,
2. artifact/report quality,
3. code refactor,
4. extra scans.

Do not sacrifice correctness or honesty to make the morning report look greener.
