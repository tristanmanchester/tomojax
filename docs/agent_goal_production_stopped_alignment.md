# Agent Goal: Production Stopped Alignment Consolidation

## Objective

Consolidate TomoJAX v2 around one honest production stopped-alignment gate.

Do not continue broad feature work until this gate is understood. Specifically:

- Do not add new geometry DOFs.
- Do not expand nuisance fitting.
- Do not expand laminography.
- Do not work on Pallas or fast paths.
- Do not chase the full five-case benchmark suite.
- Do not add more candidate-refresh, polish, weak-view, or policy variants unless they are explicitly diagnostic and answer a stated hypothesis.

The current project state is:

- Fixed-truth / oracle geometry recovery works for supported geometry.
- The v2 Schur geometry solver is therefore mostly coherent when the volume is in the correct gauge.
- Production stopped reconstruction still absorbs setup geometry.
- Phase 7, the real reconstruct-align-reconstruct loop, is the current blocker.

Treat this as the main goal:

> Make one minimal stopped-reconstruction supported-only detector-shift gate honestly work, or produce decisive evidence for why the current algorithm cannot make it work.

## Required Reading

Before changing code, read:

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `.agent/PLANS.md`
- `docs/implementation_log.md`
- The latest oracle feedback in the recent session context

The oracle verdict to preserve:

- v2 currently has a strong oracle geometry solver inside a diagnostic harness.
- It is not yet a working production alignment/reconstruction system.
- More benchmark knobs are now a risk.
- The shortest honest path is to make a single supported-only stopped-alignment gate pass without weak-view exclusion, nuisance, unsupported DOF hiding, or truth-volume assistance.

## Target Production Gate

Use this as the canonical gate:

- Synthetic supported-only parallel tomography.
- `64^3` first.
- `64` to `128` views.
- Clean data.
- Stopped reconstruction volume, not true/fixed truth.
- Pose frozen.
- Theta frozen, or explicitly labelled as a volume-orientation gauge.
- `det_u` active only.
- `det_v` frozen unless the specific diagnostic requires it.
- No nuisance.
- No bad-view exclusion.
- No unsupported-DOF pass path.

Initial success:

- `det_u` RMSE improves materially and reaches `< 1 px` at `64^3`.

Stretch success:

- `det_u` RMSE reaches `< 0.2 px`.
- `true_volume/final_geometry` loss approaches `true_volume/true_geometry` loss.
- `final_volume/true_geometry` and `final_volume/final_geometry` losses are no longer strongly divergent.
- Volume NMSE improves rather than getting worse.

Always record:

- Det_u proposed Schur step.
- Det_u accepted/final error.
- Whether Schur accepted or rejected.
- `preview_volume / initial_geometry` loss.
- `preview_volume / true_geometry` loss.
- `true_volume / final_geometry` loss.
- `final_volume / true_geometry` loss.
- `final_volume / final_geometry` loss.
- Volume NMSE.
- Artifact path.
- Exact command, seed, device, runtime, and peak sampled GPU memory when available.

## Work Order

### 1. Consolidation Note

Update `docs/implementation_log.md` with a short, clear consolidation note:

- What works now.
- What does not work yet.
- Fixed-truth gates are oracle-only.
- Stopped supported-only det_u recovery is the production blocker.
- Weak-view exclusion is diagnostic, not a plain production pass.
- Candidate refresh has not solved the production stopped-loop blocker.

Do not oversell any green gate.

### 2. FISTA Absorption Curve

Run a direct absorption diagnostic on the canonical `64^3 / 64-view` supported-only det_u case.

Use:

- Same neutral or support-projected initial volume for every run.
- Theta frozen.
- Pose frozen.
- Det_u active only.
- No nuisance.
- One det_u-only Schur solve after each preview.

Sweep preview FISTA iterations:

- `0`
- `1`
- `2`
- `4`
- `8`
- `16`

For each run record:

- Det_u proposed step.
- Det_u final error.
- Schur accepted/rejected.
- Preview volume loss under initial geometry.
- Preview volume loss under true geometry.
- True volume loss under final geometry.
- Final volume loss under true geometry.
- Volume NMSE.

Interpretation:

- If det_u recovery is best at `0` or `1` iterations and worsens as FISTA improves the preview residual, then the preview reconstruction is absorbing setup geometry.
- If Schur cannot recover det_u even from a neutral or zero-iteration preview, the current stopped objective may not provide a useful detector-shift gradient.
- If more iterations help, the prior assumption was wrong and the next plan must be revised from evidence.

### 3. FISTA Gradient / Adjoint Check

Add a focused tiny-volume finite-difference test for the reference FISTA reconstruction loss.

Compare:

- The explicit gradient used by FISTA.
- A finite-difference gradient of the exact same filtered/masked loss.

Start with:

- Raw residual.
- No TV.
- No center penalty.
- No heldout mask.

Then add coverage as feasible:

- Lowpass residual.
- Train mask / heldout mask.
- Support mask.
- Center penalty.
- TV on.

If this fails, fix the FISTA gradient / adjoint before adding more orchestration.

This is important because a hand-coded reconstruction gradient that is not the adjoint of the reported loss can make all downstream stopped-loop diagnostics misleading.

### 4. Geometry-First Det_u Bootstrap

Do not expand candidate refresh as the production cure. It improved some residuals but did not recover geometry.

Instead implement or diagnose a minimal geometry-first bootstrap:

1. Build a neutral, support-projected, nonnegative initial volume.
2. Run a det_u-only Schur update before serious FISTA reconstruction can absorb the corrupted geometry.
3. Use lowpass residuals.
4. Freeze theta.
5. Freeze pose.
6. Refresh reconstruction under the updated det_u.
7. Run one more det_u-only Schur update.

Success:

- `64^3` det_u improves from the current `~2.2-2.9 px` plateau to `< 1 px`.
- `128^3` det_u improves materially from the current `~6.586 px` failure if tested.
- Candidate refresh is not required for the production path.

Falsification:

- If neutral/bootstrap Schur still cannot move det_u beyond the current plateau, the issue is not just stale-volume absorption. Stop and write the next diagnosis from evidence.

### 5. Real Multiresolution, If Needed

If the absorption curve indicates capture range or scale is the issue, implement a real multiresolution pyramid for the supported-only det_u path.

Do not just use residual filters and call it multiresolution.

A real level should:

- Downsample projections.
- Downsample or reconstruct at the corresponding volume scale.
- Scale detector shifts and pose shifts appropriately.
- Solve det_u at low resolution.
- Carry the solved geometry forward to the next level.

Suggested levels:

- Level 4: downsampled projections/volume, det_u scaled by `1/4`.
- Level 2: refine with det_u scaled by `1/2`.
- Level 1: full-resolution verification.

Only implement this first for supported-only det_u.

Success:

- Level 4 captures the large detector shift.
- Level 2 refines it.
- Level 1 verifies without requiring many extra policy knobs.

### 6. Explicit Theta Policy

Theta offset is a volume-orientation gauge in stopped reconstruction unless there is an orientation anchor.

Implement or document two modes:

#### Reconstruction-first mode

- Theta frozen during stopped det_u bootstrap.
- Theta error is not counted as a production failure without an orientation anchor.

#### Calibration mode

- Theta active only when an orientation anchor exists, such as truth, fiducials, known asymmetric support, or a clearly defined calibration convention.
- Theta recovery can be evaluated here.

Do not conflate det_u recovery with theta gauge ambiguity.

### 7. Benchmark Honesty Fixes

Tighten labels and status semantics.

Use these meanings:

- Fixed-truth pass: `oracle_pass`, not `production_pass`.
- Weak-view exclusion: `passed_with_exclusions` or `partial_oracle_pass`, not plain pass.
- Stopped supported-only failure: `failed_absorbed_geometry` or `failed_untrusted_volume`.
- Unsupported DOFs: `unsupported_partial` or `not_evaluated_for_full_case`, never a green full-case pass.
- Catastrophic volume NMSE or huge residual must override any benign-sounding consistency classification.

Also fix known honesty issues:

- If `theta_scale` is filtered out of active setup names in the alternating path, either make it truly active where claimed or mark it unsupported there.
- Do not let unsupported or excluded metrics produce a plain passed production gate.

### 8. Cleanup / Demotion

Demote these from production defaults unless a diagnostic proves they are required:

- Candidate-refresh acceptance.
- Hard x-gauge projection.
- Weak-view exclusion as a pass criterion.
- Neutral-refresh variants.
- No-FISTA-first as a user-facing production policy.
- Extra final polish stages that mask rather than solve the production loop.

They may remain as diagnostics if clearly named and documented.

## Validation Requirements

After each completed slice:

- Run focused pytest coverage for changed code.
- Run ruff on changed files.
- Run basedpyright on changed files.
- Run `just imports`.
- If a meaningful milestone completes, run the canonical stopped det_u gate.

Commit each completed slice separately with a clear message.

After every benchmark:

- Update `docs/implementation_log.md`.
- Record command, artifact path, metrics, and interpretation.
- State whether the result confirms or falsifies the hypothesis.

## Stop Conditions

Stop broad coding and write an evidence-backed diagnosis if:

- FISTA gradient checks fail in a way that invalidates current stopped-loop results.
- Zero/neutral preview Schur cannot recover det_u.
- Real multiresolution fails to improve det_u recovery.
- The only apparent path to green requires weak-view exclusion, truth-volume information, unsupported-DOF hiding, or extra benchmark-specific knobs.

## What Not To Do

Do not:

- Add more geometry DOFs before stopped det_u works.
- Tune nuisance before stopped det_u works.
- Expand five-case benchmarks before stopped det_u works.
- Add more CLI flags to `align-auto-smoke` unless necessary for a named diagnostic.
- Interpret Schur train loss as final success.
- Interpret fixed-truth geometry as production alignment success.
- Require stopped theta recovery without an orientation anchor.
- Speed up the wrong loop with Pallas before the algorithm is correct.

## Expected Final Report

When stopping or completing this goal, report:

- Current HEAD.
- Commits made.
- Whether the canonical stopped det_u gate passes.
- Best 64^3 det_u result and artifact path.
- Whether FISTA absorption was confirmed.
- Whether FISTA gradient checks passed.
- Whether geometry-first bootstrap helped.
- Whether real multiresolution was implemented or deferred.
- Which policies were demoted from production defaults.
- Remaining blockers and exact next recommendation.
