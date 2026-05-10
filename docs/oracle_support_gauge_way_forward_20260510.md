# Oracle Support/Gauge Way Forward

Date: 2026-05-10

This document consolidates the Oracle review of the stopped `det_u` variable-projection diagnostics. It is the next implementation brief for TomoJAX v2. The goal remains differentiable physical optimisation: no COR finders, no sinograms, no cross-correlation, no sharpness/entropy sweeps, no grid-search alignment, and no benchmark laundering.

## Executive Summary

The latest diagnostic is directionally right, but the evidence should be stated precisely.

It convincingly shows that scalar projection physics and the fixed-volume detector-u objective contain the correct detector-centre signal. With the true volume fixed, the `det_u` landscape has the correct minimum.

It also shows that the stopped reconstruction path can move that signal into the reconstructed volume. The later geometry step then optimises a biased fixed-volume objective.

The safest conclusion is:

> An informative object-frame support/gauge can restore detector-u identifiability, but the current reduced-objective implementation is too under-reconstructed and too flat to prove that the specific known-phantom-support reduced objective is production-ready.

The main loss of geometry information is not Schur, JVP/VJP, detector-u sign, or projector convention. It is the x-step gauge: with insufficient object-frame constraints, FISTA can represent detector-centre error as a translated, warped, or attenuation-adjusted volume. The gauge-transfer diagnostic showing roughly 85-87% transfer of the detector-u tangent into volume space is the key smoking gun.

The next production direction should be a two-volume alignment/reconstruction formulation:

1. A constrained differentiable alignment volume estimates geometry. It carries a frozen, data-derived object-frame support/gauge from a neutral scout reconstruction, and may later include a low-frequency anchor and tangent-space gauge projection that prevents detector-u from being represented as a volume update.
2. A freer final reconstruction volume is reconstructed after geometry is solved. This avoids pretending the support prior is the final scientific reconstruction.

Do not move on to more DOFs, Pallas kernels, nuisance terms, weak-view exclusions, or benchmark reframing. Solve this detector-u gauge first.

## What The Latest Diagnostic Proves

The decisive evidence is the combination of three results.

First, the true-volume fixed objective has its minimum at the true detector-u. In the 64^3 report, `true_volume_fixed_objective` has argmin `7.25`, exactly truth, with zero error. This means the forward model, detector-u convention, scalar projection loss, and sampled detector-u landscape contain the correct physical signal when the volume is held in the correct object frame.

Second, the final stopped-volume fixed objective has its minimum at the wrong detector-u, around `5.574625`, not true `7.25`. That is the signature of gauge absorption: the stopped x-step returns a volume self-consistent with wrong geometry, so the geometry optimiser sees a plausible but physically biased objective.

Third, the gauge-transfer diagnostic gives the local explanation. At 32^3, 64^3, and 128^3, roughly `0.85-0.87` of the detector-u projection tangent is transferable into the volume tangent space. At 64^3 the artifact reports transfer ratio `0.8570` and reduced/fixed curvature ratio `0.1430`; 128^3 is similar at `0.8672`. That says an unconstrained x-step can spend volume degrees of freedom to explain detector-u motion.

So the broad conclusion is justified: some object-frame support/gauge information is needed to make the stopped x-step geometry-informative.

## What The Diagnostic Does Not Prove

The label `constraint_restores_geometry_information:reduced_known_phantom_support` is too strong if read as an algorithmic solution. It should be treated as a diagnostic classification.

Known phantom support is cheating. It proves that support/gauge information can restore identifiability, not that the current known-support reduced objective is production-ready.

The reduced objectives are extremely flat. In `objective_summary.json`, honest and ordinary-support reduced families have losses around `51.7237` across the candidate set, with volume NMSE around `0.9999`. The known-phantom-support row also has volume NMSE around `0.9999`, and its loss differences are tiny. These rows are mostly judging tiny variations around a nearly zero or severely underfit volume.

The diagnostic FISTA settings are not production-equivalent. Reduced probes currently use very small budgets and settings compared with production preview/candidate-refresh settings. In particular, source review found reduced probes using tiny iteration counts, `step_size=2.0e-3`, `initial_volume=None`, and no state carry, while production preview uses much larger `_preview_fista_step_size` values and backprojection or carried initialisation depending on orchestration. This mismatch means reduced-objective rows must not be overinterpreted.

The TV variants in the latest artifacts may not be meaningful TV tests. Reconstruction configs show `tv_weight: 0.0` for several reduced families whose names imply TV. The result does not prove properly weighted TV cannot help; it proves the currently run variants did not help.

The ordinary support definitions are too weak to break the relevant gauge. A broad centered cylinder/sphere with default radius fraction is FOV support, not object-specific support. It prevents gross mass outside the FOV but does not pin the object frame tightly enough to distinguish detector-u shift from object translation.

Loss scaling can suppress masked signals. Candidate comparisons are internally comparable under the same mask, so this does not invalidate the fixed-volume truth result, but it changes conditioning, curvature magnitudes, FISTA step-size interpretation, and apparent flatness.

The candidate grid is diagnostic, not production. In flat reduced curves, an argmin at `6.25` versus `7.25` may be numerical tie, grid artifact, or tiny regularisation effect. The follow-up is not a denser production grid search; it is local derivative, curvature, transfer-ratio, and properly solved reduced-objective evidence.

## Where Geometry Information Is Being Lost

The loss is in the stopped x-step / volume gauge.

The forward model combines setup detector-u and per-view pose dx, and pose-gauge canonicalisation handles the pose/setup gauge. It does not fix the larger object-frame gauge: the reconstructed volume can translate or redistribute to compensate detector-centre error.

The FISTA objective is too permissive. Nonnegativity does not define an object frame. A broad cylinder does not define object support. Weak TV can smooth absorbed error rather than forbid it. A generic center penalty is ill-conditioned for near-zero volumes and does not encode actual object support or the detector-u tangent.

The FISTA trace has an honesty issue: loss is evaluated at the momentum variable while the returned state is the projected candidate volume. This probably does not explain detector-u failure by itself, but it makes traces and reduced-objective claims harder to interpret.

Masking is mostly not the bug, but it contributes to handoff mismatch. FISTA now uses `projection_valid_mask`, while Schur/alignment scoring uses Otsu-derived alignment masks. Since the true fixed-volume objective is still correct under the alignment mask, the mask is not the primary cause. But x-step and geometry step are not optimising exactly the same objective, and full-array normalisation after masking changes effective conditioning.

Reduced-objective implementation is currently too weak to be decisive. If it reconstructs from neutral zero volume with tiny step size and tiny iteration count, then volume NMSE near one means the inner minimisation is not solved well enough to support a strong variable-projection conclusion.

Multires carry/initialisation helps image quality but not geometry. Carrying a volume can carry the absorbed gauge. It improves reconstruction under wrong geometry without restoring object-frame constraints.

The Schur scalar path is substantially exonerated. The diagnostic artifacts show det-u-only active setup, positive curvature, and sign agreement. Schur is taking gradients of the wrong stopped-volume objective, rather than being intrinsically broken.

Shortest diagnosis:

> TomoJAX currently has a differentiable geometry optimiser wrapped around a reconstruction objective that has not fixed the object-frame gauge. The optimiser is therefore differentiating a self-consistent but physically wrong volume.

## Production Replacement For Known Phantom Support

The honest replacement is an explicit, provenance-recorded, differentiable object-frame support/gauge used for alignment only.

Implement a two-volume formulation:

- `x_align`: constrained alignment volume. It is not the final scientific reconstruction. Its job is to make detector-u identifiable by preventing x-step absorption.
- `x_final`: final reconstruction volume. Once geometry is solved, reconstruct this more freely with weaker support/TV/nonnegativity.

The immediate replacement for known phantom support should be frozen soft support plus low-frequency anchor from a neutral scout reconstruction.

### Frozen Scout Soft Support

Run one neutral/coarse scout reconstruction using only metadata/initial geometry, full projection-valid mask, nonnegativity, and broad generic support. Do not update det-u while making the scout.

Smooth the scout strongly in 3D. Convert it into a soft support probability `p(r) in [0,1]` using robust intensity statistics, then dilate and soften conservatively. Save support and provenance before alignment.

In the alignment x-step, add differentiable penalties:

```text
lambda_out * mean(((1 - p) * x)^2)
lambda_anchor * mean((LP(x) - LP(x_scout))^2)
```

The outside-support term prevents object mass outside the object-frame support inferred before alignment. The low-frequency anchor prevents translating the low-frequency object envelope to absorb detector-u. Both are differentiable, physical, and non-grid-search.

Prefer soft support first. Hard support from a biased scout can overconstrain and bake in wrong geometry.

### Scout-Derived COM And Moment Gauge

The current center penalty is generic. Replace or supplement it with scout-derived low-frequency COM/moment constraints.

Compute scout support COM and covariance after freezing the scout. Penalise deviations of the alignment volume's low-frequency mass distribution from those scout-derived moments:

```text
lambda_com * ||COM(LP(x)) - COM_scout||^2
lambda_cov * ||Cov(LP(x)) - Cov_scout||^2
```

These are honest because they are derived from data once, frozen, and recorded. They do not scan detector-u or use phantom truth.

### Tangent-Space Gauge Projection

The gauge-transfer diagnostic can become a regularised tangent-space projection.

The diagnostic solves a version of:

```text
min_delta_x ||W A delta_x - q_u||^2 + lambda ||L delta_x||^2
```

where `q_u = dA(u)x/du` is the detector-u projection tangent. High transfer ratio means detector-u can be represented by a volume update.

Promote this into a gauge constraint. At a scout/current alignment volume, compute the volume mode `v_u` that best mimics detector-u. Then either project FISTA updates away from this mode or add a penalty on the coefficient of `x - x_ref` along this mode:

```text
R_gauge(x) = 0.5 * lambda_gauge * (<x - x_ref, H v_u>^2 / <v_u, H v_u>)
```

Start with detector-u only. This is not COR, sinogram, or grid search. It is a differentiable gauge constraint saying volume updates must not spend the tangent direction corresponding to the geometry parameter being estimated.

Do not implement tangent-space gauge alone. It needs a reasonable scout/reference volume. Best near-term path:

1. Frozen scout soft support + low-frequency anchor.
2. Tangent-space gauge if transfer remains high.

Projection-consistency or visual-hull support is lower priority. It can help, but projection silhouettes can bake in the detector-centre offset being solved. If used, make it a broad union-over-uncertainty envelope, not a detector-centre selector.

Do not use learned priors next. The failure is explainable by a concrete gauge.

## Implementation Slice 1: Make Reduced-Objective Diagnostics Honest

Goal: remove diagnostic artifacts before judging support variants. Reduced-objective probes should use reconstruction budget, step-size policy, initialisation, support, and loss reporting explicitly comparable to the production stopped path.

Likely code areas:

- `src/tomojax/align/_alternating_reduced_objective.py`
- `src/tomojax/align/_alternating_detu_landscape.py`
- `src/tomojax/align/_alternating_orchestration.py`
- `src/tomojax/recon/_fista_reference.py`
- `tools/run_detu_variable_projection_diagnostic.py`

Concrete changes:

- Use the same step-size policy as production preview/candidate-refresh, or record loudly when not doing so.
- Do not silently use tiny diagnostic step size if production uses much larger preview settings.
- Record loss at the returned FISTA candidate, not only at the momentum variable. Add `returned_candidate_loss`, `returned_data_loss`, and `returned_regularizer`.
- Add stationarity/prox-gradient metric.
- Record volume norm, support mass, iteration count, step size, init source, support/anchor source, and synthetic-only volume NMSE.
- Support zero init, backprojection init, and carried init as separate labelled modes.
- Mark reduced-objective families as `inner_solve_underfit` when all candidate volumes are near-zero or stationarity is poor. Do not report a production-significant argmin from a flat underfit curve.

Tests/artifacts:

- Unit test asserting reduced-objective reconstruction config records step size, iteration count, initialisation, support source, mask source, and loss normalisation.
- Regression test that fails or marks underfit if a reduced-objective family reports an argmin while all candidate volumes are near-zero or stationarity is poor.
- Update `objective_summary.json`, per-family `reconstruction_config.json`, per-family curve CSV, and new `inner_solve_quality.json`.

Benchmark command:

- Use the existing 64^3 variable-projection diagnostic command from `docs/benchmark_runs/2026-05-09-variable-projection-detu-64.md`, adding production-equivalent x-step settings as needed.

Success criteria:

- The diagnostic separates `inner_solve_underfit` from `geometry_information_absent`.
- If production-equivalent reduced objective still prefers wrong det-u, support/gauge diagnosis is stronger.
- If it recovers the true basin, the immediate production fix is the comparable reduced acceptance path rather than the current stopped branch.

Avoid cheating:

- Truth det-u and true volume may appear only in labelled diagnostics. No production decision may depend on them.

## Implementation Slice 2: Frozen Scout Soft Support And Low-Frequency Anchor

Goal: replace known phantom support with an honest, data-derived, differentiable object-frame gauge.

Likely code areas:

- `src/tomojax/recon/_support.py` or new `src/tomojax/recon/_scout_support.py`
- `src/tomojax/recon/_fista_reference.py`
- `src/tomojax/align/_alternating_types.py`
- `src/tomojax/align/_alternating_orchestration.py`
- artifact writers under `src/tomojax/align/`

Concrete implementation:

- Add a scout-support builder.
- Input: projections, initial/metadata geometry, projection-valid mask, level shape.
- Output: soft support `p`, scout low-frequency anchor `x_scout_low`, provenance.
- Build scout once per level before geometry updates.
- Use backprojection or short neutral FISTA under fixed initial det-u.
- Smooth scout, threshold robustly, dilate conservatively, convert to soft support probability.
- Save raw scout, smoothed scout, support probability, threshold, dilation radius, mass fraction.

Add FISTA regulariser terms:

```text
support_outside_l2 = lambda_out * mean(((1 - p) * x)^2)
low_frequency_anchor_l2 = lambda_anchor * mean((LP(x) - x_scout_low)^2)
```

Avoid periodic `jnp.roll` smoothing for the anchor. Use nonperiodic padding or existing multires/downsample operators.

Suggested config fields:

```text
preview_support_source = none | centered | scout_soft
preview_support_outside_weight
preview_low_frequency_anchor_weight
preview_scout_freeze_policy = per_level_before_alignment
preview_scout_geometry_source = initial_metadata
```

Artifacts:

- `scout_support.npy`
- `scout_low_frequency_anchor.npy`
- `scout_support_provenance.json`
- support mass fraction
- threshold
- smoothing scale
- dilation scale
- geometry source
- mask source
- `uses_truth: false`

Tests:

- Finite-difference gradient test for soft support penalty and anchor penalty.
- Provenance test ensuring scout support is built before alignment and does not read ground truth, true geometry, or true det-u.
- Synthetic diagnostic artifact may report IoU with true support only after support is frozen and only for evaluation.

Add diagnostic objective families:

```text
reduced_scout_soft_support
reduced_scout_lowfreq_anchor
reduced_scout_support_anchor
```

Benchmark run:

- First run the 64^3 PHANTOM94 variable-projection diagnostic.
- Then run rich phantom stopped det-u diagnostic at 64^3 and 128^3 with det-u only active.

Success criteria:

- Reduced scout-support/anchor objective has a clear basin near true det-u, not a one-point flat argmin.
- First threshold: argmin error <= `0.5 px` at 64^3 with non-tiny loss margin over neighbouring candidates.
- Stopped production path improves final det-u error materially without truth support.
- Volume NMSE does not collapse.

Failure criteria:

- If reduced curve remains flat/wrong, or scout support is almost whole FOV, support is not informative enough.
- If support is too tight and harms reconstruction, soften/dilate and rely more on low-frequency anchor.

Avoid cheating:

- Scout may use only observed projections, projection-valid masks, and initial metadata geometry.
- Scout must not use phantom labels, true support, true COM, true det-u, or detector-u sweep.
- Support must be frozen before alignment, and freeze point must be recorded.

## Implementation Slice 3: Detector-u Tangent-Space Volume Gauge Projection

Goal: prevent the alignment x-step from using the volume mode that mimics detector-u.

Likely code areas:

- `src/tomojax/align/_alternating_gauge_transfer.py`
- new `src/tomojax/recon/_gauge_modes.py`
- `src/tomojax/recon/_fista_reference.py`
- `src/tomojax/align/_alternating_orchestration.py`

Concrete implementation:

- Promote gauge-transfer solve from diagnostic into reusable gauge-mode builder.
- For current scout/alignment volume, compute detector-u projection tangent `q_u = dA(u)x/du`.
- Solve for volume mode `v_u` that best reproduces this tangent under projection-valid mask and current regularisation metric.

Modify FISTA alignment x-step with either:

1. Penalty:

```text
R_gauge(x) = 0.5 * lambda_g * c(x)^2
```

where `c(x)` is coefficient of `x - x_ref` along `v_u` in approximate Hessian or L2 metric.

2. Projection:

After each gradient/prox step, remove the component of candidate update along `v_u`.

Start with det-u only. Do not generalise to theta, tilt, nuisance, or multiple setup DOFs yet.

Tests/artifacts:

- Test showing transfer ratio drops when gauge projection is active.
- Finite-difference test for gauge penalty gradient if using penalty version.
- Artifact recording `transfer_ratio_before`, `transfer_ratio_after`, `mode_norm`, `mode_metric_norm`, and whether mode was recomputed or frozen.

Benchmark run:

- Run gauge-transfer diagnostic before/after gauge projection.
- Run 64^3 variable-projection diagnostic with `reduced_scout_support_tangent_gauge`.
- Then run stopped det-u gate.

Success criteria:

- Transfer ratio falls substantially, ideally from ~`0.85` to below `0.5` initially.
- Reduced objective gains curvature around true det-u.
- Stopped det-u error improves without truth support.

Avoid cheating:

- Tangent mode is computed from scout/current alignment volume and current geometry only.
- It must not be computed using true volume or true det-u.
- It is a local differential gauge constraint, not a candidate sweep.

## What Not To Do Next

- Do not add more geometry DOFs yet.
- Do not add pose, tilt, nuisance, or per-view terms before det-u-only gauge is fixed.
- Do not move to Pallas speed work before the objective is correct.
- Do not use known phantom support, true COM, true support IoU, true det-u markers, or phantom labels in production paths.
- Do not interpret weak-view exclusions, heldout loss improvements, or volume NMSE improvement as success if det-u remains wrong.
- Do not add candidate-refresh/backtracking variants until reduced objective is geometry-informative.
- Do not use sinograms, COR sweeps, cross-correlation, phase correlation, sharpness, entropy, or autofocus sweeps under different names.
- Do not treat latest support/TV/center failures as a theorem. They only show currently implemented broad support, nonnegativity, mostly-zero/weak TV, and generic center penalty did not solve the gauge.
- Do not use an under-converged reduced objective as decisive proof. Report inner-solve quality.

## Conceptual Reframing

Generic stopped reconstruction is fundamentally unable to identify detector-centre reliably without an external object-frame support/gauge.

That does not mean TomoJAX should fall back to traditional COR. It means the differentiable formulation must include the missing gauge condition.

In parallel tomography, detector-u shift and object-frame translation are nearly gauge-equivalent. With a free enough volume, variable projection can fit the data for multiple detector-u values by changing `x`. Differentiability makes this easier to see, not easier to escape. The gauge-transfer diagnostic is a strong sign that the project is asking the right question.

The honest TomoJAX framing is:

> TomoJAX estimates geometry by differentiable physical optimisation of a constrained alignment volume with explicit, documented object-frame gauge/support. It then reconstructs the final volume under solved geometry with fewer constraints.

This preserves the project identity. The support/gauge is not a COR finder; it is the missing boundary condition that makes the inverse problem identifiable.
