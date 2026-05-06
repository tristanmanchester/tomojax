---
title: Use stopped validation-LM for setup geometry calibration
date: 2026-04-25
last_updated: 2026-04-27
category: architecture-patterns
module: TomoJAX alignment geometry calibration
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - "Adding detector or instrument geometry optimization to TomoJAX alignment"
  - "A setup objective needs held-out projection evidence without reconstruction-heavy step search"
  - "A bilevel or cross-validation design starts reconstructing inside L-BFGS or line search"
  - "Fixed-volume setup scoring appears to converge but may be self-consistent under wrong geometry"
  - "Geometry calibration demos need auditable solver provenance and image-quality evidence"
tags: [tomojax, alignment, geometry-calibration, validation-lm, stopped-reconstruction, gpu-memory, l2-otsu, diagnostics]
---

# Use stopped validation-LM for setup geometry calibration

## Context

The geometry-calibration work went through several frustrating false starts before
landing on the current product shape. The durable lesson is not just "reuse
`align_multires`". It is more specific:

```text
setup geometry calibration belongs inside the unified alignment system,
but setup discovery should be solved with stopped train-fold reconstruction
plus streamed validation LM, not with fixed-volume same-data scoring and not
with reconstruction-heavy scalar L-BFGS.
```

Setup geometry is not a separate calibration product. Detector centre, detector
roll, rotation-axis direction, laminography tilt, and future instrument DOFs
should be active parameters in `AlignmentState`, selected through the same
alignment pipeline as per-view pose. What changes by stage is the objective and
optimizer contract, not the ownership of state.

The final working architecture from the April 2026 attempts is:

```text
tomojax-align / evidence generator
  -> align_multires
      -> AlignmentState
          -> setup geometry state
          -> per-view pose state
      -> ActiveParameterView + DofSpec registry
          -> native units
          -> whitening/scales
          -> bounds
          -> active/frozen masks
          -> gauge policy
      -> pure JAX geometry applier
      -> configured alignment loss adapter, normally l2_otsu
      -> setup discovery:
          -> build deterministic train/validation folds
          -> reconstruct train-fold volumes with stopped setup sensitivity
          -> stream held-out validation residual/JVP chunks
          -> accumulate small LM normal equations in active coordinates
          -> accept/reject a setup update
      -> pose / polish:
          -> fixed-volume residual objectives where appropriate
      -> shared diagnostics, manifests, panels, and checkpoints
```

This doc intentionally supersedes the earlier guidance that product setup
discovery should differentiate validation loss through the reconstruction layer.
That full bilevel hypergradient remains a useful research/reference idea, but it
was the wrong production shape for the current TomoJAX solver.

The session history matters because several plausible approaches failed for
different reasons:

- The first geometry-block path reused multires orchestration but carried a
  custom setup loss and geometry-block-specific solver logic.
- Fixed-volume setup scoring looked attractive because it was close to the
  existing pose objective, but it can become self-consistent under wrong setup
  geometry.
- Reconstruction-heavy bilevel cross-validation had the right held-out evidence
  principle, but put train-fold reconstruction inside scalar L-BFGS objective
  and line-search evaluation.
- Chunking and finite-difference patches reduced individual memory blow-ups but
  did not fix the optimizer shape.
- Validation-LM with stopped train-fold reconstruction finally matched the
  memory and residual/JVP discipline that made the main per-view alignment path
  practical.

## What We Tried

### 1. Geometry blocks inside `align_multires`

The first useful direction was to keep geometry calibration inside the alignment
workflow instead of adding a standalone calibration product. This produced real
infrastructure:

- setup geometry state and metadata,
- active setup-geometry DOFs,
- executable setup-geometry stages inside `align_multires`,
- before/after demo artifacts,
- per-case manifests,
- acquisition-span metadata,
- geometry diagnostics.

This was directionally right. The demos became more honest after two important
fixes:

- `theta_span_deg` was added explicitly, so `geometry_type="parallel"` no longer
  secretly meant every parallel-labeled case was a well-conditioned 180-degree
  acquisition.
- Arbitrary-axis pitch/yaw stress examples were moved to 360-degree acquisition
  when they were intended to demonstrate arbitrary-axis calibration rather than
  ordinary 180-degree parallel CT.

The remaining flaw was deeper: the geometry-block path still behaved like a
small private solver family. It had a custom setup loss, local diagnostics, and
solver-specific concepts instead of using the same loss/objective machinery as
pose alignment.

### 2. Fixed-volume setup scoring

The next tempting step was to score setup geometry by projecting a fixed
reconstructed volume under candidate setup states. That is natural for per-view
pose alignment, where a fixed volume and residual pose update have a useful
local residual contract.

For setup geometry discovery, especially detector centre/COR-like offsets, it
failed conceptually. A volume reconstructed under the wrong setup can absorb the
error. "Absorb" means the reconstruction itself deforms, shifts, blurs, or
aliases so that the wrong setup geometry and wrong volume explain the measured
projections together. The objective then rewards self-consistency of the wrong
pair:

```text
wrong setup geometry + wrong reconstructed volume -> low same-data residual
```

The fixed-volume objective may then prefer nominal or weakly moved setup even
when the hidden synthetic perturbation is large. This is not just an iteration
count bug. It is an identifiability problem for same-data setup discovery.

Fixed-volume objectives still matter. They are suitable for pose alignment and
can be useful as local polish after setup has reached a plausible basin. They
are not the default setup discovery objective.

### 3. Custom setup loss

One of the biggest avoidable mistakes was introducing a private normalized
projection L2-like setup loss while the repo already had the alignment loss
system and empirical evidence that `l2_otsu` worked best.

That created three problems:

- Setup geometry calibration no longer matched per-view pose alignment
  semantics.
- Evidence artifacts did not clearly say whether they were using `l2_otsu` or a
  private loss.
- A scalar custom loss pushed the implementation toward generic scalar
  optimizers instead of the residual/JVP LM/GN shape used by the main alignment
  path.

The product setup path should use the existing `LossAdapter`, normally
`l2_otsu`. If conditioning needs improvement, fix weighting, whitening,
damping, or residual normalization around that adapter. Do not introduce a new
product loss just because setup geometry is being optimized.

### 4. Reconstruction-heavy bilevel CV with L-BFGS

The next architecture looked mathematically cleaner:

```text
for each candidate setup theta:
  for each fold:
    reconstruct x_train(theta) from train views
    score held-out validation projections with l2_otsu
  optimize scalar cross-validation loss over active setup DOFs
```

The principle was right: score geometry on data not used to reconstruct the
volume. The implementation shape was wrong for production.

Putting train-fold reconstruction inside scalar L-BFGS value/gradient/line
search created exactly the memory behavior we did not want. Even after multiple
patches, a single 64^3 COR scenario could approach multi-GB GPU memory, while
the main per-view pose alignment could handle much larger 128^3 cases under
roughly 1-2 GB.

The attempted patches each taught something:

- Chunked/scanned validation projections fixed all-view prediction stacks but
  not reverse-mode through reconstruction.
- Passing `views_per_batch` through the reconstruction core helped, but the
  objective still built too much work inside one scalar optimization contract.
- Explicit FISTA backprojection gradients avoided one reverse-mode tape, but
  diagnostics and validation scoring still expanded the hot path.
- Fold-safe value/gradient accumulation helped, but L-BFGS still repeatedly
  re-entered reconstruction through line search.
- Finite-difference active gradients removed another AD tape but preserved the
  bad product shape.

The conclusion is clear: product setup geometry should not use full unrolled
reconstruction hypergradients inside scalar L-BFGS. That path can stay as tiny
research/reference infrastructure, not as the default solver.

### 5. Validation-LM with stopped train-fold reconstruction

The winning shape kept the good part of bilevel CV, held-out validation, while
removing the bad product shape:

```text
for a level and setup outer iteration:
  reconstruct train-fold volumes at the current setup state
  stop gradients through those reconstructions
  for each validation fold and view chunk:
    compute l2_otsu validation residuals
    compute JVPs with respect to low-dimensional active setup DOFs
    accumulate J^T J, J^T r, and validation loss
  solve a damped LM step in whitened active coordinates
  score candidate setup states against cached fold volumes
  accept the best improving candidate
```

This is not a retreat from the general geometry solver goal. It is the practical
expression of it:

- unified state,
- unified active DOF registry,
- unified loss semantics,
- held-out setup evidence,
- residual/JVP optimizer shape,
- small active-state normal equations,
- streamed validation memory behavior.

The product path provenance should show:

```text
objective_kind=bilevel_cv
optimizer_kind=validation_lm
outer_loss_kind=l2_otsu
recon_sensitivity=stopped
fold_eval_mode=stopped_train_recon_validation_lm
active_gradient_mode=validation_residual_jvp
views_per_batch=<configured value>
```

The word `bilevel_cv` remains acceptable here because the objective still uses
train-fold reconstructions and held-out validation folds. The important
qualification is `recon_sensitivity=stopped`: the production optimizer does not
differentiate through the train reconstruction.

## Guidance

### Use active DOFs, not solver families

The public interface should choose active parameters and schedules, not
geometry-specific solver islands.

Examples:

- `--schedule cor` activates `det_u_px` with appropriate setup validation.
- `--schedule detector_roll` activates `detector_roll_deg`.
- `--schedule axis_direction` activates `axis_rot_x_deg` and/or
  `axis_rot_y_deg`.
- `--schedule lamino_tilt` activates the setup DOF that represents the intended
  laminography tilt perturbation.
- `--schedule pose_only` keeps setup frozen and optimizes per-view pose.

Do not reintroduce flags whose meaning is "use the old geometry calibration
solver". The schedule may be domain-specific; the solver machinery should not
be.

### Keep setup and pose in one state model

Core ownership:

- `src/tomojax/align/state.py` owns `AlignmentState`, `SetupGeometryState`, and
  `PoseState`.
- `src/tomojax/align/dof_specs.py` owns `DofSpec`, active DOF registration,
  native units, whitening, bounds, and active masks.
- `src/tomojax/align/geometry_applier.py` applies setup and pose state to
  geometry arrays without Python geometry mutation in the objective hot path.
- `src/tomojax/align/fold_recon.py` owns stopped train-fold reconstruction.
- `src/tomojax/align/validation_residuals.py` owns streamed validation residual
  and normal-equation accumulation.
- `src/tomojax/align/optimizers.py` owns validation-LM and other active-state
  optimizers.
- `src/tomojax/align/pipeline.py` wires setup stages into `align_multires`.
- `src/tomojax/align/schedules.py` owns preset active/frozen DOF sets.

Keep geometry-block helpers as metadata/state helpers, not as private solver
loops. Keep projection-domain COR estimates as initializers or diagnostics, not
as the calibration result.

### Store setup DOFs in native units

Detector centre state is always stored in native/full-resolution detector
pixels:

```text
state.setup.det_u_px = native detector pixels
level detector shift = det_u_px / level_factor
```

The multiresolution conversion belongs in the geometry applier, not in the
state. This prevents level-pixel estimates from leaking into full-resolution
metadata.

The optimizer should operate in whitened active coordinates:

```text
z = (physical_value - reference_value) / scale
physical_value = reference_value + scale * z
```

Diagnostics should report both whitened values and native-unit values. Mixed
DOFs such as pixels, degrees, translations, and future spline coefficients are
otherwise hard to compare.

### Use `l2_otsu` through the loss adapter

Setup discovery should use the same loss machinery as alignment, normally
`l2_otsu`.

The validation-LM residual can be built from adapter-derived weights/masks, but
the metadata must still report the real outer loss:

```text
outer_loss_source=AlignmentLossSpec
outer_loss_kind=l2_otsu
```

Do not create a "quick" setup loss that bypasses Otsu masking just to simplify
the solver. That is exactly how the branch drifted away from the known-good
per-view alignment path.

### Keep reconstruction out of optimizer line search

The setup optimizer should not call train-fold reconstruction repeatedly inside
L-BFGS-style line search. Reconstruct fold volumes at the current setup state,
then use those cached fold volumes while building validation residual normals
and scoring LM candidate steps.

This is the key memory rule:

```text
expensive reconstruction happens once per fold/current setup state
cheap active setup candidates are scored against cached fold volumes
```

If an implementation reconstructs for every candidate step, every line-search
probe, or every central finite-difference perturbation, it is probably
recreating the failed architecture.

### Stream validation residuals

The active setup space is small. The solver should accumulate small matrices,
not materialize fold-wide residual/Jacobian stacks.

Correct memory shape:

```text
normal_matrix: [n_active, n_active]
gradient:      [n_active]
loss:          scalar
```

Incorrect memory shape:

```text
predictions: [n_folds, n_validation_views, detector_v, detector_u]
jacobian:    [n_folds, n_validation_views, detector_v, detector_u, n_active]
```

`views_per_batch` must be honored in validation scoring and reconstruction
helpers. Chunking is not an optional speed tweak; it is part of the solver
contract.

### Defer reconstruction-differentiated bilevel hypergradients

Unrolled or implicit differentiation through reconstruction may become useful
for future research-grade objectives, but it is not part of the production
setup solver.

Use this split:

```text
product setup discovery:
  stopped train-fold reconstruction + streamed validation LM

reference/research:
  differentiable reconstruction layer + bilevel hypergradient on tiny cases

pose/local polish:
  fixed-volume residual objective + GN/LM where the residual contract is valid
```

Do not delete the possibility of differentiable reconstruction. Do not make it
the default setup path.

### Treat gauge coupling as a first-class constraint

Some active sets are ambiguous:

- `det_u_px` plus mean per-view `dx`,
- detector roll plus mean per-view in-plane rotation,
- scan angle offset plus mean pose angle,
- global translations without an anchor.

Expert mode should not mean "skip validation". It should install a named gauge
policy such as:

```text
reject
anchor_mean
prior_required
diagnose_only
```

Default setup schedules should reject or stage gauge-coupled DOFs rather than
optimizing all of them together.

## Why This Matters

### The volume can absorb setup error

The phrase "the volume absorbs the error" was central to the COR failure. It is
not vague. It means that when a reconstruction is produced using wrong setup
geometry, the reconstructed object can encode that wrong geometry as shifted
features, blur, duplicated edges, streaks, or warped intensity structure. If the
same wrong setup is then used to project that same wrong volume, the projection
residual can be low.

For setup discovery, the objective must avoid rewarding:

```text
wrong setup -> wrong volume -> good same-data reprojection
```

Train/validation folds help because the held-out projections were not used to
make the fold volume. They are not magic, but they are a much better evidence
source than same-data self-consistency.

### Scalar L-BFGS was the wrong optimizer shape

L-BFGS optimizes scalar objectives. That pushed the branch toward:

```text
value(theta) =
  reconstruct folds under theta
  score validation loss
```

Then every gradient, line-search candidate, diagnostic score, or finite
difference probe risked rebuilding the expensive reconstruction path.

The main alignment path succeeds because it has a residual/JVP structure. The
setup path needed the same class of contract:

```text
r(theta) = held-out projection residuals
J(theta) = residual JVPs with respect to active setup DOFs
solve (J^T J + lambda I) delta = -J^T r
```

The final validation-LM implementation restored that shape.

### Memory failures were architectural, not incidental

The memory failures were not just preallocation noise. Runs such as
`runs/alignment-memoryfix-64-bilevel-20260426-203032/` and
`runs/alignment-memoryfix3-64-bilevel-20260426-210315/` showed the
reconstruction-heavy scalar path could create GPU HLO/input-output footprints
around the multi-GB range for a single 64^3 scenario.

The successful recon12 validation-LM run finished the 32^3, 64^3, and 128^3
checks with a peak GPU memory around 1.25 GiB. That is the right order of
magnitude relative to main's per-view alignment.

### Visual evidence can be misleading if final reconstruction is under-iterated

One validation-LM run produced plausible setup estimates and improved calibrated
FBP, but the aligned TV panel looked worse because the display/final TV
reconstruction used only three iterations. That was an evidence-budget failure,
not proof that the setup estimate was bad.

For docs and acceptance runs, final aligned TV reconstruction must be given
enough iterations to judge geometry quality. In the successful run,
`recon_iters=12` made roll and laminography results visually and numerically
meaningful.

### Provenance is part of correctness

During these attempts, stale or vague metadata repeatedly made results harder to
interpret. A run artifact is only useful evidence if it records:

- active setup DOFs,
- active pose DOFs,
- hidden synthetic perturbations,
- acquisition span,
- objective kind,
- optimizer kind,
- loss kind,
- fold policy,
- reconstruction sensitivity,
- active gradient mode,
- chunking/batching,
- setup estimates in native units,
- diagnostics/status.

If a future result cannot answer "what objective did this actually optimize?",
it should not be used as evidence.

## When To Apply

Apply this pattern when adding or debugging any global setup/instrument geometry
DOF in TomoJAX:

- detector/ray-grid centre: `det_u_px`, `det_v_px`,
- detector-plane roll: `detector_roll_deg`,
- rotation-axis pitch/yaw: `axis_rot_x_deg`, `axis_rot_y_deg`,
- laminography tilt represented through setup axis/scan geometry,
- source/detector pose families,
- future smooth global setup drift parameters.

Apply it when a demo or script starts branching by geometry-specific solver
concept. The branch should choose active DOFs, schedules, objectives, and
provenance. It should not instantiate a private calibration pipeline.

Apply it when a setup case appears to converge but does not visually correct the
data. Check these in order:

1. Is the run actually using `outer_loss_kind=l2_otsu`?
2. Is setup discovery held-out validation, not fixed-volume same-data scoring?
3. Is `recon_sensitivity=stopped` for the product validation-LM path?
4. Is `optimizer_kind=validation_lm`, not scalar L-BFGS?
5. Are setup DOFs in native units with correct multires scaling?
6. Is the acquisition span explicit and physically valid for the DOF?
7. Did the final aligned TV reconstruction use enough iterations?
8. Are gauge-coupled DOFs rejected, anchored, or staged?
9. Are validation residuals streamed with `views_per_batch` honored?
10. Do manifests and summaries record the above facts?

Apply it when GPU memory looks wrong. If a 64^3 setup smoke test uses multiple
GB while main aligns much larger volumes in about the same memory envelope, the
implementation is probably reconstructing inside optimizer probes or
materializing fold-wide residual/Jacobian stacks.

## Examples

### Correct current setup path

At a high level, the product setup path should look like this:

```text
state = AlignmentState(...)
active = ActiveParameterView.from_state(state, active_setup_dofs)
loss = LossAdapter.from_spec(AlignmentLossSpec(kind="l2_otsu"), validation_data)

for level in multires_levels:
  for outer in setup_outer_iters:
    fold_volumes = []

    for fold in folds:
      x_train = reconstruct_train_fold_nograd(
        setup_state=state.setup,
        fold=fold,
        recon_config=FoldReconstructionConfig(...),
      )
      fold_volumes.append(x_train)

    normals = accumulate_validation_normals(
      state=state,
      active=active,
      fold_volumes=fold_volumes,
      loss_adapter=loss,
      views_per_batch=config.views_per_batch,
    )

    step = solve_lm(normals.jtj, normals.jtr, damping=config.damping)
    state = accept_best_validation_candidate(state, active, step, fold_volumes)
```

The real implementation lives across:

- `src/tomojax/align/pipeline.py`,
- `src/tomojax/align/fold_recon.py`,
- `src/tomojax/align/validation_residuals.py`,
- `src/tomojax/align/optimizers.py`,
- `src/tomojax/align/dof_specs.py`,
- `src/tomojax/align/geometry_applier.py`.

### COR setup discovery

COR-like detector-u setup should be scheduled as setup discovery, not pose
alignment and not candidate enumeration:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --schedule cor \
  --loss l2_otsu \
  --out runs/cor_aligned.nxs
```

Expected provenance:

```text
objective_kind=bilevel_cv
optimizer_kind=validation_lm
outer_loss_kind=l2_otsu
recon_sensitivity=stopped
fold_eval_mode=stopped_train_recon_validation_lm
active_gradient_mode=validation_residual_jvp
active_setup_dofs=[det_u_px]
```

### Detector roll

Detector roll should be another active setup DOF, not a different solver:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --schedule detector_roll \
  --loss l2_otsu \
  --out runs/detector_roll_aligned.nxs
```

The key evidence is not only estimated angle. It is also:

- held-out validation loss decreased,
- calibrated FBP improved,
- aligned TV improved with enough final reconstruction iterations,
- provenance confirms validation-LM and `l2_otsu`.

### Pose-only remains fixed-volume

Pose alignment does not need to become setup validation-LM by default:

```bash
tomojax-align --data data/scan.nxs \
  --levels 4 2 1 \
  --schedule pose_only \
  --loss l2_otsu \
  --out runs/pose_aligned.nxs
```

The lesson is not that fixed-volume is bad. It is wrong as the default setup
discovery objective because setup geometry can be absorbed by the reconstructed
volume. It remains useful for pose and local polish where the fixed-volume
residual contract is identifying.

## Verification Evidence

The decisive successful run was:

```text
runs/alignment-validationlm-setup-recon12-20260426-214338/
```

It used the validation-LM setup path after the reconstruction-iteration fix.
Focused tests passed:

```text
61 passed in 97.83s
```

Peak GPU memory was about:

```text
1252 MiB
```

The run completed all intended scenarios without OOM or process failure:

| Scenario | Hidden setup | Estimate | Naive NMSE | Aligned TV NMSE | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `cor32` | `det_u_px=-4.0` | `-1.4346` | `1.2550` | `0.7011` | converged |
| `cor64` | `det_u_px=-4.0` | `-3.3621` | `1.1222` | `0.1142` | converged |
| `roll64` | `roll=2.5 deg` | `2.3537 deg` | `0.1479` | `0.0523` | converged |
| `lamino64` | `axis/tilt=4.4 deg` | `4.2750 deg` | `0.2161` | `0.0598` | underconverged |
| `cor128` | `det_u_px=-4.0` | `-3.8160` | `0.7157` | `0.0437` | converged |
| `roll128` | `roll=2.5 deg` | `2.3626 deg` | `0.1755` | `0.0441` | converged |
| `lamino128` | `axis/tilt=4.4 deg` | `4.2709 deg` | `0.2202` | `0.0555` | underconverged |

The lamino cases being `underconverged` is not a failure in this evidence. It
means the diagnostic still saw useful descent or nontrivial step signal at the
iteration limit. The estimates and aligned TV quality still improved strongly.

The important comparison with failed runs is memory and objective shape:

- Failed memory-fix attempts were centered around reconstruction-heavy scalar
  bilevel/L-BFGS and could hit multi-GB behavior even on 64^3.
- The validation-LM recon12 run completed 32^3, 64^3, and 128^3 scenarios with
  about 1.25 GiB peak GPU memory and all aligned-TV outputs better than naive
  FBP.

## What Not To Repeat

Do not add a setup-geometry solver outside `align_multires` unless the goal is a
clearly marked research experiment.

Do not introduce private product losses for setup geometry when the repo already
has `AlignmentLossSpec` and `l2_otsu`.

Do not use same-data fixed-volume scoring as the setup discovery default.

Do not put train-fold reconstruction inside scalar L-BFGS value/gradient/line
search for product setup alignment.

Do not claim an axis-direction example is an ordinary 180-degree parallel CT
case when the hidden perturbation requires full-rotation arbitrary-axis
conditioning.

Do not judge aligned-TV quality from a three-iteration final reconstruction in a
docs/evidence run.

Do not report "converged" without objective provenance, active DOFs, acquisition
span, loss kind, and reconstruction sensitivity.

Do not accept gauge-coupled active sets just because the optimizer accepts a
vector. Reject, anchor, stage, or require a prior.

Do not treat chunking as optional in setup validation. `views_per_batch` is part
of the memory contract.

## Related

- `docs/brainstorms/geometry-calibration-solver-requirements.md`
- Historical 2026-04-26 alignment plans removed from `docs/plans/`; see Git
  history if their step-by-step implementation detail is needed.
- `src/tomojax/align/model/state.py`
- `src/tomojax/align/model/dof_specs.py`
- `src/tomojax/align/geometry/geometry_applier.py`
- `src/tomojax/align/objectives/fold_recon.py`
- `src/tomojax/align/objectives/validation_residuals.py`
- `src/tomojax/align/optimizers.py`
- `src/tomojax/align/pipeline.py`
- `src/tomojax/align/model/schedules.py`
- `scripts/generate_alignment_before_after_128.py`
- `tests/test_alignment_state.py`
- `tests/test_geometry_applier.py`
- `tests/test_alignment_objectives.py`
- `tests/test_align_optimizers.py`
- `tests/test_bilevel_setup_alignment.py`
- `tests/test_geometry_block_taxonomy_generator.py`
- `runs/alignment-memoryfix-64-bilevel-20260426-203032/`
- `runs/alignment-memoryfix3-64-bilevel-20260426-210315/`
- `runs/alignment-validationlm-setup-20260426-211635/`
- `runs/alignment-validationlm-setup-recon12-20260426-214338/`
