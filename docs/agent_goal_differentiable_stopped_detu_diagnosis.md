# Differentiable Stopped Det-U Diagnosis Plan

This document is the next agent goal for TomoJAX v2 after the two May 9 Oracle
reviews. It supersedes any broad “make the benchmark green” direction for the
rich PHANTOM94 stopped `det_u` gate.

The goal is not to add a detector-centre finder. The goal is to make the
TomoJAX differentiable optimisation path explain, diagnose, and then fix the
stopped-volume `det_u` failure.

## Hard boundary

Do not implement, call, or recommend:

- COR grid search or centre sweeps.
- Sinogram symmetry methods.
- Phase/cross-correlation.
- Vo/Fourier COR methods.
- Entropy, sharpness, autofocus, or “try many centres and reconstruct slices”.
- Projection COM/centre heuristics as a production answer.
- Bad-view exclusion, nuisance fitting, pose freedom, theta relaxation, or
  benchmark-threshold changes to make this gate look green.

Traditional COR methods may be mentioned only as things this milestone does
not use. TomoJAX’s intended identity here is:

- physical geometry graph;
- projection-domain differentiable objectives;
- FISTA/regularised reconstruction;
- stopped-volume and reduced-objective diagnostics;
- Schur/GN/LM linear algebra;
- gauge-aware parameterisation and artifact reporting;
- eventual Pallas kernels for forward projection, backprojection, JVP/VJP,
  `JTr`, `JTJ`, and reduced Hessian-vector products.

## Current evidence

Canonical failing gate:

- Phantom: rich PHANTOM94 / `random_cubes_spheres` style object.
- Geometry: supported-only parallel tomography.
- Hidden setup: `det_u_px = 14.5`.
- Theta: zero/frozen.
- Pose: frozen.
- Active setup DOF: `det_u_px` only.
- No nuisance fitting.
- Reconstruction: FISTA preview/reconstruction.
- Geometry update: joint Schur LM/GN path.
- Latest gate loss/masks: FISTA should use the valid detector mask; Schur uses
  Otsu L2 alignment mask.

Latest recorded result:

| Level | det_u RMSE px | Volume NMSE | Schur accepted | Classification |
| ---: | ---: | ---: | --- | --- |
| `32^3` | `1.607477` | `0.740780` | true | `training_loss_not_independent` |
| `64^3` | `1.687027` | `0.512972` | true | `reconstruction_absorbed_geometry` |
| `128^3` | `3.016660` | `0.504049` | true | `reconstruction_absorbed_geometry` |

Earlier fixed-truth evidence:

- `128^3` / 128-view fixed-truth Otsu L2 recovered `det_u` to
  `0.057864 px`.
- The matching stopped reconstruction run failed at `4.10057 px`.
- Same-resolution no-skip and no-FISTA-first diagnostics still failed.
- True sidecar multires/volume carry improved volume NMSE but not the final
  `det_u` gate.

Interpretation:

- The projector, `det_u` parameter sign, residual path, and Schur machinery are
  likely coherent in oracle/fixed-truth mode.
- The failing piece is the stopped reconstruction / geometry handoff: the
  preview volume can absorb setup error and become a wrong-gauge latent volume.
- The latest mask split improved honesty, but it did not prove that every
  FISTA path now uses the valid detector mask.

## Oracle synthesis

Both Oracle responses converged on the same diagnosis:

1. The one-DOF optimizer is probably not the primary issue.
2. The most likely failure is a biased fixed-stopped-volume surrogate:

   ```text
   phi_fixed(g; x_stop) = loss(A(g) x_stop, y)
   ```

   where `x_stop` was reconstructed under corrupted geometry and is not a
   faithful object in the true geometry gauge.

3. The correct reduced objective is closer to:

   ```text
   F(g) = min_x E(x, g)
        = min_x data_loss(A(g) x, y) + R(x)
   ```

4. If the inner reconstruction is exactly stationary for the same objective as
   the outer loss, the first reduced gradient is given by the envelope theorem:

   ```text
   d/dg E(x*(g), g) = partial_g E(x*(g), g)
   ```

   So “x frozen” is not automatically wrong. It becomes wrong when `x` is
   stale, not stationary, reconstructed with a different mask/loss, or when the
   reduced objective itself is flat/biased because volume degrees of freedom can
   absorb `det_u`.

5. Implicit differentiation is likely more useful first as a curvature/gauge
   diagnostic than as the first production fix:

   ```text
   h_reduced = h_fixed - E_gx E_xx^{-1} E_xg
   ```

   If the subtracted term nearly cancels fixed curvature, `det_u` is locally
   absorbable by volume changes.

6. The next milestone should produce scalar fixed-volume curves, reduced
   objective probes, finite-difference/JVP/VJP checks, and mask provenance
   artifacts before changing the solver.

## Concrete implementation risk to check first

The context-aware Oracle spotted a specific mask path that must be verified
before trusting the latest “mask split fixed” conclusion.

Main preview FISTA currently passes `projection_valid_mask` through
`_preview_reconstruction_mask(...)`:

```text
src/tomojax/align/_alternating_orchestration.py
  main preview: mask=_preview_reconstruction_mask(mask=projection_valid_mask, train_mask=train_mask)
```

But the geometry-first bootstrap and candidate refresh still appear to pass the
alignment/train mask into FISTA:

```text
_apply_initial_geometry_first_bootstrap(..., mask=train_mask, ...)
...
fista_reconstruct_reference(..., mask=mask, ...)

_candidate_refresh_volume(..., train_mask=train_mask, ...)
...
fista_reconstruct_reference(..., mask=train_mask, ...)
```

In the canonical stopped `det_u` gate, the geometry-first bootstrap is active at
the preview level:

```text
geometry_update_volume_source == "stopped_reconstruction"
geometry_update_solver == "joint_schur"
geometry_update_pose_frozen
active_setup_parameters == ("det_u_px",)
no nuisance
level.role == "preview"
level.level_factor == 4
```

Therefore the first task is not optional: add explicit reconstruction-mask versus
alignment-mask provenance for every FISTA and Schur/eval path, then fix any
remaining leakage.

## Ranked hypotheses

| Rank | Hypothesis | Current priority | Evidence | Falsifying test |
| ---: | --- | --- | --- | --- |
| 1 | Fixed stopped-volume objective bias / gauge absorption | P0 | True-volume solve passes; stopped solve fails; more preview reconstruction can reduce preview loss while worsening `det_u`; tests already show a wrong-geometry reconstruction can make detector-centre objective self-consistent at nominal. | Stopped-volume scalar `det_u` curve has a clean minimum at true `14.5 px`, and Schur still fails on that same objective. |
| 2 | Remaining mask leakage / objective mismatch | P0 | Main preview FISTA was fixed, but bootstrap/refresh paths may still pass train/alignment masks into FISTA. Prior Oracle found a real mask leak already. | Mask provenance proves every reconstruction path uses the valid detector mask, and forcing valid masks in bootstrap/refresh does not change curves or final outcome. |
| 3 | Reduced objective is flat or biased after eliminating `x` | P0/P1 | Multires and refresh attempts improved some metrics but did not recover `det_u`; higher resolution may make volume absorption easier. | Reduced/reconstructed objective curve has a correct basin near `14.5 px`. |
| 4 | FISTA gradient/objective mismatch | P1 | Prior filtered-residual gradient bug was fixed; FISTA trace logs loss at the momentum variable while returning candidate volume; some paths normalise by full array size after masking. | Production-configuration finite-difference, adjoint, and reported-loss tests pass, including masks/filters/support/TV/boundaries. |
| 5 | Masking/boundary/filter pathology | P1 | Detector-u is boundary sensitive; lowpass currently uses periodic `jnp.roll`; Otsu masks can hide geometry-exposing rays. | Valid, Otsu, intersection, and boundary-eroded curves agree; boundary finite differences are smooth and correct. |
| 6 | Schur/LM/GN conditioning or damping | P2 | Pure `det_u` is effectively scalar; fixed-truth Schur works. | Scalar finite-difference Newton succeeds on the same stopped objective while Schur fails. |
| 7 | Geometry JVP/VJP/interpolation derivative bug | P2 | Oracle pass lowers probability, but boundary-sensitive derivatives still need proof. | JVP finite differences and scalar loss derivative checks pass for true/stopped/random/boundary volumes. |
| 8 | Need extra DOFs/nuisance/pose | P3 | The gate intentionally freezes everything except `det_u`; adding freedom creates more absorption channels. | True-volume scalar curve is wrong even with the pure one-DOF model. |

## Milestone definition

Milestone name:

```text
PHANTOM94 stopped det_u-only mask-integrity plus scalar/reduced-objective diagnosis
```

This milestone passes if it gives a decisive classification. It does not need to
make the 128^3 gate green.

Required classifications:

- `biased_fixed_stopped_volume_objective`
- `biased_or_flat_reduced_objective`
- `mask_or_boundary_pathology`
- `fista_gradient_or_loss_mismatch`
- `schur_scalar_mismatch`
- `geometry_derivative_mismatch`
- `reduced_objective_ready_for_local_acceptance`

## Ordered implementation plan

### 1. Split and prove mask provenance

Refactor reconstruction call sites so they cannot accidentally receive the
alignment/train mask.

Use explicit names:

```text
reconstruction_mask = projection_valid_mask
alignment_loss_mask = Otsu/train/eval mask used by Schur or reports
```

Apply this to:

- main preview reconstruction;
- initial geometry-first bootstrap;
- bootstrap FISTA refresh;
- candidate-refresh diagnostics;
- reduced-objective probes;
- final reconstruction/evaluation paths where relevant.

Add `mask_provenance.json` artifacts with one entry per FISTA/Schur/eval call:

```json
{
  "caller": "...",
  "stage": "...",
  "level_factor": 4,
  "operation": "fista_reconstruction",
  "mask_role": "projection_valid_mask",
  "mask_shape": [128, 128, 128],
  "valid_fraction": 0.97,
  "mask_hash": "...",
  "includes_otsu": false,
  "includes_train_gating": false,
  "normalizer": "valid_count",
  "residual_filters": [...]
}
```

Test requirement:

- The canonical stopped `det_u` gate must fail the test if any FISTA call gets
  the Otsu/alignment/train mask instead of the valid detector mask.
- Schur/eval calls must state which mask they used.

### 2. Lock the reconstruction scalar and gradient contract

Create one exact scalar function for the data term FISTA optimises. Use it for:

- finite-difference tests;
- trace recomputation;
- reduced-objective reports;
- artifact summaries.

Add checks:

1. Projector/backprojector adjoint:

   ```text
   <A x, p> == <x, A^T p>
   ```

   Test valid masks, Otsu masks as diagnostics, shifted detectors, boundary
   volumes, and all multires levels.

2. Residual filter/mask adjoint:

   If scalar loss uses `B M r`, gradient must use `M^T B^T (...)` in the matching
   order. Test raw, lowpass, bandpass/DoG, and chained filters.

3. Geometry JVP finite differences:

   ```text
   partial_det_u A(g)x ~= (A(g+eps)x - A(g-eps)x) / (2 eps)
   ```

   Test true volume, stopped volume, random volume, and boundary-heavy volume.

4. Geometry VJP / scalar derivative:

   ```text
   <partial_g A(g)x, r> == d/dg 0.5 ||A(g)x - y||^2
   ```

5. FISTA volume-gradient finite differences:

   ```text
   d/dx data_loss(A(g)x, y) == implemented explicit gradient
   ```

   Include valid mask, alignment mask as diagnostic, support, TV, centre penalty,
   residual filters, and detector boundaries.

6. FISTA trace honesty:

   The current loop evaluates loss/gradient at momentum variable `y` and returns
   `candidate`. Either recompute logged loss at returned candidate or explicitly
   label trace rows as momentum-point losses and add final-volume recomputed
   losses.

7. Loss normalisation:

   The docs say masked data terms should be averaged over valid residual
   entries. Some implementation paths currently divide by full array size after
   masking. Either align implementation to `N_valid` or report both
   `loss_per_array_pixel` and `loss_per_valid_residual` until the transition is
   deliberate.

Artifacts:

- `fista_gradient_checks.json`
- `adjoint_checks.json`
- `geometry_jvp_vjp_checks.json`
- `loss_normalisation_report.json`
- `fista_trace_recomputed.csv`

### 3. Implement fixed-volume scalar `det_u` landscapes

Build a diagnostic evaluator for scalar `det_u` curves. This is not production
COR search. It is a landscape test for the differentiable objective.

Candidate range:

```text
det_u_px in [-5, 25]
step 0.25 or 0.5 px for diagnostics
dense local samples near current estimate and true 14.5 px
```

Keep theta and pose frozen. No nuisance. Do not use the scalar argmin as the
production calibration result.

Required volume sources:

- `x_true`
- neutral/zero initial volume
- zero-FISTA or average/backprojection initial volume if available
- stopped preview volumes after selected FISTA iteration counts: `0, 1, 2, 4, 8, 16`
- bootstrap-refreshed volume
- carried multires volumes at `32^3`, `64^3`, `128^3`
- `x_recon_true_geometry`: FISTA reconstruction under the true geometry, without
  using `x_true`
- final stopped production volume

Required curves:

```text
phi_true_align(u)          = loss_align(A(u) x_true, y)
phi_stop_align(u)          = loss_align(A(u) x_stop, y)
phi_stop_valid(u)          = loss_valid(A(u) x_stop, y)
phi_recon_true_align(u)    = loss_align(A(u) x_recon_true_geometry, y)
phi_refreshed_align_i(u)   = loss_align(A(u) x_refresh(candidate_i), y)
```

For each curve report:

- argmin `det_u_px`;
- loss at current, true, final, and argmin;
- finite-difference gradient at current and true;
- finite-difference curvature at current and true;
- whether the gradient points toward true geometry;
- boundary residual fraction;
- mask role and mask hash.

Interpretation:

| Curve pattern | Meaning |
| --- | --- |
| true curve min at 14.5, stopped curve min near wrong/current | fixed stopped-volume objective bias proven |
| stopped curve min at 14.5, Schur still fails | Schur/JVP/LM implementation bug |
| valid-mask curve and alignment-mask curve disagree | mask choice/pathology is central |
| true-reconstructed curve bad while true-volume curve good | reconstruction quality/regularisation, not geometry machinery |
| curves are flat | weak identifiability; inspect reduced curvature/gauge-transfer ratio |
| jagged integer-pixel curves | interpolation/mask/boundary nondifferentiability |

Artifacts:

- `detu_loss_curves.csv`
- `detu_curve_summary.json`
- `detu_loss_curves.png`
- `detu_gradient_curves.png`
- `detu_curve_inputs.json`

### 4. Compare Schur against scalar finite differences

For pure `det_u`, Schur should reduce to boring scalar least squares. Make the
one-DOF report explicit.

At current, true, stopped-curve argmin, and selected local points, record:

```text
JTr
JTJ
damping lambda
raw GN step
damped LM step
finite-difference gradient
finite-difference curvature
scalar Newton step
predicted reduction
actual fixed-volume reduction
actual refreshed/reduced reduction, if available
accept/reject reason
```

If scalar finite-difference Newton and analytic Schur disagree on the same
fixed-volume objective, fix derivative/scaling/damping. If they agree and both
go to the wrong minimum, the optimizer is exonerated.

Artifacts:

- `schur_scalar_diagnostics.json`
- `schur_scalar_diagnostics.csv`
- per-view `JTr` / `JTJ` contribution summaries

### 5. Implement reduced-objective probes

For selected local candidate geometries, reconstruct or refresh the volume under
that candidate geometry using the valid detector mask, then evaluate independent
projection losses.

This is a variable-projection diagnostic:

```text
F(u) = min_x data_loss_valid(A(u) x, y) + R(x)
```

It is not a production centre sweep if used only for diagnostics and local
trust-region acceptance.

Candidate set for diagnostics:

```text
current u
Schur proposal u + delta
backtracking proposals: u + delta, u + delta/2, u + delta/4
true u = 14.5, only because this is synthetic and clearly labelled diagnostic
small local bracket around current and true for the landscape report
```

For each candidate:

- run identical FISTA budget and initialisation policy;
- record stationarity/prox-gradient norm;
- record data loss, regularisation loss, alignment loss, valid/eval loss;
- evaluate with both alignment mask and valid mask;
- save the refreshed volume only when artifact budget permits.

Artifacts:

- `reduced_objective_probe.csv`
- `reduced_objective_summary.json`
- `reduced_objective_curves.png`
- `reduced_objective_volume_sources.json`

Decision table:

| Reduced-objective result | Next action |
| --- | --- |
| reduced/eval curve has basin near true while fixed stopped curve is biased | implement local reduced-objective LM/trust-region acceptance |
| fixed stopped derivative is useful locally but full minimum is wrong | shrink geometry steps and refresh volume after each accepted step |
| reduced/eval curve is also wrong or flat | fix inner reconstruction/gauge/regularisation; do not touch Schur |
| reduced/eval curve correct but optimizer fails | investigate Schur/LM/JVP/scaling |

### 6. Add gauge-transfer / reduced-curvature diagnostic

Measure how absorbable the `det_u` projection tangent is by a volume update.

Let:

```text
q = partial_det_u A(g) x
```

Solve:

```text
delta_x* = argmin_delta_x || W^(1/2) (A delta_x - q) ||^2
                         + lambda || L delta_x ||^2
```

Report:

```text
h_fixed = q^T W q
eta = || W^(1/2) A delta_x* || / || W^(1/2) q ||
h_reduced_approx = h_fixed - b^T H_xx^-1 b
ratio = h_reduced_approx / h_fixed
CG iterations and residual
regularisation lambda
mask/filter used
```

Interpretation:

- `eta ~ 1` or tiny `h_reduced / h_fixed`: volume can absorb the `det_u` mode.
- `eta << 1` and healthy reduced curvature: geometry should be identifiable;
  look for implementation/optimizer/mask bugs.

This diagnostic should run at least at:

- current corrupted geometry;
- true geometry;
- final failed geometry;
- `32^3`, `64^3`, and `128^3` if feasible.

### 7. Only then implement local reduced-objective acceptance

Do this only if the reduced-objective probes show the right basin.

Production-style step:

1. Schur proposes a local `delta_det_u`.
2. Refresh current and proposed geometry volumes using the valid detector mask.
3. Evaluate independent projection loss.
4. Accept only if the reduced/eval objective improves.
5. Carry the accepted refreshed volume.
6. Log predicted fixed-volume reduction and actual reduced-objective reduction
   separately.

This should replace ad hoc candidate-refresh variants, not add another policy
layer.

### 8. Optional one-DOF implicit-gradient probe

Attempt only after the above artifacts exist.

For same inner and outer objective, the envelope theorem says no first-order
`dx*/du` correction is needed at exact stationarity. Therefore use implicit
work first to test curvature/gauge absorption.

If inner reconstruction objective `E` differs from outer alignment/eval
objective `U`, a one-DOF implicit gradient probe can test:

```text
F_align(u) = U(x*(u), u)
dF_align/du = U_u - E_gx lambda
E_xx lambda = U_x
```

Implementation constraints:

- start smooth: ridge or Huberised TV; avoid hard active-set complications;
- matrix-free Hessian-vector product:

  ```text
  v -> A^T W_rec A v + R_xx v
  ```

- compare implicit gradient to finite differences of the actual refreshed
  reduced objective;
- do not promote this to production if it fails this comparison.

## Required final report

At the end of the milestone, write/update:

- `docs/benchmark_runs/<date>-phantom94-stopped-detu-variable-projection.md`
- `docs/implementation_log.md`

The report must answer:

1. Did every FISTA path use the valid detector mask?
2. Did FISTA scalar/gradient finite-difference checks pass?
3. Does the true-volume scalar `det_u` curve minimize near `14.5 px`?
4. Where does the stopped-volume scalar curve minimize?
5. Do Schur `JTr`/`JTJ` agree with finite differences?
6. Does the reduced-objective probe have the right basin?
7. Is `det_u` locally absorbable by volume updates according to the
   gauge-transfer diagnostic?
8. Which classification applies?
9. What is the smallest next algorithmic change justified by the evidence?

## Go / no-go criteria

Proceed to algorithm change only if:

- mask provenance is clean;
- FISTA gradient/objective tests pass;
- scalar fixed-volume and reduced-objective landscapes have been recorded;
- the failure is classified.

Allowed algorithm changes after diagnostics:

- local reduced-objective trust-region/acceptance;
- step-size/trust-region limits justified by scalar landscapes;
- reconstruction regularisation/continuation changes justified by reduced
  curves and gauge-transfer diagnostics;
- boundary/filter fixes proven by finite-difference/adjoint failures.

Not allowed:

- adding new DOFs;
- adding nuisance fitting;
- weak-view exclusion;
- changing success thresholds;
- unlabelled use of truth/support/centroid from the synthetic phantom;
- traditional COR/sinogram/correlation methods;
- Pallas work before the objective is correct.

## Prompt summary for future handoffs

The next agent should treat this as a diagnostic milestone. A failed `det_u`
gate with decisive scalar/reduced landscapes is a successful milestone if it
identifies the blocker. A green metric without those artifacts is not success.
