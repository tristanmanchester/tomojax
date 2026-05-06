# 04 — Phased Implementation Plan

This plan is written for a coding agent. Each phase has deliverables, acceptance criteria, and outputs. The order is intentionally incremental: prove the reference implementation, then optimise.

## Phase 0 — Freeze baseline and benchmark contract

### Goal

Make current TomoJAX measurable before redesign changes land.

### Deliverables

```text
benchmark runner that can execute current TomoJAX
artifact parser for current run outputs
fixed synthetic dataset generation seeds
metric definitions
baseline report template
```

### Tasks

1. Add a `tomojax benchmark` command or standalone script that can run:
   - current TomoJAX staged/lightning path;
   - current COR-only or setup-only comparator;
   - future reimagined path.

2. Define common metrics:
   - final reconstruction RMSE/NMSE against synthetic truth;
   - final robust projection residual;
   - time to verified geometry;
   - final wall time;
   - geometry recovery after canonicalisation;
   - pose recovery after canonicalisation;
   - backend hit/fallback rate;
   - failure classification.

3. Generate the five synthetic `128^3` datasets in `05_synthetic_128_benchmark_suite.md`.

### Acceptance criteria

```text
current TomoJAX can be benchmarked on all 5 synthetic datasets
run artifacts are captured consistently
metrics can be computed without manual inspection
```

### Outputs

```text
benchmark_baseline_current.json
benchmark_baseline_current.md
synthetic_dataset_manifests/
```

## Phase 1 — Geometry graph, parameters, and gauges

### Goal

Create the internal representation for setup geometry, per-view pose, and gauges.

### Deliverables

```text
GeometryState
GeometryGraph
SetupParameters
PoseParameters
GaugePolicy
canonicalisation reports
unit/scaling conventions
```

### Tasks

1. Implement parameter dataclasses:
   - name;
   - value;
   - unit;
   - scale;
   - active/frozen status;
   - prior;
   - trust radius;
   - gauge group.

2. Implement setup block:
   - `det_u_px`;
   - optional `det_v_px`;
   - `detector_roll_rad`;
   - `axis_rot_x_rad`;
   - `axis_rot_y_rad`;
   - `theta_offset_rad`;
   - optional `theta_scale`.

3. Implement per-view 5-DOF pose:
   - `alpha_rad`;
   - `beta_rad`;
   - `phi_residual_rad`;
   - `dx_px`;
   - `dz_px`.

4. Implement gauge canonicalisation:
   - `mean(dx) -> det_u`;
   - `mean(phi_residual) -> theta_offset`;
   - `mean(dz) -> det_v` if active;
   - report all transfers.

5. Implement geometry serialisation:
   - `geometry_initial.json`;
   - `geometry_final.json`;
   - `pose_params.csv`;
   - `pose_decomposition.csv`.

### Acceptance criteria

```text
canonicalisation preserves realised forward geometry in exact-gauge tests
canonicalisation zero-centres residual pose components
geometry state round-trips through JSON/CSV
unit conversions are tested
```

### Outputs

```text
geometry_state.py
gauges.py
test_gauge_canonicalisation.py
```

## Phase 2 — JAX reference forward model and residual loss

### Goal

Build the correctness-first differentiable forward path.

### Deliverables

```text
JAX reference projector for parallel tomography
JAX reference projector for parallel laminography
masked residual computation
pseudo-Huber loss
low-pass/band-pass residual filters
finite-difference checks
```

### Tasks

1. Implement forward projection for a `128^3` volume and modest projection count.
2. Support detector shift, detector roll, axis rotation, theta offset, and per-view 5-DOF pose.
3. Implement valid masks and robust noise/scale estimation.
4. Implement pseudo-Huber residual and IRLS weights.
5. Implement level-aware residual filtering:
   - low-pass;
   - band-pass;
   - raw.
6. Add finite-difference tests for geometry parameters.

### Acceptance criteria

```text
finite-difference gradients match autodiff/JVP within tolerance
projector handles tomography and laminography parameterisations
residual loss ignores masked invalid pixels
robust loss downweights outliers
```

### Outputs

```text
forward/projector.py
forward/residual.py
forward/filters.py
test_projector_reference.py
test_residual_loss.py
```

## Phase 3 — Reconstruction step: Huber-TV/FISTA preview

### Goal

Implement the volume step used by the alternating solver.

### Deliverables

```text
FISTA / Huber-TV FISTA solver
multi-resolution volume schedules
warm-start support
non-negativity support
trace artifacts
```

### Tasks

1. Implement FISTA using the JAX reference projector/backprojector.
2. Implement smoothed TV / Huber-TV regularisation.
3. Add non-negativity projection.
4. Add reconstruction schedules:
   - preview level 4;
   - preview level 2;
   - final level 1.
5. Emit trace rows:
   - iteration;
   - loss;
   - data loss;
   - regulariser;
   - step size/L estimate;
   - wall time;
   - backend.

### Acceptance criteria

```text
FISTA reconstructs a clean synthetic phantom with correct geometry
loss is sane / monotonic-ish where expected
warm start works
trace artifacts are written
```

### Outputs

```text
recon/fista.py
recon/huber_tv.py
fista_trace.csv
test_fista.py
```

## Phase 4 — Pose-only 5×5 LM

### Goal

Prove the all-5 per-view gradient pose solver against fixed setup and fixed volume.

### Deliverables

```text
per-view 5×5 LM update
pose-only geometry trace
trust radii
damping adaptation
bounds handling
gauge canonicalisation after accepted steps
```

### Tasks

1. Build `PoseLM`:
   - compute per-view residuals;
   - accumulate `Jᵀr`;
   - accumulate `JᵀJ`;
   - solve damped `5×5` systems.

2. Implement actual-vs-predicted decrease acceptance.
3. Implement per-DOF trust radii.
4. Implement robust IRLS weights.
5. Apply gauge canonicalisation after accepted batch update.

### Acceptance criteria

```text
recovers ±20 px dx/dz synthetic perturbations at level 4
recovers ±10° phi synthetic perturbations at level 4 where identifiable
improves projection residual on held-out synthetic views
does not produce NaNs or runaway steps
```

### Outputs

```text
align/pose_lm.py
geometry_trace_pose_only.csv
test_pose_lm_synthetic.py
```

## Phase 5 — Setup-only LM

### Goal

Replace sequential setup stages with a single gradient setup block.

### Deliverables

```text
setup-only LM
setup observability diagnostics
theta-offset parameterisation
det_u/COR gradient solve
detector roll and axis direction gradient solve
```

### Tasks

1. Build `SetupLM`:
   - active block can include `det_u`, roll, axis, theta offset, optional `det_v`.
2. Add setup priors and parameter scaling.
3. Add observability reports:
   - Hessian/normal diagonal;
   - eigenvalues;
   - correlations;
   - weak-mode labels.
4. Add theta decomposition:
   - `theta_total = theta_metadata + theta_offset + phi_residual`;
   - `mean(phi_residual)=0`.

### Acceptance criteria

```text
recovers +14.5 px detector shift without grid search
recovers detector roll and axis tilt in setup-only synthetic dataset
theta offset and phi residual are separated by gauge policy
det_v can be labelled observable/unobservable in tests
```

### Outputs

```text
align/setup_lm.py
geometry_trace_setup_only.csv
observability_report.json
test_setup_lm_synthetic.py
```

## Phase 6 — Joint setup+pose Schur LM

### Goal

Implement the core reimagined geometry engine.

### Deliverables

```text
SchurGeometryLM
joint setup+pose update
pose-only and setup-only as special cases
normal-equation diagnostics
gauge canonicalisation
```

### Tasks

1. Accumulate per-view reductions:
   - `loss_i`;
   - `Jgᵀr`;
   - `Jpᵀr`;
   - `JgᵀJg`;
   - `JpᵀJp`;
   - `JgᵀJp`.

2. Build Schur complement.
3. Solve global setup step.
4. Back-substitute per-view pose steps.
5. Add priors/damping/trust radii.
6. Add acceptance/rejection loop.
7. Add diagnostics:
   - Schur condition;
   - global eigenvalues;
   - pose block condition stats;
   - per-DOF update norms.

### Acceptance criteria

```text
matches or beats sequential setup+pose on synthetic datasets A/B/C
recovers setup and pose after canonicalisation
fewer reconstruction calls than staged current solver
does not degrade final reconstruction vs current solver
```

### Outputs

```text
align/schur_lm.py
normal_eq_summary.json
test_schur_lm.py
```

## Phase 7 — Alternating solver and continuation

### Goal

Turn reconstruction + geometry updates into one `align="auto"` pipeline.

### Deliverables

```text
AlternatingAlignmentSolver
ContinuationSchedule
early-exit logic
verification-triggered escalation
artifact writer integration
```

### Tasks

1. Implement default schedule:
   - level 4 reconstruction;
   - 2–4 geometry updates;
   - reconstruction refresh;
   - level 2 if needed;
   - final reconstruction;
   - level 1 geometry only if verification fails.

2. Implement continuation:
   - residual filters by level;
   - robust scale by level;
   - prior strength by level;
   - trust radii by level.

3. Implement early-exit:
   - residual improvement sufficient;
   - parameter update small;
   - gauges stable;
   - held-out residual passes if enabled.

4. Implement profile presets:
   - `lightning`;
   - `balanced`;
   - `reference`.

### Acceptance criteria

```text
one command produces final volume + geometry + verification report
level 1 geometry is skipped by default when coarse geometry passes
current synthetic benchmarks run end-to-end
```

### Outputs

```text
align/alternating.py
align/continuation.py
alignment_summary.csv
verification.json
```

## Phase 8 — Nuisance models and weak DOF handling

### Goal

Prevent geometry from explaining acquisition artefacts and decide weak DOFs automatically.

### Deliverables

```text
gain/offset nuisance model
background offset model
robust scale estimation
det_v observability gate
theta_scale observability gate
failure classification
```

### Tasks

1. Add per-view gain/offset:
   - either variable projection closed-form update;
   - or small nuisance LM block.
2. Add low-frequency background model behind a config flag.
3. Add `det_v_px` active-with-prior mode.
4. Add `theta_scale` active-with-prior mode.
5. Implement weak-DOF decision rules based on:
   - curvature;
   - correlation;
   - accepted step stability;
   - validation improvement.

### Acceptance criteria

```text
synthetic flat-field drift does not become fake dx/dz motion
det_v is estimated only when observable
theta scale is estimated only when identifiable
failure classifier reports nuisance_unmodelled when appropriate
```

### Outputs

```text
nuisance/gain_offset.py
nuisance/background.py
observability_report.json
failure_report.json
```

## Phase 9 — Pallas fast paths

### Goal

Accelerate the parts that matter after the reference implementation is correct.

### Deliverables

```text
Pallas projector/backprojector where eligible
Pallas residual reductions
Pallas Jᵀr/JᵀJ reductions for canonical detector-grid cases
backend provenance
JAX/Pallas equivalence tests
```

### Tasks

1. Define backend capability checks:
   - canonical detector grid;
   - calibrated detector grid;
   - supported dtype;
   - supported interpolation mode.
2. Implement Pallas projector tiles.
3. Implement Pallas backprojection tiles.
4. Implement Pallas residual reductions.
5. Implement Pallas geometry derivative reductions:
   - pose blocks first;
   - setup blocks second;
   - mixed setup/pose blocks third.
6. Add fallback reason strings.

### Acceptance criteria

```text
Pallas and JAX agree within numerical tolerances on canonical grid tests
calibrated-grid fallback is explicit and expected
alignment loop actually reaches Pallas fast paths where eligible
speedup is measured in benchmark report
```

### Outputs

```text
backends/pallas/
backend_report.json
test_backend_fallback.py
test_pallas_reductions.py
```

## Phase 10 — Experimental modules

### Goal

Add cutting-edge research capabilities after the core is stable.

### Modules

```text
neural_scout:
    tiny neural/hash-grid volume at coarse resolution

learned_autofocus:
    auxiliary quality/preconditioner model

pnp_final:
    plug-and-play / RED final reconstruction

deformable_object:
    affine/B-spline object-frame deformation

global_uncertainty:
    Schur/Laplace/randomised uncertainty around geometry
```

### Acceptance criteria

Experimental modules must:

```text
never replace physics residual as default geometry objective
emit separate artifacts
be disabled by default
pass synthetic tests designed for their target failure mode
improve time-to-verified-geometry or reconstruction quality on benchmarks
```

## Cross-phase artifact contract

Every phase that produces runs should emit:

```text
run_manifest.json
config_resolved.toml
backend_report.json
geometry_initial.json
geometry_final.json
pose_params.csv
geometry_trace.csv
fista_trace.csv
residual_metrics.csv
verification.json
artifact_index.json
```

## Suggested milestone labels

```text
M0: current baseline frozen
M1: geometry graph + gauges
M2: reference projector + robust residual
M3: FISTA preview reconstruction
M4: pose-only LM
M5: setup-only LM
M6: Schur joint LM
M7: auto alternating solver
M8: nuisance + observability
M9: Pallas derivative fast paths
M10: experimental modules
```

## Definition of “done” for first public experimental release

A first experimental release is ready when:

```text
all five synthetic128 datasets run end-to-end
new solver beats or matches current TomoJAX on final reconstruction metrics
new solver is faster on at least 3/5 datasets
all runs emit verification and backend reports
Pallas fallback is explicit and not silent
CLI is one-command simple
failure modes are classified rather than silently producing bad geometry
```
