---
title: Use unified bilevel alignment for setup geometry
date: 2026-04-25
last_updated: 2026-04-26
category: architecture-patterns
module: TomoJAX alignment geometry calibration
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - Adding detector or instrument geometry optimization to TomoJAX alignment
  - A calibration feature starts duplicating reconstruction or multiresolution control flow
  - A solver needs the same memory, checkpoint, preview, and CLI behavior as alignment
  - Geometry calibration demos need to distinguish solver convergence from acquisition conditioning
tags: [tomojax, alignment, geometry-calibration, dry, bilevel, diagnostics, theta-span]
---

# Use unified bilevel alignment for setup geometry

## Context

The first geometry-calibration implementation had the right product instinct:
setup geometry belongs in the alignment workflow, not in a standalone calibration
product. It still kept too much solver logic in geometry-specific islands. The
detector-centre candidate search fixed one immediate COR failure, but it was not
the architecture we want long term: it solved one parameter with custom scoring
instead of extending TomoJAX's alignment objective system.

The target architecture is now:

```text
tomojax-align
  -> align_multires
      -> AlignmentState
          -> setup geometry state
          -> per-view pose state
      -> active DOF registry + whitening
      -> pure JAX geometry applier
      -> objective layer
          -> setup discovery: bilevel_cv + configured validation loss
          -> pose / polish: fixed_volume objectives
      -> generic active-state optimizer
      -> diagnostics, checkpoints, manifests, panels
```

Setup geometry and pose remain separate namespaces inside state, but they share
one product path, one loss system, one multiresolution orchestration layer, and
one evidence path.

## Guidance

Use active DOFs, not solver families, as the public interface.

- `--optimise-dofs det_u_px` means COR-like detector/ray-grid centre setup
  alignment.
- `--optimise-dofs dx,dz` means residual pose translation alignment.
- `--schedule cor` is the safe preset for detector-u centre setup discovery.
- `--schedule setup_safe` is the staged setup-plus-pose preset.
- `--optimise-dofs det_u_px,dx,dz` is gauge-coupled and must be rejected unless
  an explicit expert gauge policy anchors the ambiguity.
- `--optimise-geometry` is not part of the greenfield CLI surface.

The setup discovery objective should be `bilevel_cv`:

```text
for each fold:
  reconstruct x*(theta) from train views using the candidate setup geometry
  score validation projections with the configured AlignmentLossSpec
  differentiate validation loss through the reconstruction layer
```

For the current docs/demo profile the configured loss is `l2_otsu`. Fold-specific
loss adapters must be built from concrete validation targets before JAX tracing,
because Otsu masks are precomputed state, not traced Python/NumPy work.

Projection-domain detector-centre COM estimates can stay as initializers or
diagnostics. They must not choose the calibration result.

Fixed-volume objectives are still useful, but they are not the default setup
discovery objective. They are appropriate for pose alignment and optional local
polish. The failure mode is specific: a volume reconstructed under wrong setup
geometry can absorb that geometry error, making same-data fixed-volume scoring
self-consistent at the wrong geometry.

## Implementation Shape

Core ownership:

- `src/tomojax/align/state.py` owns `AlignmentState`, `SetupGeometryState`, and
  `PoseState`.
- `src/tomojax/align/dof_specs.py` owns the active DOF registry, unit scales,
  bounds, priors, and whitening.
- `src/tomojax/align/geometry_applier.py` applies setup and pose state to arrays
  without constructing Python geometry objects in the objective hot path.
- `src/tomojax/align/objectives.py` owns fixed-volume and bilevel-CV objective
  contracts.
- `src/tomojax/align/recon_layer.py` owns differentiable reconstruction layers.
- `src/tomojax/recon/fista_tv_core.py` owns the array-level reconstruction core.
- `src/tomojax/align/optimizers.py` owns generic active-state optimizers.
- `src/tomojax/align/diagnostics.py` owns gauge and conditioning diagnostics.
- `src/tomojax/align/geometry_blocks.py` now owns metadata/state helpers, not a
  geometry-specific solver loop.
- `src/tomojax/align/initializers.py` owns projection-domain initializers.

Detector-centre state is always stored in native/full-resolution detector
pixels. Multiresolution scaling is applied only inside the geometry applier:

```text
state.setup.det_u_px = native detector pixels
level detector shift = det_u_px / level_factor
```

This invariant prevents pyramid-level estimates from leaking into full-resolution
metadata.

## Diagnostics

Every setup solve must report enough metadata to audit the evidence:

- active setup and pose DOFs,
- objective kind,
- outer loss kind,
- fold policy,
- inner reconstruction config,
- differentiation mode,
- per-DOF movement in native units,
- whitened step and gradient norms,
- accepted/rejected optimizer status,
- gauge decision,
- conditioning warnings,
- acquisition span for axis-direction examples.

Axis-direction setup below a full-enough acquisition span should be reported as
weak or ill-conditioned rather than treated as ordinary convergence.

## Examples

COR setup discovery:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --schedule cor \
  --loss l2_otsu \
  --out out/cor_aligned.nxs
```

Explicit setup DOF:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-dofs det_u_px \
  --freeze-dofs alpha beta phi dx dz \
  --loss l2_otsu \
  --out out/detector_center_aligned.nxs
```

Pose-only:

```bash
tomojax-align --data data/scan.nxs \
  --levels 4 2 1 \
  --schedule pose_only \
  --out out/pose_aligned.nxs
```

## Tests And Evidence

Keep the evidence chain layered:

- Unit tests prove active-state packing, native-pixel scaling, fold construction,
  gauge validation, and finite setup gradients.
- Characterization tests preserve the self-consistency failure of fixed-volume
  setup scoring.
- Workflow tests prove bilevel-CV COR and other setup DOFs improve validation
  loss without candidate enumeration.
- Laptop runs produce 65^3/128^3 visual evidence using the public CLI/API path,
  not private demo solvers.

## Related

- `docs/brainstorms/geometry-calibration-solver-requirements.md`
- `docs/plans/2026-04-26-003-refactor-unified-bilevel-alignment-plan.md`
- `src/tomojax/align/pipeline.py`
- `scripts/generate_alignment_before_after_128.py`
- `tests/test_alignment_objectives.py`
- `tests/test_bilevel_recon_layer.py`
- `tests/test_align_quick.py`
