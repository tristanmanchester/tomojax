# TomoJAX Reimagined Implementation Docs

These documents describe a greenfield redesign of TomoJAX as a fast, general, gradient-first tomography and laminography alignment/reconstruction toolbox.

The design goal is:

> Synchrotron tomography or laminography in; self-calibrated geometry, corrected per-view pose/motion, and a crisp reconstruction out.

The intended public experience is simple:

```bash
tomojax recon scan.nxs --align auto
```

The internal engine is sophisticated:

```text
geometry graph
+ differentiable projector/backprojector
+ stopped-volume alternating reconstruction
+ robust Schur LM/GN bundle adjustment
+ gauge canonicalisation
+ nuisance modelling
+ observability checks
+ Pallas fast paths where proven
+ verification-triggered escalation
```

## Document index

1. [`01_high_level_architecture.md`](01_high_level_architecture.md)  
   The full conceptual design: geometry graph, alternating solver, continuation, motion/nuisance models, verification, and what not to build.

2. [`02_loss_and_optimiser_spec.md`](02_loss_and_optimiser_spec.md)  
   The concrete loss functions and optimiser choices: pseudo-Huber projection residuals, whitening, priors, FISTA/TV reconstruction, robust Schur LM/GN for geometry.

3. [`03_repo_layout.md`](03_repo_layout.md)  
   Proposed repository structure with module responsibilities, key classes, data models, and tests.

4. [`04_phased_implementation_plan.md`](04_phased_implementation_plan.md)  
   Implementation phases, deliverables, acceptance criteria, and dependencies.

5. [`05_synthetic_128_benchmark_suite.md`](05_synthetic_128_benchmark_suite.md)  
   Five precise synthetic `128^3` benchmark datasets with geometry/motion/nuisance perturbations, metrics, and comparison protocol for current TomoJAX.

6. [`06_verification_and_artifact_contract.md`](06_verification_and_artifact_contract.md)  
   Required run artifacts, manifests, verification gates, benchmark outputs, and failure classifications.

7. [`benchmark_manifest.yaml`](benchmark_manifest.yaml)  
   Machine-readable summary of the five synthetic datasets and benchmark expectations.

## Non-goals

The default system should not expose a public menu of alignment tricks. These are not the product identity:

```text
find-cor-grid
detector-roll-grid
sinogram-symmetry alignment
entropy search
phase-correlation default mode
SPSA
CMA-ES
Nelder-Mead
coordinate-search setup calibration
```

Some may exist as developer diagnostics, but the default architecture is one gradient-first physical model.

## Recommended build target

The first complete target is:

```text
JAX reference backend
explicit voxel volume
parallel-beam tomography and laminography geometry graphs
Huber-TV/FISTA reconstruction
all-5 per-view pose LM against fixed volume
joint setup LM
Schur setup+pose LM
gauge canonicalisation
level 4 -> level 2 -> final recon continuation
synthetic benchmark suite
verification report
```

Pallas derivative fast paths should come after the JAX implementation is correct and benchmarked.
