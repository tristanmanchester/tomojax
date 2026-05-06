# 03 — Repository Layout and Deep Module Rules

This document defines the preferred repository shape for the reimagined TomoJAX.

The goal is not merely to organise files. The goal is to make TomoJAX easy for humans and coding agents to navigate while supporting a serious rewrite: differentiable geometry, robust reconstruction/alignment, synthetic benchmarks, verification, and backend provenance.

The repo should favour **deep modules**: each top-level package owns a meaningful subsystem, exposes a small public API, hides implementation details, and has tests that lock down its behaviour. Avoid shallow webs of tiny files, generic helper modules, compatibility layers, and cross-module private imports.

## Core principles

1. **Small public surface, deep implementation.**
   Each subsystem exposes a compact `api.py` and re-exports public names from `__init__.py`. Implementation can be complex, but callers should not know about it.

2. **No generic utility modules.**
   Do not create `utils.py`, `helpers.py`, `misc.py`, `common.py`, `legacy.py`, `old.py`, or `scratch.py`. If a function matters, it belongs to a domain module.

3. **Private files are private.**
   Files beginning with `_` are implementation details. They may be imported inside their own deep module only. Other modules import through the public package API.

4. **The rewrite may delete aggressively.**
   This is a reimagining, not a compatibility refactor. Old staged aligners, old CLI routes, duplicated kernels, and dead experiments should be deleted unless they are required as benchmark comparators or reference primitives.

5. **Architecture is enforced.**
   Ruff and type checking are necessary but insufficient. Use import-boundary checks to prevent deep-module boundaries from eroding.

6. **Every deep module documents itself.**
   Each top-level module under `src/tomojax/` has a `README.md` that explains its purpose, public API, allowed dependencies, forbidden dependencies, invariants, and tests.

7. **Numerical provenance is part of the API.**
   Alignment/reconstruction code must record geometry state, solver traces, verification reports, backend provenance, and artifact paths.

## Top-level repository layout

```text
repo-root/
  AGENTS.md
  README.md
  pyproject.toml
  justfile
  .importlinter
  .pre-commit-config.yaml

  docs/
    implementation_log.md
    architecture_decisions/
      ADR-0001-deep-modules.md
      ADR-0002-gradient-first-geometry.md
    tomojax-v2/
      01_high_level_architecture.md
      02_loss_and_optimiser_spec.md
      03_repo_layout.md
      04_phased_implementation_plan.md
      05_synthetic_128_benchmark_suite.md
      06_verification_and_artifact_contract.md
      07_synthetic_generator_pseudocode.md
      benchmark_manifest.yaml

  src/
    tomojax/
      __init__.py
      py.typed
      core/
      geometry/
      motion/
      nuisance/
      forward/
      recon/
      align/
      verify/
      datasets/
      backends/
      io/
      cli/

  tests/
    unit/
    integration/
    synthetic/
    numerics/
    backends/

  benchmarks/
    compare_current.py
    run_synthetic128.py
    report.py

  examples/
    minimal_recon.py
    synthetic128_alignment.py

  tools/
    check_public_imports.py
    make_benchmark_report.py
```

Important points:

- Runtime code lives in `src/tomojax/`.
- The current/old TomoJAX implementation should not be mixed into the new packages. If it is needed for comparison, isolate it under a benchmark/comparator path, not as an implicit dependency.
- Large generated data should not live in the repo. Use `.artifacts/`, external storage, or user-specified output directories.
- `docs/implementation_log.md` is a living record for coding agents: milestones attempted, commands run, failures, design deviations, and unresolved risks.

## Public package shape

```text
src/tomojax/
  __init__.py
  py.typed

  core/
    __init__.py
    api.py
    _arrays.py
    _config.py
    _errors.py
    _types.py
    README.md

  geometry/
    __init__.py
    api.py
    _frames.py
    _gauges.py
    _graph.py
    _parameters.py
    _setup_models.py
    _transforms.py
    README.md

  motion/
    __init__.py
    api.py
    _pose5.py
    _drift.py
    _harmonics.py
    _object_motion.py
    _deformation.py
    README.md

  nuisance/
    __init__.py
    api.py
    _background.py
    _gain_offset.py
    _masks.py
    _noise.py
    _stripe_bias.py
    README.md

  forward/
    __init__.py
    api.py
    _project.py
    _backproject.py
    _residuals.py
    _jacobian_reductions.py
    _ray_geometry.py
    README.md

  recon/
    __init__.py
    api.py
    _fista.py
    _pdhg.py
    _schedule.py
    _tv.py
    _volume.py
    README.md

  align/
    __init__.py
    api.py
    _alternating.py
    _continuation.py
    _observability.py
    _pose_blocks.py
    _schur_lm.py
    _trust_region.py
    README.md

  verify/
    __init__.py
    api.py
    _artifact_index.py
    _backend_report.py
    _gauge_report.py
    _residual_report.py
    _synthetic_recovery.py
    README.md

  datasets/
    __init__.py
    api.py
    _manifest.py
    _phantoms.py
    _projectors.py
    _synthetic_128.py
    README.md

  backends/
    __init__.py
    api.py
    jax_ref/
      __init__.py
      api.py
      _projector.py
      _reductions.py
      README.md
    pallas/
      __init__.py
      api.py
      _fallbacks.py
      _jtj.py
      _projector.py
      README.md
    README.md

  io/
    __init__.py
    api.py
    _beamline_profiles.py
    _hdf5.py
    _metadata.py
    _nexus.py
    _standard_dataset.py
    README.md

  cli/
    __init__.py
    main.py
    _commands.py
    _formatting.py
    README.md
```

## Deep module rule

A deep module is a directory such as `tomojax.geometry`, `tomojax.forward`, or `tomojax.align`.

Each deep module has:

```text
module/
  __init__.py       # public re-exports only
  api.py            # public API definitions
  _*.py             # private implementation files
  README.md         # module contract
```

Allowed:

```python
from tomojax.geometry import GeometryGraph, GeometryState
from tomojax.forward import project, residual_and_reductions
from tomojax.align import SchurLMSolver
```

Allowed inside `tomojax.geometry` only:

```python
from tomojax.geometry._gauges import canonicalise_gauges
from tomojax.geometry._parameters import Parameter
```

Forbidden outside `tomojax.geometry`:

```python
from tomojax.geometry._gauges import canonicalise_gauges
from tomojax.geometry._graph import GeometryGraph
```

The same rule applies to every deep module. Tests should prefer public APIs. If a white-box test needs private access, it should be local, explicit, and justified in the test name or comment.

## README.md contract for each deep module

Every deep module must include a `README.md` with this structure:

```markdown
# tomojax.<module>

## Purpose

One paragraph explaining what this module owns.

## Public API

List the names re-exported by `__init__.py` and defined in `api.py`.

## Owned concepts

List domain concepts this module owns. Other modules should not redefine these concepts.

## Allowed dependencies

List which TomoJAX modules this module may import.

## Forbidden dependencies

List modules this module must not import, plus any private-import restrictions.

## Numerical invariants

List invariants that must hold for correctness.

## Artifact/provenance responsibilities

List artifacts or trace data this module must emit, if any.

## Testing strategy

List required unit, integration, numerical, and synthetic tests.
```

Example for `tomojax.align/README.md`:

```markdown
# tomojax.align

## Purpose

Owns the alternating reconstruction/alignment loop and the robust Schur LM/GN geometry optimiser.

## Public API

- `AlignmentConfig`
- `AlignmentResult`
- `AlternatingSolver`
- `SchurLMSolver`
- `run_alignment`

## Owned concepts

- continuation schedule
- LM damping and trust-region policy
- setup + per-view pose normal equations
- observability summaries
- geometry-update traces

## Allowed dependencies

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.forward`
- `tomojax.recon`
- `tomojax.verify`

## Forbidden dependencies

- no `tomojax.cli`
- no `tomojax.io`
- no imports from private files in other deep modules
- no Pallas-specific code except through `tomojax.backends`

## Numerical invariants

- Schur solve must match dense LM on small test cases.
- Accepted LM steps must not increase robust geometry loss unless explicitly marked as exploratory.
- Gauge canonicalisation after an accepted step must preserve realised forward projections within tolerance.

## Artifact/provenance responsibilities

- geometry trace
- damping trace
- normal-equation condition summaries
- accepted/rejected step records
- observability report

## Testing strategy

- pose-only LM synthetic recovery
- setup-only LM synthetic recovery
- Schur-vs-dense comparison
- alternating solver smoke test
- gauge canonicalisation equivalence test
```

## Module responsibilities

### `tomojax.core`

Owns package-wide value types, array aliases, configuration primitives, errors, and small shared abstractions.

Public examples:

```python
from tomojax.core import Array, TomoJAXError, RuntimeConfig
```

Rules:

- May not import high-level TomoJAX modules.
- May define shared array/type aliases.
- Must not become a dumping ground for helpers.
- If something is geometry-specific, it belongs in `geometry`, not `core`.

### `tomojax.geometry`

Owns the geometry graph, coordinate frames, setup parameters, detector/stage/beam models, and gauge policies.

Public examples:

```python
from tomojax.geometry import (
    GeometryGraph,
    GeometryState,
    GaugePolicy,
    SetupGeometry,
    canonicalise_geometry,
)
```

Owned concepts:

```text
det_u_px
det_v_px
detector_roll_rad
axis_rot_x_rad
axis_rot_y_rad
theta_offset_rad
theta_scale
coordinate frames
parameter gauges
geometry state serialisation
```

Rules:

- Does not project images. Projection belongs to `forward`.
- Does not optimise geometry. Optimisation belongs to `align`.
- Does not own per-view 5-DOF motion internals. Motion belongs to `motion`.

### `tomojax.motion`

Owns per-view pose, drift, harmonics, object-frame motion, and optional deformation models.

Public examples:

```python
from tomojax.motion import Pose5Table, DriftModel, ObjectMotionModel
```

Owned concepts:

```text
alpha_i
beta_i
phi_residual_i
dx_i
dz_i
smooth drift
stage harmonic wobble
sparse jumps
object-frame drift
affine/B-spline deformation, experimental
```

Rules:

- Does not decide global detector/setup geometry.
- Does not run alignment.
- Provides typed motion states and priors used by `geometry`, `forward`, and `align`.

### `tomojax.nuisance`

Owns non-geometry acquisition effects that should not be explained by pose.

Public examples:

```python
from tomojax.nuisance import MaskBundle, GainOffsetModel, NoiseModel
```

Owned concepts:

```text
valid-pixel masks
bad frames
per-view gain/offset
background fields
noise whitening
stripe/bias models
```

Rules:

- Does not align projections.
- Does not own detector geometry.
- Provides residual weighting and nuisance parameters to `forward` and `align`.

### `tomojax.forward`

Owns differentiable projection, backprojection, residual evaluation, and Jacobian/normal-equation reductions.

Public examples:

```python
from tomojax.forward import (
    project,
    backproject,
    projection_residual,
    residual_and_reductions,
)
```

Owned concepts:

```text
forward projection
backprojection
masked robust residuals
Jᵀr reductions
JᵀJ reductions
canonical detector-grid reference path
calibrated detector-grid reference path
```

Rules:

- Does not choose optimisation steps.
- Does not update geometry.
- Does not write run-level verification reports.
- Calls backend APIs rather than importing Pallas internals directly.

### `tomojax.recon`

Owns reconstruction solvers with fixed geometry.

Public examples:

```python
from tomojax.recon import FISTAConfig, ReconstructionResult, reconstruct
```

Owned concepts:

```text
FISTA / Huber-TV FISTA
PDHG, optional
TV and smoothed TV penalties
reconstruction schedules
volume containers
preview/final reconstruction policies
```

Rules:

- Geometry is fixed during reconstruction.
- Default alignment path treats reconstructed volumes as stopped latent variables.
- Does not run geometry optimisation.

### `tomojax.align`

Owns the alternating alignment loop and geometry optimiser.

Public examples:

```python
from tomojax.align import AlignmentConfig, AlternatingSolver, SchurLMSolver
```

Owned concepts:

```text
coarse-to-fine continuation
robust LM/GN updates
Schur setup+pose solve
pose-only 5x5 blocks
setup-only solves
trust radii
LM damping
observability diagnostics
```

Rules:

- Calls `recon` for x-steps.
- Calls `forward` for residuals/reductions.
- Calls `geometry` for gauge canonicalisation.
- Must not implement projectors or reconstruction kernels inline.

### `tomojax.verify`

Owns verification reports, artifact contracts, residual summaries, backend provenance summaries, and synthetic recovery assertions.

Public examples:

```python
from tomojax.verify import VerificationReport, verify_alignment_run
```

Owned concepts:

```text
residual maps
train/holdout residual summaries
gauge reports
backend reports
synthetic perturb-and-recover results
artifact index
failure classification
```

Rules:

- Does not optimise.
- Does not reconstruct.
- Evaluates realised forward geometry and records whether the run is trustworthy.

### `tomojax.datasets`

Owns synthetic benchmark generation and dataset manifests.

Public examples:

```python
from tomojax.datasets import Synthetic128Case, generate_synthetic128
```

Owned concepts:

```text
128^3 benchmark cases
32^3 smoke cases
phantom generation
true geometry manifests
corrupted geometry manifests
expected recovery tolerances
benchmark output layout
```

Rules:

- Uses public geometry/forward APIs.
- Does not depend on the alignment implementation except in benchmark runners.
- Must be deterministic from seed.

### `tomojax.backends`

Owns backend selection, JAX reference implementations, Pallas fast paths, and fallback provenance.

Public examples:

```python
from tomojax.backends import Backend, BackendReport, get_backend
```

Owned concepts:

```text
JAX reference backend
Pallas projector/backprojector kernels
Pallas Jᵀr/JᵀJ reductions
fallback reasons
backend capability declarations
JAX/Pallas equivalence checks
```

Rules:

- `jax_ref` is the correctness oracle.
- `pallas` is an optional fast path.
- No other module imports Pallas private implementation files directly.
- Every fast path must have reference comparison tests.

### `tomojax.io`

Owns loading real datasets and normalising metadata into standard TomoJAX dataset objects.

Public examples:

```python
from tomojax.io import ProjectionDataset, load_dataset
```

Owned concepts:

```text
NeXus/HDF5 readers
TIFF stacks, if supported
beamline profiles
metadata normalisation
standard projection dataset schema
```

Rules:

- Does not solve geometry.
- Does not reconstruct.
- Produces typed input objects consumed by `geometry`, `nuisance`, and workflow code.

### `tomojax.cli`

Owns command-line entrypoints and user-facing formatting.

Public commands:

```bash
tomojax recon scan.nxs --align auto --profile lightning -o run_dir
tomojax verify run_dir
tomojax synth generate synth128_setup_global_tomo -o data_dir
tomojax benchmark synthetic128 --impl current --impl reimagined
```

Rules:

- CLI calls public APIs only.
- CLI does not import private implementation files.
- CLI does not own numerical logic.
- User-facing options should express high-level policy, not internal algorithm menus.

Allowed public strategies:

```text
--align off
--align pose
--align auto
--align max

--profile lightning
--profile balanced
--profile reference
```

Forbidden public defaults:

```text
--cor-grid
--roll-grid
--phase-correlation
--spsa
--cma-es
--nelder-mead
--entropy-search
```

Developer diagnostics may exist, but not as the identity of the package.

## Dependency direction

Preferred import direction:

```text
cli
  -> io
  -> align
  -> verify
  -> recon
  -> forward
  -> backends
  -> geometry / motion / nuisance
  -> core
```

This is not meant to imply every module imports every lower module. It means lower-level modules must not depend on higher-level orchestration.

Examples:

- `geometry` may import `core`.
- `forward` may import `geometry`, `motion`, `nuisance`, `backends`, and `core`.
- `recon` may import `forward`, `geometry`, `nuisance`, and `core`.
- `align` may import `geometry`, `motion`, `nuisance`, `forward`, `recon`, `verify`, and `core`.
- `verify` may import `geometry`, `motion`, `nuisance`, `forward`, `backends`, and `core`.
- `cli` may import public APIs from `io`, `datasets`, `align`, `recon`, and `verify`.

Forbidden:

- `geometry` importing `align`, `recon`, `forward`, or `cli`.
- `forward` importing `align` or `recon`.
- `recon` importing `align`.
- Any module importing private `_*.py` files from another deep module.

## Public API pattern

Each module should follow this pattern.

`api.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExampleConfig:
    value: float

class ExampleService:
    ...

__all__ = ["ExampleConfig", "ExampleService"]
```

`__init__.py`:

```python
from tomojax.example.api import ExampleConfig, ExampleService

__all__ = ["ExampleConfig", "ExampleService"]
```

Private implementation file:

```python
# tomojax/example/_implementation.py
from tomojax.example.api import ExampleConfig

# Implementation details here.
```

Other modules import only:

```python
from tomojax.example import ExampleConfig, ExampleService
```

## Type and data-object policy

Use typed dataclasses or small immutable value objects for configuration and geometry state. Avoid untyped dictionaries except at serialisation boundaries.

Recommended patterns:

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    scale: float
    active: bool
    trust_radius: float | None = None
```

```python
@dataclass(frozen=True)
class AlignmentConfig:
    profile: str
    max_levels: tuple[int, ...]
    robust_delta: float
```

Rules:

- Public APIs must be typed.
- Array values should use shared aliases from `tomojax.core`.
- Internal dicts are acceptable for JSON serialization, but not as primary state objects.
- Prefer immutable configs; mutable numerical state should be explicit.
- Store units in names or metadata: `_px`, `_rad`, `_deg`, not ambiguous `shift` or `angle`.

## Artifact layout for runs

A run output should look like this:

```text
run_dir/
  manifest.json
  config_resolved.toml
  geometry_initial.json
  geometry_final.json
  pose_params.csv
  pose_decomposition.csv
  alignment_trace.csv
  fista_trace.csv
  residual_metrics.csv
  verification.json
  backend_report.json
  artifact_index.json

  previews/
    level04_slices.png
    level02_slices.png
    final_slices.png
    residual_maps.png

  volumes/
    final_volume.zarr
    preview_level04.zarr        # optional
    preview_level02.zarr        # optional
```

No module should invent a second artifact schema. The schema is owned by `tomojax.verify` and used by `align`, `recon`, `backends`, and `cli`.

## Benchmark/comparator layout

The new implementation should not be tangled with old TomoJAX code. If the current implementation is needed as a comparator, isolate it.

Preferred:

```text
benchmarks/
  current_impl/
    adapter.py
    README.md
  reimagined_impl/
    adapter.py
  run_synthetic128.py
  compare_reports.py
```

Rules:

- Comparator adapters may call legacy code.
- Production `src/tomojax/` must not depend on comparator adapters.
- Legacy code kept only for comparison must be clearly labelled and excluded from the new public API.

## Testing layout

```text
tests/
  unit/
    geometry/
    motion/
    nuisance/
    forward/
    recon/
    align/
    verify/
    datasets/

  integration/
    test_minimal_recon_pipeline.py
    test_alternating_solver_smoke.py
    test_backend_provenance.py

  synthetic/
    test_synthetic_manifest.py
    test_recover_setup_global_tomo.py
    test_recover_extreme_pose.py
    test_recover_lamino_axis_roll_pose.py
    test_recover_thermal_object_drift.py
    test_recover_combined_hard_case.py

  numerics/
    test_gauge_canonicalisation_equivalence.py
    test_jacobian_finite_difference.py
    test_schur_matches_dense.py
    test_robust_loss_weights.py

  backends/
    test_jax_reference_projector.py
    test_pallas_reference_equivalence.py
```

Required numerical tests:

```text
Gauge canonicalisation preserves realised projections within tolerance.
Schur LM step matches dense LM on small synthetic cases.
JAX Jacobian reductions match finite differences.
Pose-only 5-DOF LM recovers known synthetic pose perturbations.
Setup-only LM recovers known det_u/roll/axis perturbations.
Synthetic generator is deterministic from seed.
Pallas fast paths match JAX reference where supported.
```

## Guardrail checks

The repo should expose one canonical check command:

```bash
just check
```

It should run:

```text
ruff format / check
basedpyright or pyright
import-boundary checks
public/private import checks
fast tests
```

Recommended commands:

```bash
just format
just lint
just typecheck
just imports
just test
just check
just ci
just bench-smoke
just bench-128
```

The coding agent should run `just check` after every milestone and fix failures before continuing.

## Deletion policy

Delete rather than adapt when code is:

```text
an old staged alignment path superseded by the new alignment engine
a duplicate geometry representation
a shallow wrapper over one function
a compatibility route not used by benchmarks
a helper module with no domain owner
a dead backend path without tests
a CLI flag exposing a deprecated algorithm family
```

Keep only when:

```text
it is required by a benchmark comparator,
it is the only trusted reference implementation of a needed numerical primitive,
it has tests,
and it has a clear owner module in the new architecture.
```

If kept for comparison, isolate it outside production `src/tomojax/` or behind a clearly documented adapter.

## Anti-patterns

Avoid:

```text
tomojax/utils.py
tomojax/helpers.py
tomojax/legacy/
tomojax/align/grid_search.py as a public default
tomojax/experimental2/
public imports from tomojax.forward._project
large public API files that re-export everything
multiple GeometryState classes
different modules defining their own Parameter class
backend-specific logic scattered through high-level code
CLI flags for every internal experiment
```

Prefer:

```text
deep module ownership
small public APIs
typed state objects
explicit artifact contracts
reference backend first
Pallas fast paths behind backend capability checks
benchmarked optional accelerators
verification-triggered escalation
```

## Minimal public API target

The public Python API should stay small:

```python
from tomojax import recon

result = recon(
    "scan.nxs",
    align="auto",
    profile="lightning",
    output_dir="run_dir",
)

volume = result.volume
geometry = result.geometry
pose = result.pose
verification = result.verification
```

A likely result object:

```python
@dataclass(frozen=True)
class TomoJAXResult:
    volume: object
    geometry: object
    pose: object
    verification: object
    artifacts: object
```

The public API should communicate policy, not algorithms:

```text
align="off"       no alignment
align="pose"      use provided setup, solve per-view pose
align="auto"      solve setup + pose with default verification
align="max"       allow expensive verification-triggered refinement
```

## Summary

The codebase should make one thing easy: implementing and evolving a fast, gradient-first, self-calibrating tomography/laminography reconstruction engine without accumulating dead code.

The layout enforces that by combining:

```text
deep modules
small public APIs
private implementation files
module README contracts
import-boundary checks
typed numerical state
explicit artifacts
aggressive deletion of obsolete code
synthetic benchmarks
verification-first development
```

This structure is intentionally strict. It is designed to help both humans and coding agents make large changes without turning TomoJAX into a pile of partially compatible alignment experiments.
