# 03 — Proposed Repository Layout

This document proposes a clean repo layout for a reimagined TomoJAX. The goal is to make the codebase easy for a coding agent to implement incrementally while preserving the architecture.

## Top-level layout

```text
tomojax/
  __init__.py
  api.py
  cli.py
  config/
  io/
  geometry/
  forward/
  recon/
  align/
  motion/
  nuisance/
  backends/
  verify/
  benchmarks/
  datasets/
  artifacts/
  utils/
tests/
docs/
examples/
scripts/
```

## `tomojax/api.py`

Public Python API. Keep this small.

```python
def recon(
    source,
    *,
    align: str = "auto",
    geometry: str | None = None,
    profile: str = "lightning",
    output_dir: str | None = None,
    config: dict | str | None = None,
) -> "TomoJAXResult":
    ...
```

Primary result:

```python
@dataclass
class TomoJAXResult:
    volume: Array | Path
    geometry: GeometryState
    pose: PoseTable
    verification: VerificationReport
    artifacts: ArtifactIndex
```

Public API should not expose low-level algorithm families.

## `tomojax/cli.py`

CLI commands:

```bash
tomojax recon scan.nxs --align auto --profile lightning -o run_dir
tomojax verify run_dir
tomojax benchmark synthetic128 --impl current --impl reimagined
tomojax synth generate synth128_setup_global_tomo -o data_dir
```

Allowed public strategy flags:

```text
--align off
--align pose
--align auto
--align max

--profile lightning
--profile balanced
--profile reference
```

Avoid public flags like:

```text
--cor-grid
--phase-correlation
--spsa
--nelder-mead
```

## `tomojax/config/`

Configuration schemas and presets.

```text
config/
  schemas.py
  defaults.py
  profiles/
    lightning.toml
    balanced.toml
    reference.toml
    debug.toml
```

Important objects:

```python
@dataclass
class TomoJAXConfig:
    geometry_model: GeometryModelConfig
    align: AlignConfig
    recon: ReconConfig
    backend: BackendConfig
    verification: VerificationConfig
    artifacts: ArtifactConfig
```

## `tomojax/io/`

Beamline input, metadata normalisation, and synthetic dataset format.

```text
io/
  nexus.py
  hdf5.py
  tiff_stack.py
  metadata.py
  beamline_profiles.py
  dataset_manifest.py
```

Responsibilities:

```text
load projections
load dark/flat if present
normalise metadata
infer detector convention where possible
emit standard ProjectionDataset object
```

Core object:

```python
@dataclass
class ProjectionDataset:
    projections: Array | LazyArray
    theta: Array
    detector_shape: tuple[int, int]
    pixel_size: tuple[float, float] | None
    masks: MaskBundle
    metadata: dict
    provenance: dict
```

## `tomojax/geometry/`

Geometry graph and parameter system.

```text
geometry/
  graph.py
  frames.py
  parameters.py
  state.py
  gauges.py
  detector.py
  beam.py
  stage.py
  parallel.py
  laminography.py
  transforms.py
  units.py
  observability.py
```

### Key classes

```python
@dataclass
class Parameter:
    name: str
    value: Array
    unit: str
    scale: float
    active: bool
    prior: Prior | None
    trust_radius: float | None
    bounds: tuple[float, float] | None
    gauge_group: str | None
```

```python
@dataclass
class GeometryState:
    setup: SetupParameters
    pose: PoseParameters
    object_motion: ObjectMotionState | None
    nuisance: NuisanceState | None
    gauges: GaugePolicy
    metadata: dict
```

```python
class GeometryGraph:
    def forward_rays(self, geometry_state, projection_indices, level) -> RayBundle:
        ...
```

### Setup parameters

```python
@dataclass
class SetupParameters:
    det_u_px: Parameter
    det_v_px: Parameter | None
    detector_roll_rad: Parameter
    axis_rot_x_rad: Parameter
    axis_rot_y_rad: Parameter
    theta_offset_rad: Parameter
    theta_scale: Parameter | None
```

### Pose parameters

```python
@dataclass
class PoseParameters:
    alpha_rad: Array  # [n_views]
    beta_rad: Array
    phi_residual_rad: Array
    dx_px: Array
    dz_px: Array
```

## `tomojax/geometry/gauges.py`

Gauge canonicalisation rules.

```python
class GaugePolicy:
    def canonicalise(self, state: GeometryState) -> tuple[GeometryState, GaugeReport]:
        ...
```

Initial rules:

```text
mean(dx) -> det_u_px
mean(phi_residual) -> theta_offset
mean(dz) -> det_v_px if det_v active
common alpha/beta -> setup axis only if explicitly enabled
```

Tests:

```text
canonicalisation should preserve predicted projections within tolerance
canonicalisation should reduce means to near zero
gauge reports should record transferred values
```

## `tomojax/forward/`

Projector, backprojector, residual, and derivative reductions.

```text
forward/
  projector.py
  backprojector.py
  residual.py
  filters.py
  reductions.py
  jacobian.py
  ray_integral.py
  interpolation.py
```

Reference functions:

```python
def project(volume, geometry_state, indices, level, backend) -> Array:
    ...

def backproject(residuals, geometry_state, indices, level, backend) -> Array:
    ...

def residual(volume, dataset, geometry_state, indices, loss_config, backend) -> ResidualBundle:
    ...

def geometry_reductions(volume, dataset, geometry_state, active_blocks, backend) -> GeometryReductions:
    ...
```

Reduction result:

```python
@dataclass
class GeometryReductions:
    loss: float
    jtr_setup: Array       # [G]
    jtj_setup: Array       # [G, G]
    jtr_pose: Array        # [N, 5]
    jtj_pose: Array        # [N, 5, 5]
    jtj_setup_pose: Array  # [N, G, 5]
    residual_stats: ResidualStats
```

## `tomojax/recon/`

Reconstruction solvers.

```text
recon/
  fista.py
  huber_tv.py
  pdhg.py
  cgls.py
  operators.py
  schedules.py
  regularizers.py
  constraints.py
```

Core interface:

```python
class ReconstructionSolver(Protocol):
    def solve(
        self,
        dataset: ProjectionDataset,
        geometry: GeometryState,
        init: Array | None,
        schedule: ReconSchedule,
        backend: Backend,
    ) -> ReconResult:
        ...
```

Default solvers:

```text
fista_tv
huber_tv_fista
cgls_preview
pdhg_tv_future
```

Recon result:

```python
@dataclass
class ReconResult:
    volume: Array
    trace: list[ReconTraceRow]
    backend_report: BackendReport
    artifacts: dict
```

## `tomojax/align/`

Alternating solver and geometry optimiser.

```text
align/
  alternating.py
  schur_lm.py
  pose_lm.py
  setup_lm.py
  continuation.py
  active_blocks.py
  priors.py
  trust_region.py
  damping.py
  line_search.py
  early_exit.py
```

### `alternating.py`

Coordinates the full alignment process.

```python
class AlternatingAlignmentSolver:
    def run(self, dataset, initial_geometry, config) -> AlignmentResult:
        ...
```

Default flow:

```text
preflight
level 4 recon
geometry LM
canonicalise
verify
level 2 if needed
final recon
emit artifacts
```

### `schur_lm.py`

Joint setup+pose LM solver.

```python
class SchurGeometryLM:
    def step(self, volume, dataset, geometry, active_blocks) -> GeometryStepResult:
        ...
```

Must support:

```text
pose-only
setup-only
joint setup+pose
optional nuisance blocks
optional object-motion blocks later
```

### `pose_lm.py`

Independent per-view 5×5 pose solver for early implementation and debugging.

### `setup_lm.py`

Small dense setup solver for early implementation and debugging.

## `tomojax/motion/`

Object-frame motion and deformation models.

```text
motion/
  rigid.py
  drift.py
  splines.py
  harmonics.py
  sparse_jumps.py
  affine.py
  deformation.py
```

Start with:

```text
rigid object-frame drift
B-spline drift coefficients
stage harmonic wobble
sparse jump model
```

Do not make deformation default.

## `tomojax/nuisance/`

Acquisition nuisance models.

```text
nuisance/
  gain_offset.py
  background.py
  masks.py
  stripes.py
  robust_scale.py
```

Start with:

```text
per-view gain/offset
background offset
mask bundle
robust residual scale estimation
```

## `tomojax/backends/`

JAX reference and Pallas fast paths.

```text
backends/
  base.py
  jax_reference/
    projector.py
    backprojector.py
    reductions.py
  pallas/
    projector.py
    backprojector.py
    residual_reductions.py
    jtj_reductions.py
    capability.py
  provenance.py
```

Backend interface:

```python
class Backend(Protocol):
    def project(...): ...
    def backproject(...): ...
    def residual_reductions(...): ...
    def geometry_reductions(...): ...
    def capabilities(self) -> BackendCapabilities: ...
```

Backend report:

```python
@dataclass
class BackendReport:
    requested: str
    actual: str
    fallback_reason: str | None
    canonical_grid: bool
    calibrated_grid: bool
    pallas_eligible: bool
    pallas_used: bool
    numerical_agreement: dict | None
```

## `tomojax/verify/`

Verification, diagnostics, and failure classification.

```text
verify/
  residual_report.py
  geometry_report.py
  gauge_report.py
  backend_report.py
  observability_report.py
  perturb_recover.py
  reconstruction_quality.py
  failure_classifier.py
```

Required reports:

```text
projection residual before/after
residual by view
residual by detector u/v
geometry update trace
gauge canonicalisation deltas
Schur eigenvalues
weak DOF labels
backend provenance
final reconstruction metrics
```

## `tomojax/benchmarks/`

Synthetic and real benchmark runners.

```text
benchmarks/
  runner.py
  metrics.py
  compare_current.py
  compare_reimagined.py
  geometry_zoo.py
  report.py
```

Benchmark interface:

```bash
tomojax benchmark synthetic128 \
  --datasets all \
  --impl current \
  --impl reimagined \
  --output benchmark_runs/
```

## `tomojax/datasets/`

Synthetic dataset generation.

```text
datasets/
  synthetic128/
    phantom.py
    projector_truth.py
    perturbations.py
    nuisance.py
    manifests.py
    generate.py
```

Command:

```bash
tomojax synth generate synth128_pose_random_extreme -o data/synth128_pose_random_extreme
```

Each generated dataset should contain:

```text
projections.zarr or projections.npy
ground_truth_volume.npy
nominal_geometry.json
true_geometry.json
true_pose.csv
true_motion.csv
nuisance_truth.json
dataset_manifest.json
preview.png
```

## `tomojax/artifacts/`

Artifact writing and schemas.

```text
artifacts/
  writer.py
  schemas.py
  index.py
  plots.py
```

Common output structure:

```text
run_dir/
  config_resolved.toml
  run_manifest.json
  artifact_index.json
  geometry_initial.json
  geometry_final.json
  pose_params.csv
  pose_decomposition.csv
  fista_trace.csv
  geometry_trace.csv
  residual_metrics.csv
  verification.json
  backend_report.json
  gauge_report.json
  observability_report.json
  final_volume.zarr
  previews/
  residual_maps/
```

## `tomojax/utils/`

Utilities:

```text
utils/
  arrays.py
  tree.py
  logging.py
  timing.py
  random.py
  units.py
  finite_difference.py
  plotting.py
```

## Tests

Suggested test layout:

```text
tests/
  test_geometry_graph.py
  test_gauge_canonicalisation.py
  test_projector_reference.py
  test_residual_loss.py
  test_pose_lm_synthetic.py
  test_setup_lm_synthetic.py
  test_schur_lm.py
  test_fista.py
  test_backend_fallback.py
  test_artifact_contract.py
  test_synthetic128_generation.py
```

## Coding-agent priority order

Start implementation here:

```text
1. geometry parameters + state + gauge policy
2. JAX reference projector/residual for small volumes
3. pseudo-Huber residual + masks + robust scale
4. FISTA preview reconstruction
5. pose-only 5×5 LM
6. setup-only LM
7. Schur setup+pose LM
8. alternating solver
9. artifact contract
10. synthetic benchmark generator
11. Pallas fast paths
```
