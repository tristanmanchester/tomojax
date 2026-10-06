# Numerical and performance benchmarks

The [exact-projector public alignment rerun](../docs/research/public-free-voxel-exact-2026-10-04.md) retains all 48 calls;
all six cells still fail joint acceptance.


These development scripts check physical accuracy as well as speed. They are
not installed in the `tomojax` package. The measured comparison and remaining
gaps are in [the performance report](../docs/performance.md).

The [complete 27-cell reconstruction score](../docs/research/system-matrix-2026-10-04.md)
and [diagnostic attribution](../docs/research/system-profile-2026-10-04.md) retain failed
quality gates. `public_alignment_benchmark.py` separately exercises the actual
alignment API with independent free voxels; its declared six-cell
[pilot protocol](../docs/research/public-free-voxel-pilot.md) does not replace the larger
motion/robustness goals. `voxel_truth.py` provides independently integrated
trilinear voxel-basis data for this pilot.
The [completed baseline](../docs/research/public-free-voxel-baseline-2026-10-04.md)
reports every failed call, including cold/warm times and peak process GPU memory.

The complete reconstruction comparison loads fixtures through the lightweight
`physical_case` container. External workers must not import JAX/Pallas merely
to read scan metadata or build ASTRA geometry. Each worker records its loaded
solver modules so this can be audited. Earlier cold measurements that imported
`compare_projectors` in every worker included unrelated solver startup in the
external workflows; retain those records as historical evidence, and use the
corrected isolated runs for fresh-process comparisons.

Methods that miss the fixed quality gate retain seven warm attempts at their
last tested budget. These report attempted latency and quality; their time to
an accepted result remains undefined and they cannot count as speed wins.

## Joseph first-order differentiation

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python bench/joseph_derivatives.py \
  --sizes 32 64 128 256 --views 60 --repeats 7 \
  --output bench/results/joseph-derivatives.json
```

The fixed `joseph-derivatives-v2` suite measures resident-input components:
projection, ordinary CUDA reverse AD, and fused raw least squares with both
volume and matrix-pose gradients. Geometry preparation and output
synchronization are included; imports, host transfers, reconstruction and
alignment are excluded. Each case reports compilation, first execution, at
least 200 ms of synchronized warmup for every method, and all seven warm
samples. Version 1 had no duration-based warmup and is retained separately;
its sub-millisecond timings could span different GPU clock states after
compilation. Compiler argument/output/temporary bytes are estimates,
not peak process VRAM. CUDA outputs are compared with full JAX reverse AD at
sizes 32 and 64, and a JAX directional derivative at every size. Directions
include a signed volume perturbation and tangent vectors to rigid poses. Gates
are finite outputs, relative L2 errors below `2e-4` for loss/gradient comparisons
and norm-scaled directional error below `2e-5`. These checks complement the
small independent physical matrices and finite differences in the tests.

Each completed case is written atomically; failed checks stay in the record,
and an existing output file is never replaced implicitly. Timings should run
without simultaneous GPU work. Do not interpret these component timings as a
joint alignment recovery result.

`bench/joseph_gradient_memory.py` separately samples whole-process GPU memory
with fresh serial workers and per-PID `nvidia-smi` accounting requested every
10 ms. Its default fixed probe covers both JAX reverse AD and fused CUDA at
64 cubed, plus fused CUDA at 256 cubed, all three geometries and 60 views.
Imports, preparation, compilation and repeated calls are included in the peak;
brief allocations can be missed. It checks finite outputs, while derivative
accuracy is established by the comparison above and the numerical tests.

## Reproduce the comparisons

Use Python 3.12 on Linux with a CUDA-capable NVIDIA GPU:

```bash
uv sync --locked --extra cuda12 --dev --group benchmark
uv run --no-sync python bench/compare_projectors.py \
  --sizes 32,64,128,256 --views 60 --repeats 7 \
  --geometries parallel,lamino,anisotropic --libraries tomojax,astra
uv run --no-sync python bench/reconstruction_benchmark.py \
  --sizes 32,64,128,256 --views 180 --repeats 7
uv run --no-sync python bench/adjoint_benchmark.py --sizes 16,64,128 --repeats 7
```

The optional `benchmark` dependency group installs ASTRA 2.5 and CuPy 14.2
for the GPU-filtered external FBP baseline. The comparison
also supports CERN TIGRE. Install it from its source repository with a compatible
CUDA toolkit and C++ compiler; the similarly named PyPI package is unrelated:

```bash
git clone https://github.com/CERN/TIGRE.git .artifacts/TIGRE
git -C .artifacts/TIGRE checkout 6b0951a8a88aa88a7db5b2a95181a33a95aa8b28
uv pip install --python .venv/bin/python .artifacts/TIGRE
uv run --no-sync python bench/compare_projectors.py \
  --sizes 32,64,128,256 --views 60 --repeats 7 \
  --libraries tomojax,astra,tigre
```

`--no-sync` preserves the locally built TIGRE installation. Each script accepts
`--output path.json`; default outputs go to ignored `bench/results/`. Small CPU
accuracy runs are available through `just benchmark-smoke` and run in CI.
Use `--libraries tomojax` to run without external tomography libraries.

## What is measured

`compare_projectors.py` samples two asymmetric Gaussian ellipsoids. Their exact
infinite-ray integrals provide an independent reference, including amplitude;
the script does not fit a scale factor. The ellipsoids decay near the volume
boundaries. Finite support and interpolation contribute to the reported error.

The cases cover isotropic parallel scans, 30-degree laminography, and non-cubic
anisotropic volumes with detector offsets and odd detector dimensions. All use
60 uniformly spaced views over a half turn by default. TomoJAX uses FP32 gathers,
its default integration step, `(16, 4)` detector tiles, one warp, automatic
kernel selection, and inline traversal state. These are kernel measurements;
they are not timings of a complete alignment or iterative reconstruction.

ASTRA uses `parallel3d_vec`: ray, detector origin, and detector basis vectors are
transformed from each TomoJAX pose. Volume bounds describe the same physical voxel
centres. Its native array orders are handled explicitly. TIGRE uses parallel
geometry, interpolated integration with `accuracy=1`, and Siddon integration.
The TIGRE adapter currently validates only centred isotropic parallel scans;
other cases are recorded as `unsupported_comparison`. This is an adapter limit,
not a statement about TIGRE's capabilities. The projectors use different
discretizations, so agreement with the analytic reference matters more than
elementwise agreement across libraries.

Timing records distinguish:

- `device_resident`: inputs already reside on the device. ASTRA consumes JAX
  buffers through DLPack and reuses its output allocation. Each call is explicitly
  synchronized, including ASTRA's external CUDA work.
- `host_to_host`: host arrays are uploaded, projected, and downloaded. Arrays are
  prepared in each library's native order before timing; geometry setup and array
  order conversion are excluded. TIGRE is compared only in this scope.

The first call is recorded separately as `cold_ms`, followed by two warmups and
individual synchronized samples. Cold calls can reuse compilation from previous
scopes and are not a measurement of fresh-process startup. Steady-state medians
exclude compilation, phantom generation, and geometry setup. Device timings are
not compared to host-to-host timings. `XLA_PYTHON_CLIENT_PREALLOCATE=false` leaves
memory for the external libraries. Laptop clock, thermal, and power variation
remain sources of noise; inspect the samples and rerun on your target machine.

`reconstruction_benchmark.py` reconstructs an analytic Gaussian from independent
sinograms with a detector covering the diagonal field of view. It reports both
the filter/backprojection operation and the public `fbp()` call, with inputs
already on the device. The public call includes geometry preparation and the
final angular scaling. It compares TomoJAX backends, not ASTRA/TIGRE reconstructions.

`adjoint_benchmark.py` checks the explicit backprojector against a JAX VJP using
long oblique rays, anisotropic voxels, shifted detectors, and random signed data.
It also reports the inner-product identity error. The matched discrete adjoint
used by iterative solvers is a different operator from voxel-driven FBP.

## Accuracy gates and diagnostics

All three scripts return a nonzero exit status on a failed accuracy gate.
The default relative L2 limits are 6% for the small forward-phantom matrix,
2% for FBP, and `2e-5` for the explicit adjoint versus autodiff. Pallas must also
agree with the JAX reference within `5e-5`. Thresholds can be tightened with
`--max-relative-error`. Coarse grids or too few reconstruction angles may fail
legitimately. Requested but unavailable libraries and runtime failures fail the
projector comparison rather than being counted as successful measurements.

For actual GPU regressions and memory safety:

```bash
just test-cuda
XLA_PYTHON_CLIENT_PREALLOCATE=false compute-sanitizer \
  --tool memcheck --error-exitcode=99 \
  .venv/bin/python -m pytest -q -m gpu \
  tests/test_pallas_numerics.py tests/test_projector_adjoint.py tests/test_fbp_accuracy.py
```

The existing `fista_projection_benchmark.py` exercises weighted loss/gradient and
FISTA integration variants. It is a separate stochastic diagnostic; its summary
must be inspected for failures and fallback paths.

The batched adjoint memory and timing experiment is also reproducible without
checking out an older implementation. It compares the current shared accumulator
with `vmap` of the single-view adjoint followed by reduction:

```bash
uv run --no-sync python bench/adjoint_benchmark.py --stack \
  --sizes 32,64,128 --batches 4,16 --repeats 5 \
  --output bench/results/adjoint-stack.json
```

## Public workflow measurements

```bash
uv run --no-sync python bench/workflow_benchmark.py \
  --sizes 32 64 --geometries parallel lamino anisotropic \
  --methods fista spdhg align_gn --repeats 3
uv run --no-sync python bench/workflow_benchmark.py \
  --sizes 32 --methods align_gd align_lbfgs --repeats 3
```

This runner times entire public calls with resident inputs: validation, geometry
preparation, automatic norm estimation when enabled, compilation, reconstruction,
alignment, and diagnostic conversion. It records the cold call separately from
repeated calls, all configurations, per-outer alignment statistics, finite output,
and analytic volume NRMSE. `--auto-norm` enables automatic FISTA norm estimation;
SPDHG always uses it here. `--batch`, `--views`, iteration budgets, and size lists
allow larger and irregular workloads. The default 0.6 NRMSE limit catches failed
or empty reconstructions; it is a smoke threshold, not a scientific quality target.
Use more iterations and a suitable `--max-nrmse` for a time-to-quality study.

Alignment starts with small sinusoidal pose errors against analytically generated
data. Discretization mismatch means exactly zero recovered poses are not an
accuracy requirement. These runs measure TomoJAX workflows, without claiming
algorithm-matched comparisons against ASTRA or TIGRE. The retained before/after
record includes the old anisotropic-volume failure and excludes it from speedups.

## Fixed-quality complete reconstruction comparison

The stretch targets and the frozen `gaussian-v1` acceptance thresholds are in
[the optimization goal](../docs/research/optimization-goal.md). Run:

```bash
uv run --no-sync python bench/compare_reconstructions.py \
  --sizes 64 128 256 --views 180 --repeats 7 \
  --methods tomojax_fista tomojax_cgls_jax tomojax_cgls_pallas astra_cgls astra_sirt tigre_cgls
```

Each method/case runs in an isolated process using identical host inputs. Data
are generated separately from exact Gaussian line integrals. Complete solves
include geometry, layout conversions, norm estimation where applicable,
compilation, transfers, diagnostics, and full-volume quality verification.
No amplitude fitting, positivity, or regularization is applied. FISTA uses zero
regularization weight. Discrete integration models still differ across libraries.

To continue an interrupted sweep, repeat the same command with `--resume` and
the same output path. Resumption requires identical source hash, environment,
case lists, budgets, repeat count and timeout. Completed failures are retained
and skipped just like completed successes; resumption never selects a better
retry. Only missing comparisons run, and fully completed fixtures are skipped.
An ordinary run refuses to overwrite an existing output. Result writes use
atomic replacement so an interrupted write preserves the preceding record.
If source or hardware changed, use a new output file and keep the old evidence.

Each candidate starts at zero and runs one uninterrupted solve with a fixed
budget of 1, 2, 4, 8, 16, 32, 64, 128, or 256 iterations. This matters because
[ASTRA CGLS resets its state on each `run` call](https://astra-toolbox.com/docs/algs/CGLS3D_CUDA.html).
The first accepted budget is repeated seven times. `cold_search_verified_ms`
includes process startup, fixture loading, and all smaller failed candidates.
`selected_budget_warm_verified_median_ms` measures repeated full solves at the
selected budget; it excludes offline budget selection and is not itself a cold
time-to-quality claim. The individual samples and all rejected candidates are
retained. Reaching an image-quality gate is independent of a solver's internal
stationarity test.

Peak process GPU memory is sampled by `nvidia-smi` at a requested 10 ms interval,
filtering by worker PID. It covers imports, the budget search and repeated solves,
including allocator/context overhead. Sampling may miss brief peaks; `null` means
no measurement, never zero memory. It is not a claim about the selected solve's
minimum memory requirement. Failed execution, timeouts, quality failures, and
unsupported adapter comparisons remain explicit records and give a nonzero exit.
The TIGRE adapter currently supports only the validated centered parallel scan.

The optional `tomojax_multires_cgls_pallas` method uses the fixed two-level
policy in the goal document, counts both levels within the same total iteration
budget, and includes preprocessing and interpolation in each complete solve.
That policy was developed on Gaussian phantoms; use the separate structured
datasets to check its limitations without changing the policy:

```bash
uv run --no-sync python bench/compare_reconstructions.py \
  --suite structured-v1 --sizes 64 128 256 --batch 180 --repeats 7 \
  --methods tomojax_cgls_pallas tomojax_multires_cgls_pallas astra_cgls
uv run --no-sync python bench/compare_reconstructions.py \
  --suite structured-noisy-v1 --sizes 64 128 256 --batch 180 --repeats 7 \
  --methods tomojax_cgls_pallas tomojax_multires_cgls_pallas astra_cgls
```

These suites use analytic chord lengths through five fixed ellipsoids. The noisy
version adds reproducible 1%-RMS Gaussian measurement noise. Quality is measured
against voxel-centre density, with thresholds frozen in the goal document.
Budget selection still consults the truth: it is an offline time-to-quality
experiment, not a deployable stopping rule. The doubling grid also quantizes the
crossing iteration; results cannot establish an exact first-crossing speedup.
The current serial worker ordering does not control laptop thermal drift.
Independent noise seeds, dose-dependent noise, real data, direct external FBP
baselines and interleaved comparisons remain required for broad claims.

The Pallas adjoint shared-accumulator experiment is separately reproducible:

```bash
uv run --no-sync python bench/adjoint_benchmark.py --stack --backend pallas \
  --sizes 32,64,128 --batches 4,16 --repeats 7
```

This compares the shared accumulation to a `vmap` of single-view kernels and
reports compiler temporary bytes, which are distinct from process peak memory.

## References

- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
- [Pallas grids, shapes, and partial blocks](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [ASTRA 3D geometry conventions](https://astra-toolbox.com/docs/geom3d.html)
- [ASTRA 3D projector and direct API](https://astra-toolbox.com/docs/proj3d.html)
- [CERN TIGRE source and installation](https://github.com/CERN/TIGRE)

- [JAX compilation caching and function identity](https://docs.jax.dev/en/latest/jit-compilation.html)

## Plane-sampled CGLS model

`CGLSConfig(projector_model="joseph")` selects voxel-centre plane sampling with
bilinear interpolation and a matched transpose. The CUDA transpose gathers
contributions into each voxel without atomic writes. The default remains
`projector_model="ray"`; these are different discretizations. Compare analytic
accuracy and complete reconstruction quality, not just kernel throughput:

```bash
uv run --no-sync python bench/operator_models.py --sizes 64 128 256 --repeats 9
uv run --no-sync python bench/compare_reconstructions.py --sizes 64 128 \
  --batch 180 --repeats 7 --suite structured-v1 \
  --methods tomojax_joseph_cgls_pallas tomojax_multires_joseph_cgls_pallas astra_cgls
```

The resident operator comparison uses dynamic poses and includes coefficient
preparation in every call. Complete solve comparisons retain the same frozen
budgets and acceptance gates as the ray model. Repeat with `gaussian-v1` and
`structured-noisy-v1`; failed gates remain failures.

## Independent material projections with gVXR

Install the optional simulator separately from the solver environment:

```bash
uv venv --python .venv/bin/python .artifacts/gvxr-env
uv pip install --python .artifacts/gvxr-env/bin/python gvxr==2.1.0 numpy==2.5.3
.artifacts/gvxr-env/bin/python bench/gvxr_dataset.py \
  --size 64 --views 180 --tilt 0 --output bench/results/gvxr-parallel-64.npz
.artifacts/gvxr-env/bin/python bench/gvxr_dataset.py \
  --size 64 --views 180 --tilt 30 --output bench/results/gvxr-tilted-64.npz
uv run --no-sync python bench/check_gvxr_reconstruction.py \
  bench/results/gvxr-parallel-64.npz bench/results/gvxr-tilted-64.npz \
  --iterations 32 --output bench/results/gvxr-reconstruction.json
```

Headless EGL rendering was verified on the NVIDIA test machine. gVXR is used
only by the generator. Its [official tutorials](https://github.com/TomographicImaging/gVXR-Tutorials)
cover the simulator API and additional mesh/material options.

`gvxr-materials-v1` contains three disjoint water, aluminium and PMMA cuboids,
physical detector offsets, translated rigid poses and optional 30-degree tilt.
The NPZ stores poses, material attenuation truth at 80 keV, raw expected photon
counts, normalized mono/poly projections, and a reproducible noisy poly channel.
The JSON sidecar records materials, spectrum, units, versions, hashes and checks.
Geometric lengths use mm; gVXR attenuation tables use cm^-1 and are converted to
mm^-1 before forming volume truth. Detector images use `(view, v, u)` order.

Monochromatic 80 keV data test the linear attenuation model. The synthetic
40/60/80/100/120 keV photon spectrum has fractions 0.12/0.26/0.30/0.22/0.10;
it is not a calibrated scanner spectrum. Both use 100,000 incident photons per
pixel by default. The detector is ideal photon-counting, without blur or
scatter. Noise uses NumPy PCG64 seed 128904 and independent Poisson draws from
gVXR's expected poly counts. Flat normalization uses the known noiseless incident
count; zero observed counts receive an explicit 0.5-photon floor before logging.
Noise is added by NumPy, not gVXR's internal noise generator.

Every generated projection stack must agree with independent analytic box-ray
lengths and the supplied material coefficients within `1e-4` relative L2.
This verifies units, ray geometry, orientation and spectral summation; it does
not independently verify the attenuation database. Native-resolution raster
interpolation failed this check at small tilted sizes. Rendering at five times
the detector resolution and selecting the original pixel centres reduced that
error without changing the measurement model or loosening the check. This is
centre selection, not detector-area averaging.

The reconstruction script uses the exact saved poses and runs both TomoJAX
models and ASTRA CGLS for a fixed budget. It reports physical-scale volume error
and material biases, and rejects non-finite results or solver breakdown. It has
no image-quality acceptance threshold and makes no speed claim. Poly data do
not satisfy a single-energy linear model; error against 80 keV truth is a model
mismatch diagnostic. These fixtures supplement the frozen benchmark suites and
do not replace their thresholds. Run simulation separately from timed solvers
to avoid GPU contention.


## Multi-material chip-package phantom

[`phantoms/`](phantoms) builds a laminography phantom closer to a real DIAD
sample: a 1.8 mm chip package with eight materials. A Blender script models a
glass-epoxy substrate, an internal copper ground plane with clearance rings,
copper vias and traces, a silicon die, gold bond wires, a silica-filled epoxy
mold and solder balls of tin-silver-copper, three with gas voids. Every mesh is
closed and the meshes are disjoint, apart from the die, wires and traces
nested inside the mold. gVXR traces each mesh's path length; a simulator turns
these into 25 keV measurements with xraylib attenuation and refraction
coefficients, Fresnel propagation over 50 mm, a 0.7 px Gaussian detector blur,
3× supersampled pixel integration, Poisson noise at 20,000 flat-field counts,
a 1% fixed-pattern gain and 20 noisy flats. It writes static and moving scans
(720 views at 30° tilt, 344×264 px at 8 µm, a 3.2 px centre-of-rotation offset,
about 2 px of per-view shift and 0.1° of tilt), each as exact line integrals
(`-ideal`) and as measured data (`-realistic`). The truth volume, 240×240×88 at
8 µm, comes from an exact voxeliser that integrates each voxel's z overlap
along vertical rays; its material masses match the meshes within 0.03–0.9%.

```bash
# Blender 4.5 LTS from https://download.blender.org/release/Blender4.5/,
# unpacked under .artifacts/tools/blender
uv pip install --python .artifacts/gvxr-env/bin/python xraylib==4.3.0
uv run --no-sync python bench/phantoms/chip_package.py build /tmp/chip
```

The build runs Blender headless if the meshes are missing, then the TomoJAX
planning step, the gVXR render in its own environment and the final noise and
NeXus step; it takes about 12 minutes on the laptop GPU. `summary.json`
records a projector check: TomoJAX's Joseph projection of the truth volume
differs from gVXR's line integrals by 6.7% relative L2, from the partial-volume
error of 20–25 µm copper and gold features on 8 µm voxels.

At these settings the phase-contrast fringe width, √(λz) ≈ 1.6 µm, is a fifth
of a pixel, so pixel integration and blur average the fringes away; smaller
pixels or longer distances make them visible. Scattering, harmonics, beam
drift and partial coherence are not modelled. The laminography missing cone
removes structure such as the thin ground plane parallel to the plate, so even
an inverse-crime reconstruction stays about 0.5 relative L2 from the truth;
score alignment against a reconstruction of the same data with the true
geometry instead. On the moving realistic scan, `tomojax align --mode
cor_then_pose` recovers the offset to 3.18 px (3.178 px is identifiable from
the motion), rotations to 0.073° and shifts to 0.087 px RMS in 127 s; its
volume differs from the true-geometry reconstruction by 0.048 relative L2,
against 0.82 without alignment.

## External direct reconstruction baselines

The same fixed-quality runner also supports single-pass methods:

```bash
uv run --no-sync python bench/compare_reconstructions.py --sizes 64 128 \
  --batch 180 --repeats 7 --suite structured-v1 \
  --methods tomojax_fbp_pallas astra_fbp3d_cupy astra_fbp2d tigre_fbp
```

`astra_fbp3d_cupy` combines a CuPy/cuFFT discrete Ram-Lak filter with ASTRA's
voxel-driven `accumulate_BP`. Detector area / voxel volume cancels ASTRA's native
backprojection weighting, then `pi/n_views` supplies half-turn angular
quadrature. This scale is derived from geometry and independently tested under
changes of physical length units; it is not fitted to reference data. The
adapter includes uploads, filtering, layout conversion, geometry construction,
backprojection, synchronization, download and cleanup in each complete call.
Large stacks are filtered/uploaded in bounded view batches and accumulated into
a linked GPU volume. The batch uses a 512 MiB FFT-workspace estimate, separately
implemented from TomoJAX's policy. The record reports the selected batch size;
the estimate excludes volume buffers, allocators and library workspaces.
Single-batch cases retain ASTRA's cheaper `direct_BP` call without registered
data handles; only multi-batch cases need `accumulate_BP`.
The CuPy allocator and FFT plan caches retain normal repeated-call behavior;
the memory sampler includes their process allocations.

`astra_fbp2d` runs native `FBP_CUDA` on individual axial slices. `tigre_fbp`
calls CERN TIGRE's public `fbp` with its default Ram-Lak filter. These two
adapters currently support centred isotropic parallel fixtures only. Their
orientation and physical scaling are checked using an off-centre Gaussian on
non-square axial slices. The TomoJAX Pallas FBP API also handles the shifted,
anisotropic parallel fixture, but does not expose a tilted voxel-driven FBP.
The filtered ASTRA BP can run tilted scans, where it is an approximate
initializer whose reconstruction error is still subject to the full gate.

`tomojax_fbp_host_pallas` calls the public `fbp_host` API with a fixed policy of
16 output slices and 32 filtering views per batch. Input and output stay in
host memory; transfers, slab assembly and output writes are included in every
call. The same policy applies to all sizes, suites and supported geometries.
The final partial slab uses the same compiled shape. Tilted geometry is recorded
as unsupported, and no case-specific slab tuning is performed in these records.

`tomojax_fourier_cupy` calls the opt-in public `fourier_reconstruct` API with
16 output slices per batch for every case. It uses six-point Kaiser–Bessel
radial interpolation, linear angular interpolation and a detector-Nyquist disk
cutoff. Host slab assembly, FFT/interpolation work, transfers, output writes and
the same full-volume physical quality check are included. The NumPy reference
is tested independently; the timed backend is explicitly CuPy CUDA. Tilted scans
are unsupported. The reconstruction is an approximate Fourier-slice inverse,
so both successful and failed quality gates are retained alongside FBP results.

Single-pass methods are evaluated once and repeated only if accepted. Their
records carry `budget_kind="single_pass"`; the legacy `iterations=1` field
means one whole call, not a CGLS iteration. A failed direct method is never
repeated at larger iteration budgets. Unsupported comparisons and failed
quality gates remain failures. Native/library filter differences can affect
errors, so always inspect physical-scale quality alongside time.

The current direct adapters zero-extend raw rows to cover projected volume
corners before filtering. This retains nonzero filtered tails outside the
acquired row, matching the corrected built-in TomoJAX parallel FBP. The external
helper computes support independently from physical volume corners and saved
poses; padding work is included in every call, including native ASTRA/TIGRE FBP.
It uses no phantom truth or additional measured rays. Raw zero extension assumes
the object is not truncated; it cannot supply missing attenuation in a truncated
acquisition. Earlier `direct-*` records predate this correction and remain as
historical measurements.


`tomojax_fbp_joseph_cgls_pallas` and `astra_fbp_cgls` also compose FBP with a
subsequent CGLS solve. Each budget and repeat starts by recomputing FBP, so its
cost and transfers are included. TomoJAX's initializer stays on the GPU; native
ASTRA CGLS requires the host handoff even though its direct BP accepts GPU data.
The TomoJAX composition supports parallel and shifted anisotropic cases; tilted
Pallas FBP is recorded as unsupported. CGLS budget fields count refinement
updates and the record explicitly identifies FBP initialization. These
compositions do not change the default public solver or the acceptance gates.

## CGLS numerical stability

```bash
uv run --no-sync python bench/cgls_stability.py \
  --output bench/results/cgls-stability.json
```

The fixed 120-case experiment uses 20 signed datasets, three damping values,
two backends, and amplitudes from `1e-3` to `1e3`. A separately assembled forward
matrix and FP64 dense least-squares solve check the volume and stationarity.
It tests solver arithmetic on small systems, not the physical accuracy of the
projection model or conditioning of large scans. The retained record is
[`reference/cgls-stability-rtx4070-laptop.json`](reference/cgls-stability-rtx4070-laptop.json.gz).

## Large reconstruction check

```bash
uv run --no-sync python bench/compare_reconstructions.py \
  --suite gaussian-v1 --sizes 256 512 --views 720 --geometries parallel \
  --repeats 7 --methods tomojax_fbp_host_pallas tomojax_fbp_pallas astra_fbp3d_cupy astra_fbp2d tigre_fbp \
  --output bench/results/large-fbp.json
```

The full-volume error threshold remains 0.03. The changed view count is recorded
and does not count as the original 180-view standard suite. Large phantom truth
and analytic projections are generated in bounded FP64 chunks, preserving the
original component order and casting each output element once to FP32. The
[bitwise parity record](reference/fixture-chunking-parity.json.gz) checks 18 cases
against the previous generator. This check covers the reconstruction part of
the showcase only, not motion recovery or generalization to real scans.


## Known-volume pose recovery

`pose_recovery.py` fits independent analytic Gaussian measurements with the public
Joseph projector. It records truth, initial and estimated per-view parameters,
subpixel/angle errors, every iteration, cold workflow time and repeated workflow
times. Timings include transfers, integer FFT correlation and output transfer;
fixture generation and error diagnostics are outside the timer. The object is
known. This is not a joint reconstruction/alignment speed benchmark.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python bench/pose_recovery.py \
  --sizes 64 128 256 --views 30 --interpolation cubic --noise .005 \
  --output bench/results/known-volume-cubic.json
```

The default case has three independent initial rotation errors within +/-3
degrees and two translation errors within +/-10 native detector pixels.
Rotations are right-composed with nominal poses; translations are in detector/
world x,z coordinates. These are explicitly different from the legacy alignment
API's object-frame translation parameters. The physical volume margin samples
the same continuous object beyond the detector-sized grid to bound omitted
Gaussian tails. It does not alter measured rays, phantom scale, or perturbations.
Noise is independent Gaussian with sigma 0.005 times clean-data RMS by default.
The gate counts a view only if its rotation-vector norm is <=0.01 degrees and
its two-dimensional translation norm is <=0.05 pixels, with >=99% accepted views
in every recorded run. Weakly observable noisy cases remain in the output even
when they fail. The script exits 1 when any case misses that gate.

`joseph_derivatives.py --interpolation cubic` benchmarks the cubic component.
Its version-3 records explicitly store interpolation. Ordinary JAX reverse AD
is limited to size 32 for cubic (64 for linear) to bound its sample tape; all
larger cases still compare against independent ordinary-JAX JVPs. The warmup
and timing policy is unchanged.


`joseph_pose_normals.py` compares the public fused pose-normal-equation operation
with an explicit Jacobian using the same CUDA projector, and checks a separate
ordinary-JAX directional gradient and quadratic form at every size. Both methods
return residuals and include geometry preparation. It reports compiler storage
estimates separately from process peak memory.

Version 2 of `pose_recovery.py` accepts `--normal-method explicit|fused` (default
`fused`). Both paths retain the current residual for line search and evaluate
four nonzero candidates plus a zero step. The explicit control uses a shared
parameter probe to obtain each view's local Jacobian columns without forming
zero derivatives between different views. The two paths use the same damping,
clipping, line-search candidates, initialization and error gates. Version-1
records remain available and used five projected candidates per iteration.

Version 3 adds `--line-search streamed|stacked` (default `streamed`). Streaming
retains only the current candidate image and selected per-view parameters/loss;
it preserves the same candidate order and tie behavior. `stacked` retains the
prior implementation for comparison. The fixture defaults remain unchanged.
An additional `--phantom nine-gaussian --initialization nominal --margin 0`
case uses nine fixed asymmetric, noncoplanar smooth blobs and true perturbations
within +/-3 degrees and +/-10 native pixels, starting at nominal poses. Its
complete object specification is stored in the output. This additional
informative case does not replace the weak two-Gaussian fixture or its failures.
Use `--seeds 9345 128904 61937 --repeats 7` for the retained richer-object sweep.
The acceptance gates and noise definition are unchanged.

## Quality-verification buffers

Complete reconstruction timings include the same full-volume physical relative
L2 check for every method. Verification now converts and reduces bounded chunks
in FP64 instead of allocating several full FP64 volumes. Shape mismatches are
rejected, and nonfinite values are checked in every chunk. The metric and fixed
acceptance thresholds are unchanged; earlier retained records still include
their original verification implementation and source hash. Compare new solver
timings against competitors rerun with the same verification code.

Pose-fixture generation constructs the scan geometry directly and evaluates
analytic Gaussian integrals and noise in bounded view batches. It preserves
the original FP64 arithmetic, FP32 conversion, random-number order and measured
values. Fourteen saved fixtures, including anisotropic and tilted scans, both
phantoms and both initialization modes, match bit for bit after this change.
Independent numerical line quadrature checks cover batch boundaries. Fixture
generation remains outside the timed reconstruction/recovery interval.
