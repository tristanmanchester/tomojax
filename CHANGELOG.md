# Changelog

## Unreleased

- Estimate a detector-centre offset together with per-view motion:
  `tomojax align --mode cor_then_pose` now runs the pose solver and saves the
  constant part of the recovered detector-u shifts as the detector centre,
  leaving the per-view motion in the pose table (`auto` and `max` add it to
  their setup estimate). The fold runs in `align_multires`, so the Python
  API's `cor_then_pose` schedule, which requires detector-frame translations,
  behaves the same; the API also adds `fold_detector_offset`. With a +3.7 px offset and ±0.5°/±8 px motion on
  analytic 128³ scans, rotation errors fall from 0.23–0.26° to 0.005–0.006°,
  and the offset matches its identifiable value to 0.004 px. On the gVXR chip
  phantom it recovers a 3.2 px offset to 3.18 px. The previous mode searched
  for the offset before correcting any motion, which biased it.
- `tomojax align` now uses detector-frame translations by default
  (`--translation-frame detector`, with `--gauge-fix none`). Object-frame
  translations move the sample along its own x and z axes, so at views where
  its x axis lies along the beam they cannot shift its image horizontally; a
  constant shift needed 70 px translations near 90° and 270° on the chip
  phantom, with 0.37 px errors.
  Detector-frame recovery is as accurate or better on every benchmark tried.
  `--translation-frame object` restores the previous pose tables.
- `tomojax recon --apply-saved-alignment` applies saved poses in the
  translation frame they were estimated in, which `tomojax align` now records
  with the gauge metadata; files without it are read as object-frame poses.
- A saved detector roll of zero no longer gives `tomojax recon` an explicit
  detector grid, which forced FISTA-TV onto the ray-model reference path one
  view at a time. On the 720-view chip phantom, one iteration took more than
  400 s; 100 iterations now take 29 s.
- `tomojax recon` streams host projections for CGLS and SPDHG-TV as well as FBP
  and FISTA-TV.
- Stream projections from host memory in CGLS and SPDHG-TV, as FISTA-TV
  already did. Streamed CGLS solves the equivalent normal equations: only
  volume-sized arrays stay on the device, and each residual recomputation
  reads the views once. Streamed SPDHG-TV keeps the data, any weights and its
  sinogram-sized dual variable in host memory and moves one block per
  iteration (bitwise identical to device-resident data). On a 512-cubed,
  3072-view laminography scan (3.2 GB) on an 8 GB GPU, CGLS peaks at 3.2 GB
  and SPDHG-TV at 4.3 GB. `CGLSConfig` and `SPDHGConfig` gain
  `stream_projections` (``None`` streams stacks above 40% of free device memory).
- Report the detector-u (centre-of-rotation) offset implied by pose
  alignment: `tomojax align` logs it and writes `implied_detector_u_px` to the
  manifest, and `tomojax.align.api.implied_detector_offset` computes it. Pose
  mode absorbs such an offset exactly into the per-view translations; the
  report separates the constant part from the view-dependent shift of a rigid
  object translation.
- Add a gVXR chip-package phantom for laminography (`bench/phantoms`): a
  Blender script models eight closed, disjoint material meshes (glass-epoxy
  substrate, copper ground plane, vias and traces, silicon die, gold bond
  wires, silica-filled mold, voided SAC solder balls), and a simulator turns
  gVXR path lengths into 25 keV measurements with xraylib attenuation and
  refraction, Fresnel phase contrast, detector blur, pixel integration,
  Poisson noise, pixel gain and noisy flats, with per-view motion and a
  detector offset. An exact mesh voxeliser provides the truth.
- Seed `tomojax align --mode cor` with a search for the detector-u offset
  whose FBP reprojects most consistently (a coarse scan over a quarter of the
  detector, then golden section), replacing the opposite-view pairing that
  needs a parallel half or full turn. All CLI alignment modes now default to
  Joseph integration. With a +3.7 px offset on analytic 128-cubed scans, COR
  mode recovers 3.693 and 3.677 px (parallel, laminography) in 41 and 84 s,
  where it previously reached 3.50 and 3.59 px in about 380 s.
- Cache the pose Jacobian columns of the coupled solver whenever five
  sinograms fit in a quarter of free device memory (previously a fixed 64 MB)
  and apply cached columns to all views in single batched contractions. On a
  256-cubed, 361-view laminography scan the columns were recomputed in every
  conjugate-gradient iteration, 60% of GPU time; full-resolution
  `tomojax align --mode pose` now takes 188 s instead of 13.6 minutes
  (22 minutes this morning) with rotations to 0.0030 deg.
- Solve each view's 5-by-5 pose block in symmetrically scaled variables and,
  only where its FP32 Cholesky factor still fails, with Marquardt damping of
  1e-5 relative to its diagonal. Rotation and translation columns differ in
  size by orders of magnitude, and unscaled factorisations of weakly
  determined blocks returned NaN, which rejected whole Gauss-Newton steps.
- Seed translations at the first coarse-to-fine level, where the shift search
  previously never ran because that level starts from explicit zero poses.
- Fix the reconstruction step inside alignment with Joseph integration. Its
  explicit gradient backprojected with the ray model's transpose (2% from the
  Joseph adjoint), and traced poses sent it to the JAX reference operators
  instead of the CUDA kernels, which accept dynamic poses. It now uses the
  matched plane transpose and the CUDA kernels, with 64 views per batch.
  `tomojax align --mode pose` on analytic 128-cubed scans takes 45 and 41 s
  instead of 86 and 72 s (laminography, parallel) at unchanged accuracy, and
  the README example 14 instead of 35 s.
- Add a CUDA C gather transpose for linear Joseph plane sampling, compiled at
  run time with CuPy (now part of the `cuda12` extra) and launched on XLA's
  stream inside compiled solvers. Each thread loops once over the joint
  footprint of four consecutive z voxels, and weights round exactly like the
  forward projection. Laminography backprojection is 1.6x and parallel 1.3x
  faster at 256-cubed scale; structured 256-cubed laminography CGLS runs in
  1023 instead of 1372 ms warm and a 512-cubed FISTA-TV solve in 48.7 instead
  of 63.1 s, and a full-resolution 256-cubed laminography alignment 13.6 instead
  of 22 minutes (rotations to 0.0029 deg). It is used for volumes of 2^24 voxels and more, since its start-up
  costs about 0.3 s per process; `TOMOJAX_CUDA_KERNELS` forces it on (1) or
  off (0).
- The alignment shift search reprojects, shifts and correlates 32 views at a
  time, so its padded correlation spectra stay bounded for long scans, and
  host-streamed FBP reads the next view batch while the current one runs.
- Stream projections from host memory in FISTA-TV. NumPy or memmap stacks
  larger than 40% of free device memory are read one view batch at a time
  inside the compiled solve (`FistaConfig(stream_projections=...)` forces
  either way), with identical results. A 512-cubed laminography scan with 3072
  views (3.2 GB) runs on an 8 GB GPU at a 4.3 GB peak; where both fit,
  streaming costs 4%. `tomojax recon --algo fista` passes host projections.
- Add `tomojax recon --algo cgls`, optionally FBP-initialised with
  `--warm-start fbp`, so the command line has the fastest-converging
  unregularised solver the Python API already offered. Document aligning a
  large scan at reduced resolution and reconstructing the full data with the
  recovered poses: a 256-cubed laminography alignment stopped at half
  resolution takes 185 s instead of 22 minutes (rotations to 0.0051 instead
  of 0.0030 deg).
- Seed pose alignment with a global per-view shift search. Each pass
  reconstructs with the current shifts removed (FBP), reprojects, and moves
  every view to its cross-correlation peak, searching up to a quarter of the
  detector and discarding the shift pattern of a rigid object translation.
  `--seed-translations` is now on by default for `tomojax align --mode pose`
  (`--no-seed-translations` disables it), and `coupled_pose_config` enables
  it; it now also runs in single-resolution alignment, and replaces the
  previous single phase correlation against a ray-model FISTA reconstruction.
  On 64-cubed scans with +/-0.5 deg tilts and +/-15 px shifts (23% of the
  detector), rotation errors fall from 7.7 and 12.6 deg to 0.034 and 0.012 deg
  in parallel and laminography; 128-cubed results are unchanged.
- Reconstruct laminography and posed scans larger than device memory with
  FBP. `fbp` given a NumPy array or memmap streams view batches from host
  memory (bitwise-identical to device input), and `fbp_host` now accepts every
  geometry `fbp` does, writing x-slabs sized to the free device memory. A
  1024-cubed, 1024-view laminography FBP from and to memmaps (4.3 GB each) runs
  in 37 s on an 8 GB GPU, where 768 cubed previously ran out of memory.
  `tomojax recon --algo fbp` keeps projections on the host. The FBP kernel
  accumulates view batches in place. `cgls`, `project_joseph` and parallel
  `fbp_host` no longer make a transient second device copy of NumPy inputs.
- Cut iterative-solver GPU memory. On a 512-cubed, 768-view laminography scan
  on an 8 GB GPU, CGLS (previously out of memory) peaks at 5.6 GB, FISTA-TV
  falls from 7.2 to 4.6 GB and SPDHG-TV (previously out of memory) peaks at
  4.9 GB. The Joseph CUDA kernels keep sinograms in their (view, v, u) layout,
  so XLA no longer stores transposed copies of every sinogram around solver
  loops, and the gather transpose accumulates view batches in place.
  FISTA-TV computes its data gradient batch by batch without a sinogram-sized
  temporary, its TV prox runs projected gradient on the dual (Chambolle) with
  three persistent volumes instead of five, and zero starting states are
  created inside the compiled solves. CGLS always adopts the recomputed
  residual and no longer retains a second state for breakdown. SPDHG-TV
  recomputes the TV divergence instead of differencing dual fields and no
  longer allocates a sinogram of unit weights. The forward kernel's 8-by-16
  ray tiles also make laminography projection 18% faster. The CLI sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` unless already set.
  FISTA-TV on the 128-cubed TV comparison now reaches 0.076 in 1.13 s.
- Add `bench/compare_tv.py`. On a 128-cubed structured parallel scan with 3%
  noise and 50 iterations, TomoJAX FISTA-TV reaches 0.077 relative error in
  1.28 s; TIGRE's FISTA reaches 0.101 in 9.7 s and ASD-POCS 0.146 in 5.7 s.
- Align coarse to fine by default in `tomojax align --mode pose` when the
  reconstruction grid's shortest axis has at least 64 voxels (factors 4, 2, 1
  from 128). A 256-cubed, 361-view laminography alignment takes 22 instead of
  44 minutes with the same 0.003 deg rotation accuracy and a better volume, and
  +/-3 deg motion is recovered in 64-cubed laminography where a single level
  stopped at 0.037 deg.
- Add `tomojax.align.coupled_pose_config(**overrides)`, the configuration
  `tomojax align --mode pose` runs, for Python callers; `AlignConfig()` keeps
  its older alternating defaults. Add an alignment example and README figure:
  a 96-cubed laminography scan with +/-1 deg and +/-2 px per-view motion goes
  from 0.57 to 0.087 relative error in 35 s, with rotations recovered to
  0.0026 deg, on analytic data from continuous objects.
- Add Joseph plane sampling to alignment (`ray_integrator="joseph"` or
  `"joseph_cubic"`, CLI `--ray-integrator`) and make it the coupled pose
  solver's default. Its matched gather transpose replaces the exact
  integrator's atomic scatter: on analytic 128-cubed scans of continuous
  objects (181 views), `tomojax align --mode pose` recovers parallel and
  laminography poses to 0.0061 and 0.0024 deg in 46 and 93 s, where exact
  integration took 588 and 1044 s for 0.0059 and 0.0016 deg. Joseph plane
  coefficients now accept affine detector grids (calibrated offsets and roll),
  so the model also serves calibrated detectors.
- Fix pose alignment through `tomojax align` and `align_multires`. Pose
  stages skipped reconstruction, so pose-only schedules (the default `pose`
  mode) optimised against an all-zero volume and returned the nominal poses.
  Pose stages now alternate with reconstruction again; a supplied `recon_L` is
  honoured at full resolution.
- Make the coupled volume-and-pose solver the default for `tomojax align
  --mode pose` (`--pose-solver coupled`; `alternating` keeps the previous scheme,
  and remains the default for modes with setup stages). It
  implies exact ray integration, least squares without TV, fp32 gathers and up
  to 30 early-stopped outer iterations, and rejects explicitly conflicting
  options. On the six free-voxel pilot cells the default CLI now recovers five
  (rotation 0.0006-0.008 deg), where the alternating scheme left 0.1-4 deg.
  `gn_coupling="joint"` now applies to every pose-only stage of a named
  schedule, and the coupled solver no longer shifts poses alone to fix the
  translation gauge, which had capped its accuracy at about 0.08 deg.
  The pilot's data share the exact integrator's voxel-basis model; on analytic
  data from continuous objects all solvers settle at a 0.09-0.5 deg
  discretisation floor at 32 cubed, with reconstructions as good as with the
  true poses.
- Run FISTA-TV and SPDHG-TV on the same batched operators as CGLS, and make
  Joseph plane sampling with Pallas kernels on CUDA the default projector for
  all three (`projector_model="auto"`). FISTA-TV previously used the JAX ray
  model one view at a time: 50 iterations on a 64-cubed scan fall from 20 s to
  0.15-0.19 s, and 40 SPDHG-TV iterations from 1.3 s to 0.05-0.07 s, with equal
  or lower reconstruction error. Joseph matches analytic line integrals as well
  as the ray model. Explicit detector grids and the exact ray integrator keep
  the ray-model path; `projector_model="ray"` restores the previous operator.
  FISTA's `views_per_batch` now defaults to 64 batched views (CLI default too).
  Alignment's internal reconstructions keep the ray model of its pose objective.
- Size alignment's reconstruction batches from free GPU memory by default
  (`views_per_batch=0`, also the CLI default). Both alignment profiles used one
  view per batch, launching a projector call for every view in every
  reconstruction pass. On the six free-voxel alignment cells, warm alignment is
  1.9-2.3x faster and cold 1.4-2x, with the same accepted cells and unchanged
  peak GPU memory. An explicit positive value is still honoured.
- Clip each Joseph forward ray to the planes it crosses. Skipped planes
  contributed exact zeros, so projections are bitwise unchanged; forward
  projection is 8-13% faster at 256 cubed in parallel, anisotropic and tilted
  scans. Trace each Joseph kernel once per configuration instead of at every
  call site, cutting a CGLS solve's tracing time by about a third.
- Weight filtered backprojection exactly for every circular parallel-beam
  scan. `fbp` now fits the rotation axis, arc and angular spacing from the view
  poses and applies the matching per-view filter: a ramp along u scaled by the
  sine of the ray-axis angle, divided by how many acquired views measure each
  frequency. Tilted (laminography) axes, partial or full turns and irregular
  angles were previously weighted as a uniform untilted half turn. In tests,
  tilted reconstructions now match the measured-frequency truth to 5% (14% and
  42% before for full and half turns); uniform untilted half turns are unchanged.
  An explicit `FBPConfig.scale` keeps the previous uniform weighting.
- Backproject FBP voxel by voxel with bilinear detector interpolation in every
  geometry, using the Pallas kernel on CUDA. Laminography and anisotropic FBP
  previously used the ray-model adjoint, which blurred the result (5% error on
  a smooth parallel phantom, against 0.1% now). Explicit `det_grid` inputs keep
  the ray-model path.
- Start faster. Importing `tomojax.geometry` and `tomojax.recon` no longer loads
  JAX or SciPy until a JAX-based function is used, so a Fourier reconstruction
  imports in about 50 ms instead of 450 ms. Compiled JAX programs are cached on
  disk by default (`TOMOJAX_JAX_CACHE=off` disables, `TOMOJAX_JAX_CACHE_DIR`
  relocates; an existing JAX cache setting wins). CGLS checks its inputs inside
  the solve instead of compiling separate programs, cutting a cold 64-cubed
  laminography call from about 930 ms to 675 ms, or 385 ms with a warm cache.

## 0.3.0 — 2026-10-04

This release adds matched Joseph/CGLS reconstruction, exact trilinear ray
integration, opt-in coupled alignment, and revised installation and data guides.
It remains an early research release: the frozen reconstruction comparison has
accepted pairs in 26/27 cells, and the opt-in free-voxel alignment pilot passes
5/6 cells. The stretch performance and recovery targets are not met. See
[measurement scope](docs/measurements.md) and [known limitations](docs/known-limitations.md).

- Fix NaN derivatives through Huber-TV updates in flat image regions and through
  the Huber conjugate proximal map at zero duals. Preserve the ordinary update
  values; verify regularized unrolled and implicit reconstruction derivatives
  against dense systems across parallel, anisotropic, and tilted geometry.

- Store bulk benchmark evidence as lossless gzip archives with verified original
  and archive hashes. Keep summaries readable and exclude regenerated raw logs
  from the source change list; preserve every recorded measurement and failure.

- Rewrite the README and task guides with explicit installation, physical units,
  input domains, supported scope, and failure-inclusive measurement links. Add
  a reproducible public-API reconstruction figure and image provenance catalog.
- Fix the Python example's removed projector import and use a matched public
  Joseph/CGLS workflow. Run it in CI. Extend the fresh installed-wheel check
  through reconstruction and labelled slice export using shared smoke steps.
- Share angle-sidecar parsing between TIFF ingestion and preprocessing. Reject
  empty/nonfinite vectors, non-vector NPY arrays, and malformed rows after
  numeric data begins; preserve acquisition order and leading CSV headers.
- Correct TIFF ingestion's `--det-center-u/v` conversion from detector pixels
  to physical length. Previously non-unit detector pitch produced wrong stored
  offsets. Re-ingest affected stacks from the original data and requested pixel
  offsets; existing files are not silently rewritten.
- Make developer recipes preserve the selected environment with `--no-sync`.
  Replace the blanket surface-test marker with explicit API/CLI/IO/checkpoint
  selections; keep numerical tests and aggregate coverage in the full gate. Add an optional plotting dependency group.

- Add opt-in exact line integration of the zero-extended trilinear voxel basis
  to alignment and iterative reconstruction. Carry `ray_integrator="exact"`
  through reconstruction, pose scoring, setup validation and translation seeds;
  reject resuming under a different operator. CUDA provides a matched volume
  adjoint, while alignment pose scoring retains the differentiable JAX reference.

- Add opt-in coupled volume/pose Gauss–Newton updates, with a stacked solve
  or exact elimination of the small pose blocks. Preserve the fixed-volume
  default. The frozen free-voxel pilot passes five of six cells; noisy
  anisotropic recovery still misses the rotation gate.
- Reuse compiled joint objectives across scans with matching shapes and solver
  options, while passing each scan's measurements, geometry arrays and weights
  as inputs. The six-cell comparison shows faster warm calls, with unchanged
  cold startup, process GPU memory and recovery coverage.

- Stop inflating the reused FISTA step bound by 20% after every alignment
  outer iteration. Preserve the solver's effective bound instead of shrinking
  voxel updates exponentially, including across checkpoint resume. Convert the
  carried total bound to the public solver's data bound on fallback, so Huber-TV
  curvature is included once.
- Replace alignment's dimension-only initial step estimate with the maximum
  row sum of the nonnegative normal operator. Account for physical voxel and
  detector spacing, offsets, sampling density and tilted geometry; a change of
  length units no longer changes unregularized voxel updates.

- Keep rigid pose composition, inverse-pose translation and Gauss–Newton
  normal equations at full FP32 multiplication precision on CUDA. Reduced
  matrix-multiply defaults could bias recovered small rotations.

- Add an explicit central finite-difference Gauss–Newton Jacobian for sharp
  object recovery at trilinear interpolation boundaries, with physically scaled
  translation and rotation stencils. Preserve default autodiff and exact
  forward/volume-adjoint semantics; numerical columns cost extra projections.

- Add explicit detector-frame translations to the public alignment API,
  preserving object-frame defaults and checkpoint compatibility. Carry the
  convention through pose updates, reconstruction, multiresolution stages,
  checkpoints and parameter exports. Reject incompatible mean-shift gauge
  constraints and resuming under a different frame.
- Apply existing per-view poses when reconstructing training subsets for setup
  objectives; previously those subsets discarded the pose offsets.
- Correct the direction of phase-correlation translation seeds and convert
  detector displacements into the chosen pose frame. Preserve frozen offsets
  and discard unobservable seed directions in the legacy object-frame model.
- Correct comparison-worker startup isolation: external ASTRA/TIGRE fixture
  loading no longer imports JAX/Pallas. Retain measured warm attempts for failed
  reconstruction budgets without counting them as accepted-result timings.

- Overlap bounded host row preparation and output writes with large CUDA
  Fourier reconstructions, including asynchronous transfers through pinned
  buffers and two pending GPU slabs. Cache bounded immutable small-geometry
  arrays and reduce fractional-row interpolation allocations while preserving
  arithmetic and independent per-call working buffers. Reuse device queues
  across calls to prevent growth of cached stream-specific allocations.

- Extend the exact unit detector-row CUDA adjoint specialization to cubic
  interpolation, reusing vertical weights while preserving negative lobes and
  absolute-weight error bounds. Tilted and non-unit mappings retain the general
  gather path.

- Add opt-in `CGLSConfig.gradient_damping` for squared physical voxel differences,
  with free boundaries and a matched regularized normal-residual stopping check.
  The zero default preserves unregularized execution.

- Add `joseph_pose_normal_equations` for per-view least-squares pose updates.
  CUDA accumulates the gradient and small Gauss–Newton matrices directly and
  returns residuals for line search, without storing a full projection Jacobian.

- Add explicit Keys cubic interpolation to Joseph projection, fused loss and
  CGLS, with matched CUDA volume/pose derivatives and a gather transpose that
  retains negative weights. Linear interpolation remains the default. CGLS
  accounts for those negative weights in its componentwise roundoff estimate.

- Add `project_joseph` with bounded-memory first-order CUDA volume/pose
  derivatives and `joseph_l2_value_and_grad` with fused projection/pose-gradient
  accumulation. Preserve a JAX reference, physical coordinates, batching and
  first-order forward/reverse transformations; existing alignment defaults stay
  on the trilinear ray model.
- Remove repeated dynamic integer division from the Joseph gather transpose
  while preserving its footprint and accumulation order.

- Add opt-in `fourier_reconstruct`/`FourierConfig` for uniform parallel scans, with an independently checked NumPy reference, CuPy FFTs and a CUDA Kaiser–Bessel interpolation kernel. Support detector offsets, anisotropic and cropped grids, opposing/reordered views and bounded host-output slabs; keep the default reconstruction selection unchanged.

- Fix CGLS stopping when a bright region masked meaningful updates in a weak region. Verify convergence with recomputed residuals and report componentwise FP32 stagnation separately.
- Check CGLS residuals periodically near the FP32 noise floor, preventing occasional JAX CUDA atomic-reduction noise from evading stagnation checks until the iteration limit.

- Add opt-in Joseph plane sampling to CGLS, with a deterministic CUDA gather transpose and independent physical-ray matrix checks. The default ray model is unchanged.
- Add optional isolated gVXR material fixtures with verified units and geometry, synthetic spectral attenuation and reproducible Poisson noise.
- Add optional `cgls_multires` with explicit per-level iteration budgets, physical-coordinate-preserving initialization, and a final full-data solve.
- Preserve all physical volume faces when coarsening odd or explicitly positioned grids; keep coarse detector rays uniformly spaced without duplicated edge samples. Old multiresolution alignment checkpoints are rejected because their sampling convention can differ; restart those runs. Single-resolution checkpoint compatibility is unchanged.
- Add frozen sharp-ellipsoid and noisy-ellipsoid comparison suites with independent chord-length data and retained failed quality gates.
- Add public `cgls`/`CGLSConfig` for matched-adjoint least squares with scalar damping, JAX/CUDA backends, dynamic iteration budgets, and explicit termination diagnostics.
- Accumulate the Pallas batch adjoint into one volume instead of allocating a volume per view.
- Factor Pallas interpolation and share ray traversal across kernels; preserve masked boundary loads and the forward sample recurrence.
- Build built-in laminography pose stacks with one device transfer and preserve custom parallel-geometry pose overrides.
- Add isolated, fixed-quality reconstruction comparisons against ASTRA/TIGRE, including unsuccessful budgets, cold search costs, and sampled process GPU memory.
- Reuse compiled FISTA/SPDHG loops and shared operator-norm estimation with dynamic data and geometry inputs; transfer loss histories to the host in one operation.
- Accumulate FP32 batches of adjoint rays into one volume, reducing temporary memory while preserving mixed-precision adjoint semantics.
- Cache alignment gauge projection and compile L-BFGS evaluations/updates once per optimization call.
- Fix false rejection of finite anisotropic volumes caused by float32 finiteness-average rounding.
- Remove duplicate implicit-alignment reconstruction and propagate its measurement-data gradient through the implicit adjoint.
- Remove exact detector-size tuning overrides and preserve explicitly requested JAX reconstruction backends.
- Add full public workflow benchmarks and regressions for fresh inputs, weighted solver steps, operator norms, implicit gradients, and alignment validity.
- Correct FBP attenuation scaling and replace circular FFT filtering with a zero-padded discrete ramp. Quantitative FBP output changes; regenerate reconstructions that depend on the old normalization.
- Preserve filtered detector tails through the full volume in built-in parallel FBP, removing corner bias from premature cropping. Raw data remain zero outside the measured detector; genuinely truncated acquisitions still need an appropriate correction.
- Stream direct FBP padding, filtering and backprojection in bounded view batches, avoiding full-stack FFT temporaries. Preserve partial batches, shifted detector coordinates and the explicit Pallas helper's translated poses.
- Add `fbp_host` and `FBPHostConfig` for axial-slab parallel reconstruction from NumPy/memmap inputs into optional host output storage, bounding projection and volume device storage by slab dimensions while preserving physical coordinates and partial slabs.
- Add automatic voxel-driven CUDA/Pallas parallel FBP and `FBPConfig.backprojector` (`auto`, `jax`, `pallas`); remove the superseded tiled FBP kernel.
- Fix long-ray adjoint coordinate drift and scatter FP32 gradients directly into the accumulator.
- Synchronize FBP fallback chunks before committing progress, so asynchronous out-of-memory retries preserve every view exactly once.
- Preserve useful Pallas tiles on odd detector dimensions with masked loads/stores; reject unsafe integer-slice specializations for tiny tilts and accumulated row-spacing drift.
- Fix Pallas loop unrolling and CPU interpretation of masked/repeated-index atomic updates.
- Upgrade JAX to 0.11.2 with a `<0.12` compatibility bound and use ImageIO's maintained tifffile backend.
- Add analytic accuracy, adjoint, weighted-gradient, real-CUDA, and iterative convergence regressions, plus reproducible ASTRA/TIGRE and FBP benchmarks with retained results.
- Consolidate the public CLI/API, minimal examples, and product workflow tests.
- Remove obsolete benchmark harnesses, historical/v1-parity gates, one-off runners, oversized development logs, and diagnostic artifact builders from the shipped tree; retain them in a separate development archive. Current reproducible development benchmarks live in `bench/` outside the installed package.
- Remove `tomojax.bench` and `tomojax.verify` entirely from the shipped tree.
- Laminography geometry now aligns the rotation axis with the volume z-axis, so lamino reconstructions produce z-stacks with square x–y slices; regenerate datasets/recons if you relied on the old orientation.
- Recon CLI now crops to detector FOV by default (`--roi auto`); pass `--roi off` to keep legacy behavior.
- Normalize NX volume IO: write volumes on disk in `zyx` order with `@volume_axes_order` metadata, transpose on load, and warn (silence via `TOMOJAX_AXES_SILENCE`).
- Add CLI `--volume-axes` override for `recon`/`align` and update NX data wrangler to tag volumes.
- Fix CUDA “invalid image” faults on Turing GPUs by replacing the projector’s small GEMM with element-wise transforms, ensuring SPDHG/FWD projections compile cleanly on RTX 6000/8000 while keeping mixed-precision gather heuristics.

## 0.2.0 — v2 at repo root
- Promote the v2 implementation to the primary package at `tomojax`.
- Move v2 code under `src/tomojax`, tests under `tests`, docs under `docs`.
- Add root `pyproject.toml` (src/ layout) and update `pixi.toml` tasks.
- Remove legacy root examples and experimental folders.
