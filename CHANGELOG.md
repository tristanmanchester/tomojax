# Changelog

## Unreleased

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
