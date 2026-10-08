# Known Limitations

The shipped package covers the workflows listed in
[`support-matrix.md`](support-matrix.md), including 5-DOF pose alignment,
detector-centre/COR alignment, and mixed setup and pose correction.

## Alignment limitations

Alignment is experimental and needs scan-specific review. The opt-in coupled
solver passes five of six modest-motion synthetic cells; noisy anisotropic
recovery still fails. The [measurement guide](measurements.md) separates these
results from default CLI behavior and larger-motion targets.

- Pose-only correction can absorb some setup errors. The reconstruction may
  look good while the recovered parameters differ from true geometry. COR mode
  fits setup offsets explicitly but still requires independent validation.
- Mixed setup and pose correction has gauge ambiguity. `--mode full` fixes a
  gauge policy for each stage; an expert direct parameter set mixing setup and
  pose parameters (`optimise_dofs`) needs an explicit `gauge_policy`, such as
  `anchor_mean`, in its `--config` file.
- Detector-v or sample-elevation reference shifts are physically ambiguous
  and not reliably recoverable.
- `AlignConfig`'s default five-parameter pose update uses object-frame
  translations: `T_nominal @ se3_from_pose_params(params)`, with translation
  `(dx, 0, dz)` in physical units. These are not two independent
  detector-plane shifts. Near a 90-degree view, their projection onto the
  detector becomes nearly singular, including for tilted geometry. This
  representation cannot recover arbitrary detector-plane motion, or a constant
  detector shift such as a centre-of-rotation offset, at those views; changing
  optimizer damping or kernel speed cannot restore the missing degree of
  freedom. `tomojax align` defaults to detector-frame translations
  (`pose_translation_frame = "detector"`), and the Python API offers the same
  setting.
  See [translation frames](alignment-guide.md#choose-the-translation-frame).
- A per-view shift with a nonzero mean over the scan cannot be told apart from
  a detector-centre offset; `--mode cor-then-pose` reports it as the offset.
- Abrupt jumps and short bursts of bad views need more robust diagnostics or
  specialized workflows.
- The default autodiff Gauss–Newton Jacobian is one-sided at trilinear voxel
  boundaries. Sharp objects can become trapped after a wrong shift update.
  The explicit `gn_jacobian="central"` option uses symmetric numerical columns
  at extra projection cost; see [the solver guidance](alignment-guide.md#gaussnewton-updates-at-interpolation-boundaries).
- Gradients through regularized reconstruction are finite after the Huber-TV
  correction, but sampled-ray pose derivatives remain local to sampling and
  interpolation branches. A central finite difference across a boundary can
  disagree with that derivative. See the [retained derivative checks](research/huber-derivatives-2026-10-04.md),
  including unsuccessful controls; finite gradients alone do not prove reliable
  pose recovery.
- Large combined setup and pose errors can still need staged initialization,
  stronger priors, or manual review.

## Data and geometry boundaries

- TIFF `import` packages data; it does not apply flat/dark correction or a log.
  Reconstruction expects absorption/log-attenuation projections.
- TIFF preprocessing through the CLI records unit detector spacing and parallel
  geometry. Use the Python API to supply measured pitch, grid, and tilt; see the
  [real scan guide](real-laminography.md#prepare-tiff-data).
- A laminography dataset without explicit tilt metadata currently uses 30° when
  building geometry. Inspect and set measured geometry before reconstruction.
- `tomojax inspect --preview` PNGs are display-scaled central slices. Use the
  floating-point dataset volume for quantitative analysis and honor its
  recorded axis order.

## Implementation limitations

- The explicit CUDA Joseph projection API supports first-order derivatives
  for linear and cubic interpolation. Cubic has negative lobes, can overshoot,
  and uses a larger stencil; it does not remove physical support truncation or
  poor pose observability. The current alignment pipeline still uses its
  existing trilinear ray model.
  Use its JAX reference for higher derivatives, and parameterize poses as rigid
  transforms. Derivatives are local to the selected dominant axis and
  interpolation cell. The fused loss is unweighted half squared error; other
  losses use the differentiable projector. This API does not automatically
  switch the existing alignment pipeline from its trilinear ray model.
- Cone-beam scans: FDK assumes a circular source orbit (it is exact only in
  the orbit plane, with cone artefacts growing away from it) and supports full
  turns, including offset-detector (half-fan) turns with Wang's weights, and
  Parker-weighted short scans on a centred detector (an offset detector in a
  short scan logs a warning). The iterative solvers model any per-view poses exactly. Cone-beam
  projection needs the canonical detector grid (no `detector_roll_deg`
  replay grid; detector roll is part of `ConeBeam`). The CUDA kernels need
  CuPy; without it the JAX reference runs, about ten times slower. On CUDA
  the matched transpose is slower than ASTRA's approximate backprojector (see
  [measurements](measurements.md#cone-beam-projection-and-fdk)). Pose
  derivatives come from the JAX reference; the CUDA forward differentiates in
  the volume only, so alignment's Gauss-Newton columns use central differences
  on CUDA. Cone-beam setup calibration estimates the axis offset and detector
  roll (`calibrate_cone_axis`, also `tomojax align --mode cor`) from slab
  sharpness, which needs in-plane structure at the slab heights and a scan of
  at least 180 degrees plus the fan angle; it does not estimate detector pitch
  or yaw, the axis direction, or the source distances. The alternating
  solver's non-Gauss-Newton optimizers are not validated for cone geometry.
- FBP weights every view exactly for rotation about one fixed axis, at any
  tilt, arc length or angular spacing. It cannot recover frequencies no view
  measured: laminography's missing cone reconstructs as zero, so FBP gives
  elongated features along the rotation axis. Iterative solvers can recover
  part of the cone only through a bounded volume or a prior.
- FBP retains filtered tails across the full volume, assuming zero raw
  attenuation outside the measured detector. This does not recover missing
  measurements of a truncated object. With explicit detector coordinates
  (`det_grid`), FBP uses the ray-model adjoint, uniform `pi / n` weights and the
  supplied detector support.
- Pallas projector paths are optional accelerator backends. Parallel FBP selects
  Pallas automatically on CUDA; `FBPConfig(backprojector="jax")` selects JAX.
  Reference/JAX paths and independent analytic data both validate correctness.
- `fourier_reconstruct` is an opt-in, non-differentiable Fourier-slice inverse
  for uniform, unique half-turn parallel scans. It rejects tilted/custom geometry
  and nonuniform/repeated angles. Its disk frequency cutoff and angular
  interpolation differ from FBP and can produce artifacts on sharp or
  undersampled scans. Missing detector data are assumed zero, without truncation
  correction. CUDA requires the optional `fourier-cuda12` extra; the NumPy
  reference is the default. Full projection/output storage stays on the host,
  while FFT plans, slab arrays and allocator caches add device memory.
- `fbp_host` provides NumPy/memmap output in volume slabs for every geometry
  `fbp` accepts. It is not differentiable. Laminography slabs each stream and
  refilter every view, so smaller slabs reduce device storage at the cost of
  repeated filtering and transfers; runtime and compiler allocations remain
  additional memory costs.
- The iterative solvers need the volume and a few volume-sized work arrays on
  the device. FISTA-TV, CGLS and SPDHG-TV stream NumPy or memmap projections
  from host memory when they are large; SPDHG-TV also keeps its sinogram-sized
  dual variable and any weights there. Streamed CGLS solves the equivalent
  normal equations, which square the condition number of each step's
  recurrence; it recomputes the exact gradient periodically, as in-core CGLS
  recomputes its residual, and streams only with canonical detector grids.
- Several GPUs (`devices=`) share a scan's views; each holds the whole volume
  and the solver's volume-sized arrays, so they add speed, not room for a
  larger volume. Only `tj.project`, `tj.backproject`, CGLS and FISTA take
  them: FBP, SPDHG and alignment run on one device. Shared projections are
  held in device memory, never streamed from the host, and the devices must
  belong to one process (no multi-host runs).
- Pallas currently uses JAX's deprecated Triton backend. JAX 0.11.2 is tested
  on CPU and an Ada CUDA GPU, with the dependency constrained below 0.12.
  Migration and additional GPU coverage are still required before widening
  compatibility. See [the performance report](performance.md).
- `just accelerator-smoke` verifies the Pallas projector in interpret mode on
  CPU and attempts a real Pallas run only when JAX reports a non-CPU backend.
  Real CUDA coverage therefore depends on the host CUDA/JAX installation, not
  only this package. Use `just accelerator-smoke-cuda` on CUDA hosts that must
  prove the real accelerator path is available. `just test-cuda` additionally
  runs every test with the GPU visible, numerical tests of real kernels
  included; ordinary CPU CI skips those cases.
- The published timings cover one laptop GPU, synthetic phantoms and the FIPS
  walnut. The exact cone-beam transpose remains slower than ASTRA's
  approximate voxel-driven backprojector, most of all when the detector
  samples finer than the grid; bin such detectors (`Scan.binned`).
  The TIGRE adapter only compares centred isotropic parallel scans so far.

## Next steps

See [`alignment-guide.md`](alignment-guide.md) to choose an alignment mode.
