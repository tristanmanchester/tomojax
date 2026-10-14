# Known limitations

The shipped package covers the workflows listed in
[`support-matrix.md`](support-matrix.md), including 5-DOF pose alignment,
detector-centre/COR alignment, and mixed setup and pose correction.

## Alignment limitations

Alignment is experimental and needs scan-specific review. What it cannot tell
apart or recover, such as setup error absorbed by the poses, a nonzero-mean
shift against a detector-centre offset, detector-v shifts and abrupt jumps, is
listed with how to check a result in the
[alignment guide](alignment-guide.md#check-the-result).

- On the six-cell free-voxel pilot at 32³, every cell misses the 0.01°
  rotation gate with the default Joseph integrator, and the noisy anisotropic
  cell misses it even with the exact one; see the
  [measurements](measurements.md#free-voxel-pilot).
- `AlignConfig`'s default five-parameter pose update uses object-frame
  translations, `(dx, 0, dz)` in physical units. Near a 90-degree view their
  projections onto the detector become nearly parallel, including for tilted
  geometry, so this representation cannot recover arbitrary detector-plane
  motion, or a constant detector shift such as a centre-of-rotation offset, at
  those views; optimiser damping or kernel speed cannot restore the missing
  degree of freedom. `tomojax align` and `tj.align` default to detector-frame
  translations instead; see
  [translation frames](alignment-guide.md#choose-the-translation-frame).
- The default autodiff Gauss–Newton Jacobian is one-sided at interpolation-cell
  boundaries. Sharp objects can become trapped after a wrong shift update.
  The explicit `gn_jacobian="central"` option uses symmetric numerical columns
  at extra projection cost; see
  [the solver reference](alignment-solver.md#gaussnewton-updates-at-interpolation-boundaries).
- Gradients through regularised reconstruction are finite after the Huber-TV
  correction, but sampled-ray pose derivatives remain local to sampling and
  interpolation branches. A central finite difference across a boundary can
  disagree with that derivative, and finite gradients alone do not prove
  reliable pose recovery.

## Data and geometry boundaries

- TIFF `import` packages data; it does not apply flat/dark correction or a log.
  Reconstruction expects line integrals (log attenuation).
- `tomojax preprocess` of TIFF frames records unit detector spacing and parallel
  geometry. `tomojax import` them with their geometry first, or pass
  `geometry=` to `tj.load_frames`; see the
  [real scan guide](real-laminography.md#prepare-tiff-data).
- NXtomo and Data Exchange files not written by TomoJAX get a parallel beam
  with a centred detector, and unit pixels unless the NXtomo detector records
  `x_pixel_size` and `y_pixel_size`. Pass `detector=` or `geometry=` to
  `tj.load_frames` to set the measured geometry; see the
  [quickstart](quickstart.md#correct-raw-detector-frames).
- A float stack with no flats or `image_key` is taken to be line integrals
  already: TomoJAX cannot tell raw float counts from corrected data. Correct
  such frames with `tj.load_frames(path, flats=...)`.
- A laminography dataset without explicit tilt metadata currently uses 30° when
  building geometry. Inspect and set measured geometry before reconstruction.
- `tomojax inspect --preview` PNGs are display-scaled central slices. Use the
  floating-point dataset volume for quantitative analysis and honour its
  recorded axis order.

## Implementation limitations

- The explicit CUDA Joseph projection API supports first-order derivatives
  for linear and cubic interpolation. Cubic has negative lobes, can overshoot,
  and uses a larger stencil; it does not remove physical support truncation or
  poor pose observability. Alignment samples with linear Joseph interpolation
  by default (`ray_integrator = "joseph"`; `"joseph_cubic"` selects cubic).
  Use the JAX reference for higher derivatives, and parameterise poses as rigid
  transforms. Derivatives are local to the selected dominant axis and
  interpolation cell. The fused loss is unweighted half squared error; other
  losses use the differentiable projector. The alignment pipeline does not
  call the fused loss or pose normal equations.
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
  solver's non-Gauss-Newton optimisers are not validated for cone geometry.
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
  for built-in parallel half-turn scans, including nonuniform encoder positions
  and redundant directions. It rejects tilted/custom geometry. Redundant views
  are averaged after physical detector phase/orientation. Its disk frequency
  cutoff and angular interpolation differ from FBP and can produce artefacts
  on sharp or undersampled scans. Check weak features away from the rotation
  axis: good bulk density or central-feature agreement does not certify
  peripheral detail.
  A finer output grid does not add angular measurements. Missing detector data
  are assumed zero; FFT padding prevents wraparound, not object truncation.
  Extending attenuation beyond the measured detector imposes a scientific
  prior, not additional measurements. CUDA requires the optional
  `fourier-cuda12` extra; the NumPy reference is the default.
  Full projection/output storage stays on the host,
  while FFT plans, slab arrays and allocator caches add device memory.
- The iterative solvers need the whole volume and a few volume-sized work
  arrays on the GPU, so a volume larger than the GPU can be reconstructed only
  by FBP or FDK in host slabs. Several GPUs (`devices=`) add speed, not room
  for a larger volume, except in FDK's host slabs. SPDHG-TV and `fbp_host` run
  on one GPU, the `tomojax` commands always use one, and the GPUs must belong
  to one process. [Large scans and several GPUs](large-scans.md) says which
  solvers stream projections from host memory and what each GPU holds.
- Alignment fixes an object's position only up to what the data can see. In
  a cone beam, moving the object sideways needs the along-beam `dy` in some
  views, which the default pose parameters leave out, so the solve may place
  it a voxel or so from another estimate of the same scan (four of the 42
  FIPS walnuts, by 1 to 1.25 voxels); the images are as sharp either way.
- Pallas currently uses JAX's deprecated Triton backend. JAX 0.11.2 is tested
  on CPU and an Ada CUDA GPU, with the dependency constrained below 0.12.
  Migration and additional GPU coverage are still required before widening
  compatibility. See the [test environment](measurements.md#test-environment).
- `just accelerator-smoke` verifies the Pallas projector in interpret mode on
  CPU and attempts a real Pallas run only when JAX reports a non-CPU backend.
  Real CUDA coverage therefore depends on the host CUDA/JAX installation, not
  only this package. Use `just accelerator-smoke-cuda` on CUDA hosts that must
  prove the real accelerator path is available. `just test-cuda` additionally
  runs every test with the GPU visible, numerical tests of real kernels
  included; ordinary CPU CI skips those cases.
- The published timings cover one laptop GPU (and four H100 GPUs for the
  several-GPU runs), synthetic phantoms and the FIPS walnut. The exact
  cone-beam transpose remains slower than ASTRA's
  approximate voxel-driven backprojector, most of all when the detector
  samples finer than the grid; bin such detectors (`Scan.binned`).
  The TIGRE adapter only compares centred isotropic parallel scans so far.

## Next steps

See [`alignment-guide.md`](alignment-guide.md) to choose an alignment mode.
