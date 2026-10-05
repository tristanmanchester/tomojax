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
- Mixed setup and pose correction has gauge ambiguity. Use `--mode auto` only
  with an explicit `--gauge-policy`, such as `anchor_mean`.
- Detector-v or sample-elevation reference shifts are physically ambiguous
  and not reliably recoverable.
- The default five-parameter pose update uses object-frame translations:
  `T_nominal @ se3_from_5d(params)`, with translation `(dx, 0, dz)` in physical
  units. These are not two independent detector-plane shifts. Near a 90-degree
  view, their projection onto the detector becomes nearly singular, including
  for tilted geometry. An asymmetric voxel-object check confirms that a
  nonzero combination can translate along the beam without changing its image.
  This representation cannot recover arbitrary detector-plane motion at those
  views; changing optimizer damping or kernel speed cannot restore the missing
  degree of freedom. The Python API's explicit
  `pose_translation_frame="detector", gauge_fix="none"` option supplies two
  lab detector-plane directions while preserving existing pose-table semantics.
  See [translation frames](alignment-guide.md#choose-the-translation-frame-in-the-python-api).
- Abrupt jumps and short bursts of bad views need more robust diagnostics or
  specialized workflows.
- The default autodiff Gauss–Newton Jacobian is one-sided at trilinear voxel
  boundaries. Sharp objects can become trapped after a wrong shift update.
  The explicit `gn_jacobian="central"` option uses symmetric numerical columns
  at extra projection cost; see [the solver guidance](alignment-guide.md#gaussnewton-updates-at-interpolation-boundaries).
- Gradients through regularized reconstruction are finite after the Huber-TV
  correction, but sampled-ray pose derivatives remain local to sampling and
  interpolation branches. A central finite difference across a boundary can
  disagree with that derivative. See the [retained derivative checks](huber-derivatives-2026-10-04.md),
  including unsuccessful controls; finite gradients alone do not prove reliable
  pose recovery.
- Large combined setup and pose errors can still need staged initialization,
  stronger priors, or manual review.

## Data and geometry boundaries

- TIFF `ingest` packages data; it does not apply flat/dark correction or a log.
  Reconstruction expects absorption/log-attenuation projections.
- TIFF preprocessing through the CLI records unit detector spacing and parallel
  geometry. Use the Python API to supply measured pitch, grid, and tilt; see the
  [real scan guide](real-laminography.md#prepare-tiff-data).
- A laminography dataset without explicit tilt metadata currently uses 30° when
  building geometry. Inspect and set measured geometry before reconstruction.
- CLI slice PNGs are display-scaled. Use the floating-point dataset volume for
  quantitative analysis and honor its recorded axis order.

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
- Built-in geometries use parallel rays. Cone-beam and fan-beam projectors are
  not provided.
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
- `fbp_host` provides NumPy/memmap output using axial slabs for built-in parallel
  geometry. It is not differentiable and does not support tilted scans. Smaller
  slabs reduce device storage but can increase transfer and dispatch costs;
  runtime and compiler allocations remain additional memory costs.
- Pallas currently uses JAX's deprecated Triton backend. JAX 0.11.2 is tested
  on CPU and an Ada CUDA GPU, with the dependency constrained below 0.12.
  Migration and additional GPU coverage are still required before widening
  compatibility. See [the performance report](performance.md).
- `just accelerator-smoke` verifies the Pallas projector in interpret mode on
  CPU and attempts a real Pallas run only when JAX reports a non-CPU backend.
  Real CUDA coverage therefore depends on the host CUDA/JAX installation, not
  only this package. Use `just accelerator-smoke-cuda` on CUDA hosts that must
  prove the real accelerator path is available. `just test-cuda` additionally
  runs numerical tests of real kernels; ordinary CPU CI skips those cases.
- The published timings cover one laptop GPU and synthetic phantoms. Large
  tilted forward projections remain slower than ASTRA in those measurements.
  The TIGRE adapter only compares centred isotropic parallel scans so far.

## Next steps

See [`alignment-guide.md`](alignment-guide.md) to choose an alignment mode.
