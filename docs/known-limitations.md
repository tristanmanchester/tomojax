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

- Ptychography's joint projection matching estimates translations on axial
  parallel scans, not rotations or laminographic motion. Sub-0.1-pixel recovery
  and unbiased motion estimates are not established. Independent known-zero
  tests confirm false motion that grows with noise and persists without it,
  but do not establish that all pose differences on the measured PSI scan are
  bias. The estimator does not reject changes on already aligned input. See the
  [ptychography recipe](alignment-guide.md#complex-ptychography-projections).
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

## Detector-ring correction

`Rings` separates detector response from object structure statistically; it is
not a no-harm guarantee. On four real parallel-scan cuts, known injected
response fields retain 28–61% of their RMS after differential correction.
Independent off-axis analytic structure increments change by up to 11.1% on
I13 and 0.9–3.7% on the other three scans; see the
[ring measurements](measurements.md#detector-response-and-structure-controls).
Review corrected and uncorrected structures, especially on moving samples.
A centred radial feature and an even detector error are indistinguishable in
a stationary full turn: correction of that component depends on the method's
prior, not information uniquely supplied by the scan. Natural motion can
supply extra calibration constraints, but `Rings` does not fit those constraints.

## Data and geometry boundaries

- Projection-only centre reports apply to axial parallel scans. Pair intervals
  describe a selected rigid matching mode, not ambiguous peaks or sample motion;
  opposing shifts determine sums, not individual per-view motion. Row-centre
  slope is an apparent detector trend, not a calibrated three-dimensional axis
  tilt. Use the pair diagnostics when automatic alignment reports disagreement.
  In five low-contrast gVXR mesh cases, wrong-mode errors reached a 3.91 px
  p95 and the conditional 90% intervals covered only one truth; check competing
  modes against an independent calibration rather than treating a narrow local
  interval as a global accuracy certificate. Runtime reports mark global
  coverage uncalibrated for every estimate, including a single pair without
  a disagreement test; automatic alignment remains limited and loading warns.
- `tomojax inspect` checks storage and missing calibration in recognized raw
  acquisitions. It does not run data-driven centre inference; a centre pending
  inference can still fail during loading. A valid report does not certify
  physical geometry or reconstruction readiness.
- External HDF5 links require readable backing files at their recorded paths.
  The loader reports missing detector sources; it does not repair damaged files
  or guess relocated links.
- Flat/dark correction needs at least one detector pixel with resolved
  illumination. If none exists, supply illuminated flats and matching darks,
  or include an illuminated region in the crop. Unilluminated margins and
  bracketed interior gaps remain supported; their correction does not certify
  that the supplied calibration describes the acquisition.

- TIFF `import` packages data; it does not apply flat/dark correction or a log.
  Reconstruction expects line integrals (log attenuation).
- `tomojax preprocess` of generic TIFF frames requires a `detector` table in
  `--config` (`nu`, `nv`, `du`, `dv`, `det_center`). For non-parallel geometry,
  pass `geometry=` to `tj.load_frames`; see the
  [real scan guide](real-laminography.md#prepare-tiff-data).
- Missing detector pixel sizes require `detector=` or `geometry=`. For axial
  acquisitions with recorded pitch, loading estimates an absent rotation centre
  from corrected rows and records its provenance; at least a usable half-turn
  is needed. Failed estimates require explicit `detector=` and never return the
  internal zero seed. Missing angles require `angles=`. See the
  [quickstart](quickstart.md#correct-raw-detector-frames).
- Content-discovered external acquisitions remain detector counts even when
  stored as floats. Missing flat/dark calibration requires `flats=` (frames or
  one level) and `darks=`; a floating dtype is not evidence of correction.
  External NXtomo also requires frame labels (`image_key_path=`); Data Exchange
  defines its separate `data` array as projections. TomoJAX dataset containers
  store line integrals unless explicitly labelled as raw.
- A laminography dataset without explicit tilt or axis metadata requires
  `geometry=` in `tj.load_frames`; no 30° tilt is assumed.
- `tomojax inspect --preview` PNGs are display-scaled central slices. Use the
  floating-point dataset volume for quantitative analysis and honour its
  recorded axis order.

## Implementation limitations

- Held-out projection error is not an image-error certificate. Tested CGLS
  MSE selectors miss the best tested image iteration on 2DeteCT and Walnut
  reference comparisons and an I13 injected-response control; see the
  [stopping-rule limits](held-out-evaluation.md#stopping-rule-limits).
  Held-out iteration and regularisation selection are not automatic.
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
  turns, including offset-detector (half-fan) turns weighted by actual
  conjugate-panel coverage, and
  Parker-weighted short scans on a centred detector (an offset detector in a
  short scan logs a warning). The iterative solvers model any per-view poses exactly. Cone-beam
  projection needs the canonical detector grid (no `detector_roll_deg`
  replay grid; detector roll is part of `ConeBeam`). The CUDA kernels need
  CuPy; without it the JAX reference runs, about ten times slower. On CUDA
  the matched transpose is slower than ASTRA's approximate backprojector (see
  [measurements](measurements.md#cone-beam-projection-and-fdk)). CUDA forward
  projection supports first-order pose and volume derivatives;
  Gauss–Newton evaluates analytic pose columns in one ray traversal. Derivatives
  hold the dominant ray axis and bilinear cell fixed; use the JAX reference for
  higher derivatives. Cone-beam setup calibration estimates the axis offset and detector
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
- CUDA projector paths are optional accelerator backends. Parallel FBP selects
  CUDA C kernels automatically on CUDA; `FBPConfig(backend="jax")` selects JAX.
  Reference/JAX paths and independent analytic data both validate correctness.
- `fourier_reconstruct` is an opt-in, non-differentiable Fourier-slice inverse
  for built-in parallel half-turn scans, including nonuniform encoder angles,
  rejected views and repeated or opposing exposures. It rejects tilted/custom
  geometry. Its disk frequency cutoff and angular interpolation differ from FBP
  and can produce artefacts on sharp or undersampled scans; a large missing
  angular sector is interpolated, not recovered. Missing detector data are
  assumed zero, without truncation correction. CUDA requires the optional
  `fourier-cuda12` extra; the NumPy reference is the default. Full
  projection/output storage stays on the host,
  while FFT plans, slab arrays and allocator caches add device memory.
- Most iterative solvers need the whole volume and several work arrays on the
  GPU. Cone-beam MLEM/OSEM can keep the volume in host memory; unposed axial
  parallel SIRT/MLEM can run in independent slabs. Raw-frame FBP and FDK with
  `tj.reconstruct_slabs` need neither the whole corrected stack nor the output
  volume in host memory. Several GPUs (`devices=`) share views, not the storage
  of a device-resident volume. SPDHG-TV and `fbp_host` run
  on one GPU, the `tomojax` commands always use one, and the GPUs must belong
  to one process. [Large scans and several GPUs](large-scans.md) says which
  solvers stream projections from host memory and what each GPU holds.
- Alignment fixes an object's position only up to what the data can see. In
  a cone beam, moving the object sideways needs the along-beam `dy` in some
  views, which the default pose parameters leave out, so the solve may place
  it a voxel or so from another estimate of the same scan (four of the 42
  FIPS walnuts, by 1 to 1.25 voxels); the images are as sharp either way.
- CUDA projectors compile C kernels with NVRTC. Install `tomojax[cuda12]` and
  a working NVIDIA driver; see [installation](installation.md).
- `just accelerator-smoke` verifies the portable reference mode on
  CPU and attempts CUDA kernels when JAX reports a CUDA backend.
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
  The TIGRE benchmark adapter also represents calibrated cone vectors,
  including detector offsets and per-view rotations; comparisons must retain
  those calibrations rather than substituting a circular centred scan.

## Next steps

See [`alignment-guide.md`](alignment-guide.md) to choose an alignment mode.

ESRF raw Bliss and SPEC EDF metadata do not establish a calibrated detector
centre: it must be estimated from data or supplied. SPEC scan range alone also
needs explicit `angles=`; accumulated darks without `DARK_N` need `darks=`.
An explicit PyHST `.par` supplies its own angles and rotation axis, but only
raw single-image EDFs with fixed axial parallel geometry are supported. PyHST
multi-frame, cone, helical, binning and distortion modes are rejected. Every
projection and calibration file named by the selected input must be available.

HDF5 virtual detector stacks require all referenced pixel files and mapped
regions to be present; missing sources are errors, not zero-filled exposures.

Nikon vendor reconstruction regions (including `OffsetX/Y/Z`) remain source
metadata and are not applied: their coordinate mapping is not established.
TomoJAX uses its own acquisition field-of-view grid unless `grid=` is supplied.
Missing detector offsets or axis calibration require explicit geometry.
