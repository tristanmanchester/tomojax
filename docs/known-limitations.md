# Known limitations

The shipped package covers the workflows listed in
[`support-matrix.md`](support-matrix.md), including one joint alignment solver for selected
translations, rotations and global geometry corrections.

## Reconstruction detail and regularization

On synthetic data scored against truth, FISTA-TV is more accurate than FBP.
On phase-contrast data, FISTA-TV renders propagation fringes as smooth bands
where FBP keeps the bright and dark fringe pair; more iterations do not bring
the pair back. Apply the alignment poses (`Scan.with_poses`) before comparing
methods on a real scan: residual motion blurs edges in every method and can
be mistaken for regularization.

The automatic TV minimum is normalized in physical coordinates, so changing
voxel pitch does not require an additional power-of-pitch weight correction.
This normalization does not guarantee a fixed edge width: the object, forward
model and discrete TV penalty also affect smoothing.

## Alignment limitations

Alignment is experimental and needs scan-specific review. Use the
[alignment guide](align-a-scan.md#check-the-result) to choose parameters and check the result.

- `fit="translations"` is the default for every geometry. It cannot correct a missing global
  tilt or roll, or per-view rotation. Add the corresponding parameter only when it explains a
  remaining artefact. A roll can leave the central slice almost unchanged while blurring slices
  above and below it.
- The image and geometry are jointly estimated. Common rigid object motion is a gauge, and a
  constant parallel-ray translation shares a gauge with the detector centre. Reported parameter
  values depend on the stated gauge; they are not independent instrument measurements.
- Per-view rotations carry a physical angular size prior. This limits unsupported motion but
  does not resolve repeated-pattern ambiguities. A TEM grid can match at a wrong orientation.
  Finite detector pixels also average transmitted intensity before the logarithm, whereas the
  linear projector fits line integrals. At sharp, strongly attenuating edges, that mismatch can
  look like motion. A smaller residual alone does not establish more accurate poses.
- The derivatives are local to interpolation cells and dominant ray axes. Sharp, poorly sampled
  objects can give biased or weakly constrained geometry updates. Exact transpose semantics
  establish a consistent discrete solve, not a complete physical forward model.
- A diffracting crystal, detector fault, changing contrast or material outside the field of view
  can create views that no pose explains. Robust residual weighting and the same-image-step
  line-search comparison reduce those views' influence but cannot recover their missing physics.
- The alignment image uses a bounded working scale while retaining the physical support and
  original scan data. An explicit `grid=` fixes sampling and can exceed device memory. Cropping
  support leaves attenuation outside it unmodelled.
- The full watch, spark-plug, K11 and PSI qualification of the consolidated solver is separate
  from its focused numerical tests. Earlier measurements in this repository describe their
  recorded implementations; they do not certify this solver's accuracy or default runtime.

## Detector-ring correction

`Rings` separates detector response from object structure statistically; it is
not a no-harm guarantee. On four real parallel-scan cuts, known injected
response fields retain 28–61% of their RMS after differential correction.
Independent off-axis analytic structure increments change by up to 11.1% on
I13 and 0.9–3.7% on the other three scans; see the
[ring measurements](measurements.md#detector-response-and-structure-controls).
On six native I13 rows, off-axis image increments change by 6.7–17.4%,
and a separate four-seed finite-reference test retains 48–83% of known added
image-error RMS; see the
[image-domain controls](measurements.md#calibration-backed-gaps-and-response-only-correction).
Further image-domain checks fail the 5% no-harm gate on ESRF bamboo (28.5%)
and Elettra (8.0%), despite sinogram-domain passes; see the
[cross-facility controls](measurements.md#cross-facility-image-domain-controls).
Review corrected and uncorrected structures, especially on moving samples.
`Rings` fits angular bin means of about six views each on a 3000-view scan;
pass `views=None` to fit every view when fine structure far from the axis
crosses several columns within a bin.
A centred radial feature and an even detector error are indistinguishable in
a stationary full turn: correction of that component depends on the method's
prior, not information uniquely supplied by the scan. Natural motion can
supply extra calibration constraints, but `Rings` does not fit those constraints.
`Rings` does not identify or interpolate dead pixels from the sinogram. Raw
calibration fills bracketed pixels without measured illumination; other defective
measurements require independent calibration evidence and repair before `Rings`.

## Data and geometry boundaries

- Projection-only centre reports apply to axial parallel scans. Pair intervals
  describe a selected rigid matching mode, not ambiguous peaks or sample motion;
  opposing shifts determine sums, not individual per-view motion. Row-centre
  slope is an apparent detector trend, not a calibrated three-dimensional axis
  tilt. Inspect the pair diagnostics when estimating a detector centre.
  In five low-contrast gVXR mesh cases, wrong-mode errors reached a 3.91 px
  p95 and the conditional 90% intervals covered only one truth; check competing
  modes against an independent calibration rather than treating a narrow local
  interval as a global accuracy certificate. Runtime reports mark global
  coverage uncalibrated for every estimate, including a single pair without
  a disagreement test; loading warns about this calibration limitation.
- `tomojax inspect` checks storage and missing calibration in recognized raw
  acquisitions. It does not run data-driven centre inference; a centre pending
  inference can still fail during loading. A valid report does not certify
  physical geometry or reconstruction readiness.
- External HDF5 links require readable backing files at their recorded paths.
  The loader reports missing detector sources; it does not repair damaged files
  or guess relocated links.
- Metadata-only files do not supply reconstruction observations. `tj.load`
  requires an acquisition containing detector projections, even when a separate
  metadata file describes its angles or instrument.
- Flat/dark correction needs at least one detector pixel with resolved
  illumination. If none exists, supply illuminated flats and matching darks,
  or include an illuminated region in the crop. Unilluminated margins and
  bracketed interior gaps remain supported; their correction does not certify
  that the supplied calibration describes the acquisition.


- Zeiss frames are numbered image streams, not the planned `NoOfImages` count;
  interrupted acquisitions load when their per-view metadata matches the stored
  frames. Metadata-only drift records contain no reconstructible projections,
  and missing interior image streams require a complete acquisition file.
- MATLAB ptychography files are not raw detector-count acquisitions. Prepare
  their complex transmission with the [ptychography example](../examples/align_ptychography.py)
  before constructing a `Scan`. Walnut collection folders require their vector
  geometry and calibration, as in the [walnut example](../examples/align_walnut_orbits.py).
  2DeteCT folders require explicit fan/cone geometry and flat/dark correction;
  the directory name does not supply that metadata to `load_frames`.
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
  poor pose observability. The alignment default uses linear interpolation;
  requesting any angular parameter uses cubic interpolation for parallel rays.
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
  higher derivatives. Joint alignment can fit detector centre, roll and a shared axis-tilt
  correction. It does not fit detector pitch or yaw, the source distances or arbitrary detector
  deformation. The
  [cone-setup alias measurements](measurements.md#cone-setup-aliases-and-a-conditional-calibration-test)
  show exact redundancies under unrestricted per-view poses; they do not
  provide uncertainty or identify a calibrated setup. A fixed-pose Walnut
  focal-distance test fails the 1% held-out correspondence improvement gate.
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
- Alignment fixes an object's position only up to what the data can see. Cone translations
  include along-beam motion, but the common rigid gauge remains unobservable. Compare geometry
  after one rigid gauge and include detector-centre changes in detector-space scores.
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

See [align a scan](align-a-scan.md) to choose the alignment parameters.

ESRF raw Bliss and SPEC EDF metadata do not establish a calibrated detector
centre: it must be estimated from data or supplied. SPEC scan range alone also
needs explicit `angles=`; accumulated darks without `DARK_N` need `darks=`.
An explicit PyHST `.par` supplies its own angles and rotation axis, but only
raw single-image EDFs with fixed axial parallel geometry are supported. PyHST
multi-frame, cone, helical, binning and distortion modes are rejected. Every
projection and calibration file named by the selected input must be available.
Cropping the detector preserves an explicit PyHST output grid; supply `grid=`
for a smaller preview or use slab reconstruction for a volume larger than memory.

An explicit file-owned reconstruction grid survives detector cropping. The
ESRF `scan_3_.par` fixture specifies 2048×2048×1200 voxels: one float32 volume
alone needs 20,132,659,200 bytes, above the 18 GB input-audit memory limit.
The 2026-10-27 audit on `2626e6e5` loaded it but stopped before reconstruction
(`TOMO_SOURCE_COMMIT=2626e6e5 JAX_PLATFORMS=cpu uv run --no-sync python
shared/agent06-audit/run_default.py 510`, with `LAB_ARTIFACTS` set to an output
directory).
Supply a smaller `grid=` to `tj.load_frames` when the file grid exceeds available
memory; a successful reduced preview is not a default full-volume result.

HDF5 virtual detector stacks require all referenced pixel files and mapped
regions to be present; missing sources are errors, not zero-filled exposures.
An unreadable external-root master needs its referenced files and datasets restored;
its link names and targets are reported when no detector stack can be found.

Nikon vendor reconstruction regions (including `OffsetX/Y/Z`) remain source
metadata and are not applied: their coordinate mapping is not established.
TomoJAX uses its own acquisition field-of-view grid unless `grid=` is supplied.
Missing detector offsets or axis calibration require explicit geometry.

## Ring-correction differentiation

Full-turn ring correction uses tiled CUDA kernels for its adaptive spatial
reference in float32. The kernels retain the weighted quadratic fits, robust
scatter, correlations and noise blending, but do not provide automatic
differentiation through that reference. CPU and float64 retain the JAX
implementation. CUDA fits use a weighted polynomial basis and retain its full
Gram matrix; they are mathematically equivalent, not bit-identical, to the
JAX normal-equation solves.
