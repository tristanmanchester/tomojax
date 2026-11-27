# tomojax.corrections

## Purpose

`tomojax.corrections` turns detector counts into line integrals and corrects
them. `tomojax.Frames.corrected(*steps)` runs any steps on counts, then
`(I - D) / (F - D)` with each view's flat blended from the flat sets taken
around it ([flat fields](#flat-fields)), then steps on transmission, the `-log`,
and steps on line integrals. `tomojax.Scan.corrected(*steps)` runs line-integral steps on a scan
that is already corrected. Views are corrected on the device a batch at a time.
Each step is recorded in `Scan.corrections` and saved with the scan.

HDF5 and TIFF sample and calibration frames stay lazy until correction, so
`Frames.cropped` reads only that detector window of all three. Unfiltered HDF5
frames (chunked or contiguous, including external links) are read at their stored
byte offsets with up to 32 parallel positional reads in blocks of at least 256 MiB,
with chunk offsets cached while the file size and modification time stay unchanged.
Flat and dark means accumulate in float64 across row bands on up to 8 CPU threads,
corrected view batches copy into the host output across up to 8 threads, and the
active bracketing flat sets stay on the device while consecutive batches share them.
A pixel whose flat does
not exceed the dark by `epsilon` and three standard errors of their means (or
whose calibration is not finite) is unmeasured. Interior gaps are interpolated
in transmission along the same detector row in each view, before phase retrieval
or the log; pixels outside the row's measured span take no absorption. The
`flat_dark` record counts unmeasured pixels. Interpolation cannot recover features
narrower than the missing span.

`Frames.corrected(..., binning=f)` averages blocks of the corrected line
integrals on the device, so no unbinned stack reaches the host. When only
whole-row steps (`Rings`) follow the `-log` and the source reads row windows
cheaply (NumPy, uncompressed TIFF, unfiltered HDF5), it corrects slabs of rows
read straight from the source; otherwise the steps that need full frames run
unbinned first.

`tomojax correct` is a thin layer over these: its options build the same
steps.

## Public API

Steps, each a frozen dataclass of its settings:

| Step | Acts on | Settings | `tomojax correct` option |
| --- | --- | --- | --- |
| `Zingers(threshold=0.1, size=3)` | transmission | specks brighter than their neighbors' median by `threshold` (a fraction of the open beam) | `--zingers [THRESHOLD]` |
| `RadialDistortion(center, coefficients)` | transmission | calibrated backward radial polynomial in zero-based `(column, row)` pixels, as in Algotom; bilinear, on normalized transmission before the log | Python only |
| `Paganin(delta_beta, distance, energy_kev, pixel_size)` | transmission | single-material phase retrieval; distance and pixel size at the sample in meters, energy in keV | `--paganin DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL` (expert) |
| `Rings(geometry=None, *, views=512)` | line integrals | fine and wide column offsets and transmission-dependent responses; binds geometry automatically in `Frames.corrected` and `Scan.corrected` | `--rings` |
| `RejectViews(z=6.0)` | line integrals | drops the views between an abrupt step in the mean line integral and the step back (a beam loss, a missed frame), and their geometry; smooth changes of the object are kept | `--reject-views [Z]` |
| `BeamHardening(rates, weights, energy_kev=None)` | line integrals | single-material linearization, see below | Python only |

Also exported:

- `Step`: the base class. A step has a `domain` (`"counts"`, `"transmission"`
  or `"line integrals"`) and an `apply` on a `(views, rows, columns)` batch, or,
  for a step that selects views, a `statistic` and a `select`.
- `Correction`: the record of one step, its `settings` and what it `found`.
- `correct_frames` and `correct_projections`: the array-level engines behind
  `Frames.corrected` and `Scan.corrected`, returning a `Corrected` (the line
  integrals, the records and the indices of the views kept).

See the [quickstart](../../../docs/quickstart.md#correct-raw-detector-frames)
for loading frames and choosing steps.

## Flat fields

A view recorded between two flat sets takes a blend of them. The views
themselves set the weights: no step or option turns this on.

Two flat sets differ either because the beam or the detector changed between
them, or because one of them does not show what the views saw (part of the
sample stage stood in the beam while it was taken, say). In the first case the
right flat moves from one set to the other through the scan. In the second it
does not, and blending by position prints the difference into every view with a
weight that changes through the scan, which no ring correction removes.

For each pair of neighboring sets, TomoJAX reads about one view in 64 (at
least 64) and measures, in cells of 16 detector pixels, how much of the
difference between the two sets each view contains. The measurement is a
regression on the difference's structure along detector rows, which is
unrelated to the sample. A straight line through the interval is fitted to
those shares in regions 144 pixels across, with linear interpolation in
position as the prior, and then in windows of 48 pixels, which are held to
their region's line unless they differ from it by more than their own
uncertainty.

- **Sets that differ by noise, by level, or smoothly along a row** tell the
  views nothing, and are interpolated in position exactly as before. The sets'
  own noise is subtracted, and a difference counts only as far as it stands
  three standard errors above that noise.
- **Structure that changes through the scan** (a detector response that
  drifts) is followed through it: the views show none of the later set at the
  start and all of it at the end.
- **A set that does not match the views** is left out where it does not match.
- **Where neither set matches**, the blend is fixed through the scan. What is
  left is then fixed to detector columns, so `Rings()` can remove its fine part.
- **Level** is kept continuous along each row where the weights change. An
  edge that only one set has is not in the views, so the flat carries its level
  across the edge instead of stepping there.

The `flat_dark` record reports what was found (the last two only when the
views set a weight somewhere):

| Key | Meaning |
| --- | --- |
| `flat_weights_from_views` | Fraction of the detector where the views, not position, set the weights |
| `later_flat_weight` | The later set's weight at the start and at the end of the interval, averaged where the views set it (`[0, 1]` is linear interpolation) |
| `flat_structure_unmatched` | RMS of the sets' fine structure that the views show in neither set, in line-integral units |

The views carry no measurement of the beam's broad level where they show no
unattenuated beam, so a region that both sets cover with something the views
do not show keeps a broad error beyond what continuity recovers. When
`flat_structure_unmatched` is large, flats from another scan of the same beam
are better: `tj.load_frames("scan.nxs", flats="other_scan.nxs")` or
`tomojax correct scan.nxs --flats other_scan.nxs` takes the other file's flat
frames (and `darks=` its dark frames) as one set.

A set of one frame has no measured noise, and scans with one flat set or a
white level have nothing to blend: both are corrected as before.

## Ring removal

`Rings()` is the library's ring-artifact correction (`tomojax correct --rings` on
the CLI). It takes no width or tuning parameters. `Frames.corrected`,
`Scan.corrected`, `tj.reconstruct_slabs`, and `tomojax correct --rings` bind the
scan's geometry automatically when `geometry=None`, and `tomojax correct --rings`
runs with the selected and cropped geometry before view rejection and detector
binning.

`Rings()` models each detector column as reporting `q + a + b (1 - exp(-q))`
instead of the true line integral `q`, where `a` is an additive offset and `b`
is a transmission-dependent response. After regularization, a constrained refit
keeps every observation on an invertible branch with response derivative at
least 0.5 so corrected line integrals are never clipped.

How `Rings()` separates detector errors from the sample depends on the scan
geometry:

- **Half-turn, non-parallel, or non-uniform scans:** a local quadratic fit across
  neighboring columns estimates fine column offsets and transmission slopes using
  independent neighboring views as an instrument. Ranked transmission profiles
  across angular arcs and persistent jumps between adjacent columns estimate
  wider offsets without a fixed window width. Variation across disjoint angular
  arcs weights both terms.
- **Stationary parallel-beam full turns (evenly sampled 360°):** opposed columns
  measure the same rays through the sample. `Rings()` compares short angular arcs
  half a turn apart at reflected detector coordinates (with cubic interpolation at
  the physical rotation axis) to estimate the detector power available for broader
  offsets and transmission slopes. Where reflected neighborhoods overlap near the
  axis, the offset weights account for the shared columns. A weighted median over
  locally smooth views refines the column intercept.

`Rings()` removes stationary column errors only. Artifacts that vary with view
angle, such as flux drops or angular swirls, are not fixed to detector columns
and stay in the reconstruction; drop affected frames with `RejectViews`. Flat
sets that disagree are weighted from the views before the log
([flat fields](#flat-fields)), which leaves a column-fixed error for `Rings()`.
`Rings()` does not detect or interpolate dead pixels (flat/dark calibration interpolates bracketed unmeasured pixels;
other bad pixels need repair beforehand). A circular sample feature centered on
the rotation axis produces the same parallel-beam projections as a symmetric
detector offset and cannot be separated without external calibration. Fine
stationary object structure and broad detector defects can also remain ambiguous.

By default `Rings()` fits on the means of about `views=512` contiguous angular
bins and applies the fitted inverse to every view. Pass `views=None` to fit
every view directly when sample structure moves across columns within a bin. On
an 8,000-view 2160 × 2560 streamed Fourier reconstruction on one V100S, `Rings()`
adds about 10 s on a half turn and 20 s on a full turn, with ~15 MiB extra GPU
memory ([ring measurements](../../../docs/measurements.md#ring-correction)).
Because it acts on whole detector rows, it runs inside `tj.reconstruct_slabs`
and fused row-slab correction without holding an unbinned stack in host memory.

Built-in angular geometries order views by angle within each trajectory and keep
cone segments separate. Without such geometry, supply views in continuous
acquisition order with at least two views per trajectory. The step records
`mirror_axis` (the physical axis column on a full-turn parallel scan, or `None`)
and `views` in `Scan.corrections`.

## Beam hardening

A polychromatic beam hardens as it crosses material, so thick paths look less
attenuating than they are: cupping and dark streaks between dense features.
Through one material the measured line integral is
`q = g(p) = -ln sum_j w_j exp(-k_j p)`, where `p` is the material's line
integral and the mixture of rates `k_j` with weights `w_j` is the detected
spectrum seen through the material. Every curve one material can make has this
form, and every such curve is increasing, concave and linear at large `p`.
`BeamHardening` replaces each `q` by `p = g^-1(q)`, through a tabulated inverse
accurate to float32.

The units of `p` are declared: with `energy_kev=None` the curve's slope at zero
is one, so thin paths keep their attenuation and the corrected image is in the
unhardened beam's mean attenuation; with `energy_kev` the rates are relative
to the material's attenuation at that energy, and so is the image.

The curve comes from an explicit polynomial or spectrum, each only as good as
the tube settings and sample material it is given: the scan itself is not used
to check it, so a curve computed for the wrong spectrum or material makes the image
worse rather than better.

Nikon coefficients, Zeiss constants and Bruker percentages are retained unapplied
in `scan.source.geometry_metadata["beam_hardening"]`, alongside tube settings.
Their vendor intensity domains and scaling conventions are not established.

- `BeamHardening.from_polynomial(coefficients, upto=q_max)` turns an explicit
  polynomial linearization `p = c0 + c1 q + c2 q^2 + ...` into the closest
  physical curve over `0 <= q <= upto`.
- `BeamHardening.from_spectrum(energies, weights, material)` builds the curve of
  a sample material (a chemical formula or NIST name) seen by a detected
  spectrum, with attenuation from xraylib (`pip install 'tomojax[spectra]'`).
  `tube_spectrum(kv, filters=..., detector=...)` models a tungsten-target tube's
  spectrum behind its filters, as detected by a scintillator:

  ```python
  from tomojax.corrections import BeamHardening, tube_spectrum

  energies, weights = tube_spectrum(
      90, filters=[("Be", 0.3), ("Al", 0.5), ("Cu", 0.2)], detector=("CsI", 0.6, 4.51)
  )
  step = BeamHardening.from_spectrum(energies, weights, "Al", energy_kev=60)
  ```

A second material, scatter and photon starvation break the single-material
model; one curve cannot correct them. `step.curve(p)` and
`step.inverse(q)` evaluate the curve on the host for plotting it against data.

## Complex transmission images

For ptychography or holotomography, `unwrap_phase(complex_images)` integrates
wrapped phase gradients with nonperiodic boundaries on each image. It returns
zero-mean radians on the last two axes, `(..., rows, columns)`. Then
`remove_phase_ramp(phase, air_mask=air)` subtracts a plane fitted to known air
pixels; the boolean mask or nonnegative weights may be shared or per view.
Crop unreliable probe borders first. Air must exclude the sample, and phase
residues produce a least-squares integral rather than an exact unwrap. Both
functions return JAX arrays and compose under `jax.jit`; process large stacks
in batches of views.

On a grid measured in detector pixels, convert the corrected phase to line
integrals of the refractive-index decrement with
`-phase * wavelength / (2 * np.pi * pixel_size)`, then build a `tj.Scan`.
For a grid in physical length units, omit the division by `pixel_size`.
Wavelength, pixel size and grid lengths must use consistent units. For joint
alignment, keep `-phase` in radians on a unit-pixel grid and apply the same
conversion factor to the reconstructed volume afterwards, keeping the solver
data near unit scale.

## Dependencies

Allowed: `tomojax._data`, `tomojax.recon`, `tomojax.forward`,
`tomojax.backends`, `tomojax.nuisance`, `tomojax.motion`, `tomojax.geometry`,
`tomojax.core`, JAX and NumPy.

Forbidden: `tomojax.io`, `tomojax.datasets`, `tomojax.alignment`, the CLI and
the workflow modules, which sit above it in the import layers
([`.importlinter`](../../../.importlinter)).

## Invariants

- Steps run by domain, counts first and line integrals last, each in the order
  given within its domain.
- Every step applied is recorded, with its settings, in the scan it produced.
- A step that drops views also drops their geometry; selected views keep their
  place among the flat fields.

## Tests

- `tests/test_beam_hardening.py` covers the curve and its inverse, each
  coefficient source and the vendor records.
- `tests/test_corrections.py` covers loading frames (NXtomo, other HDF5
  layouts, APS Data Exchange, Nikon, TIFF), flat interpolation, each step,
  view selection and cropping, and the saved records.
- `tests/test_radial_distortion.py` checks the lens map against SciPy's
  interpolation.
- `tests/test_correction_medians.py` checks zinger medians against SciPy and wide-offset recovery under moving features.
- `tests/test_correction_batch_memory.py` covers automatic batch sizing for
  phase retrieval.
