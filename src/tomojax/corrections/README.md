# tomojax.corrections

## Purpose

`tomojax.corrections` turns detector counts into line integrals and corrects
them. `tomojax.Frames.corrected(*steps)` runs any steps on counts, then
`(I - D) / (F - D)` with each view's flat interpolated between the flat sets
taken around it, then steps on transmission, the `-log`, and steps on line
integrals. `tomojax.Scan.corrected(*steps)` runs line-integral steps on a scan
that is already corrected. Views are corrected on the device a batch at a time.
Each step is recorded in `Scan.corrections` and saved with the scan.

HDF5 and TIFF sample and calibration frames stay lazy until correction, so
`Frames.cropped` reads only that detector window of all three. Flat and dark
means are accumulated in float64 over bounded batches. A pixel whose flat does
not exceed the dark by `epsilon` and three standard errors of their means (or
whose calibration is not finite) is unmeasured. Interior gaps are interpolated
in transmission along the same detector row in each view, before phase retrieval
or the log; pixels outside the row's measured span take no absorption. The
`flat_dark` record counts unmeasured pixels. Interpolation cannot recover features
narrower than the missing span.

`Frames.corrected(..., binning=f)` averages blocks of the corrected line
integrals on the device, so no unbinned stack reaches the host. When only
whole-row steps (`Stripes`) follow the `-log` and the source reads row windows
cheaply (NumPy, uncompressed TIFF, unfiltered HDF5), it corrects slabs of rows
read straight from the source; otherwise the steps that need full frames run
unbinned first.

`tomojax preprocess` is a thin layer over these: its options build the same
steps.

## Public API

Steps, each a frozen dataclass of its settings:

| Step | Acts on | Settings | `tomojax preprocess` option |
| --- | --- | --- | --- |
| `Zingers(threshold=0.1, size=3)` | transmission | specks brighter than their neighbours' median by `threshold` (a fraction of the open beam) | `--zingers [THRESHOLD]` |
| `RadialDistortion(center, coefficients)` | transmission | calibrated backward radial polynomial in zero-based `(column, row)` pixels, as in Algotom; bilinear, on normalised transmission before the log | Python only |
| `Paganin(delta_beta, distance, energy_kev, pixel_size)` | transmission | single-material phase retrieval; distance and pixel size at the sample in metres, energy in keV | `--paganin DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL` (expert) |
| `Stripes(width=9, varying=False)` | line integrals | constant pixel offsets, or selected intensity-dependent responses with `varying=True` | `--stripes WIDTH` (constant offsets) |
| `RejectViews(z=6.0, neighbours=5, min_jump=0.05)` | line integrals | drops views whose median jumps from their neighbours', and their geometry | `--reject-views [Z]` |
| `BeamHardening(rates, weights, energy_kev=None)` | line integrals | single-material linearisation, see below | `--beam-hardening recorded` |

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

`Stripes(..., varying=True)` fits gain and offset only at pixels whose
view-to-view differences are anomalous. Unselected centred features are
preserved, but their constant-offset stripes are left too; sparse views or
sharp moving edges can cause false detections. Selected dead pixels receive
only the neighbouring mean; their missing angular information is not recovered.

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

The curve comes from one of three sources, each only as good as the tube
settings and sample material it is given: the scan itself is not used to check
it, so a curve computed for the wrong spectrum or material makes the image
worse rather than better.

- `BeamHardening.recorded(scan)` applies the linearisation the vendor recorded:
  Nikon CT Pro's `CoefX0`-`CoefX4` polynomial, as the closest physical curve.
  Zeiss's beam hardening constant and Bruker's percentage are recorded with the
  scan (`scan.source.geometry_metadata["beam_hardening"]`, with the tube voltage
  and filter) but not applied, because their formulas are unpublished.
- `BeamHardening.from_polynomial(coefficients, upto=q_max)` turns any
  polynomial linearisation `p = c0 + c1 q + c2 q^2 + ...` into the closest
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
- `tests/test_correction_medians.py` checks the zinger and stripe medians
  against SciPy and NumPy references.
- `tests/test_correction_batch_memory.py` covers automatic batch sizing for
  phase retrieval.
