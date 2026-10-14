# tomojax.corrections

## Purpose

`tomojax.corrections` turns detector counts into line integrals and corrects
them. `tomojax.Frames.corrected(*steps)` runs any steps on counts, then
`(I - D) / (F - D)` with each view's flat interpolated between the flat sets
taken around it, then steps on transmission, the `-log`, and steps on line
integrals. `tomojax.Scan.corrected(*steps)` runs line-integral steps on a scan
that is already corrected. Views are corrected on the device a batch at a time.
HDF5 sample and calibration frames stay lazy until correction, so
`Frames.cropped` reads only that detector window for all three. Calibration
means accumulate in FP64 in bounded host batches. `np.asarray(frames.flats)`
or indexing the flat/dark stack reads it explicitly.
Each step is recorded in `Scan.corrections` and saved with the scan.
For a device-backed `Scan`, an initial chain of per-view line-integral steps
runs on its existing device storage when the producer devices have known
workspace headroom, before bounded corrected batches come back to the host.
Automatic batches reserve half that headroom. Unknown budgets, insufficient room
for one view, or oversized explicit batches retain the original host-first route;
explicit batches are not silently reduced. The result remains independent, writable float32 host storage;
whole-row steps and view selectors still see the complete stored stack.
The log record counts nonpositive transmission after the transmission steps,
including any undershoot introduced by phase retrieval, not before filtering.

Nonfinite calibrations and flat-minus-dark signals that are nonpositive or
unresolved at three standard errors of their calibration mean are unmeasured
pixels. They take neutral
transmission (one, hence zero line integral) before spatial corrections, rather
than dividing their counts by an epsilon. The flat-field record reports the
affected pixels. This avoids spreading invalid detector values through phase
retrieval; it does not recover absorption at those pixels.
At an interpolation endpoint, only the calibration with nonzero weight is
used; a missing value in an unused neighbouring flat does not discard a valid ray.

Integer calibration sets fixed at their dtype's maximum value are warned about
and reported as `flat_pixels_at_dtype_ceiling` or `dark_pixels_at_dtype_ceiling`.
These counts include each flat set separately. Zero variance at that ceiling is
not evidence for a well-measured illumination: clipping or stuck bright pixels
can still create severe artifacts. This diagnostic does not change the pixels
or recover their missing measurements; use an unsaturated calibration or a
validated detector-defect repair. Floating calibrations and lower ADC limits
within an integer storage type cannot be diagnosed from dtype alone.

Integer projection counts at their storage ceiling are diagnosed separately:
the log record reports `counts_at_dtype_ceiling` and its `counts_dtype_ceiling`.
The tally covers the corrected input readings before counts steps or view
rejection, and reuses each already-read batch. A healthy flat does not exclude
clipped sample readings, for example at bright phase-contrast fringes. The
warning does not replace, mask, or clip those readings; transmission above one
can also be physically valid. Floating counts and lower ADC limits within a
wider storage type cannot be diagnosed from dtype alone.

Spatial steps need measured neighbouring pixels. Prefer cropping the corrected
scan after Paganin: its FFT edge padding cannot restore measurements discarded
by cropping raw frames. For a bounded raw window, correct an enlarged window
and retain the interior, checking convergence as the window grows.

`RejectViews` detects local jumps, not arbitrary beam interruptions. Bad views
occupying a majority of its rolling neighbourhood can pass undetected (for
example, three adjacent shuttered views with the default five-view window).
Bad neighbours can also bias the baseline and reject healthy views. Inspect
the rejected indices, and use `Frames.selected(valid_views)` for a known
acquisition mask before correction. Increasing `neighbours` may expose longer
runs but needs checking against genuine changes with angle; it is not a
guarantee of the correct mask.

For scans whose corrected stack is too large for host RAM, `Frames.corrected`
and `Scan.corrected` accept `out=`: a writable float32 NumPy array or memmap
with the input shape, separate from all inputs (including calibrations).
Corrections write into this caller-owned storage; view rejection compacts
kept views into its beginning using bounded batches. The returned projections
share it, and may have fewer views than the original buffer. The caller owns
the file's lifetime and flushing; buffer contents are unspecified if a
correction fails. When both an input and output are memory-mapped, they need
separate named files: distinct mappings of an unnamed file cannot be checked
portably for shared storage. See the quickstart for a memmap example.

`tomojax preprocess` is a thin layer over these: its options build the same
steps.

## Public API

Steps, each a frozen dataclass of its settings:

| Step | Acts on | Settings | `tomojax preprocess` option |
| --- | --- | --- | --- |
| `Zingers(threshold=0.1, size=3)` | transmission | specks brighter than their neighbours' median by `threshold` (a fraction of the open beam) | `--zingers [THRESHOLD]` |
| `RadialDistortion(center=..., coefficients=...)` | transmission | calibrated backward radial polynomial, in zero-based `(column, row)` pixel coordinates | Python only |
| `Paganin(delta_beta, distance, energy_kev, pixel_size)` | transmission | single-material phase retrieval; distance and pixel size at the sample in metres; scalar pitch or `(u, v)` pitches; energy in keV | `--paganin DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL_U[,PIXEL_V]` (expert) |
| `Stripes(width=9)` | line integrals | ring removal: each pixel's offset against a median over `width` columns | `--remove-stripes WIDTH` |
| `RejectViews(z=6.0, neighbours=5, min_jump=0.05)` | line integrals | drops views whose median jumps from their neighbours', and their geometry | `--reject-outliers [Z]` |
| `BeamHardening(coefficients=(1.0,))` | line integrals | linearisation `p -> c1 p + c2 p^2 + ...` | `--beam-hardening C1,C2,...` |

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

## Calibration limits and missing pixels

Clipped counts do not retain intensity above the sensor's range. A binned
flat below the encoding ceiling does not prove that every native pixel was
in range: averaging can conceal clipped subpixels. Retain native calibration
frames for this check; binning does not recover the lost measurements.

Interpolating a declared missing-pixel mask is a continuity prior, not a
recovered measurement. It can reduce a visible ring while removing real
narrow features. A smoother image or better reference score alone does not
establish that those features were preserved.

Track missing counts in sensor coordinates. Spatial remapping and phase
retrieval mix input pixels, so errors from a missing input can affect apparently
healthy output pixels. Masking only the final line integrals does not undo
that mixing; keep the missing-data and interpolation assumptions explicit.

## Choosing ring removal

`Stripes` estimates a detector-fixed offset in line integrals and subtracts it
from every view. A residual multiplicative gain error has this form after the
logarithm; a raw additive count bias, such as an incorrect dark field, does not.
It preserves each pixel's differences between views, but a
real feature symmetric about the rotation axis also looks stationary in a
sinogram. Ring removal can therefore alter genuine centred features, especially
ones narrower than `width`; defect-free data are not guaranteed to be unchanged.
Use the smallest window that covers the detector defect, and compare corrected
and uncorrected reconstructions when centred fine structure matters. Completely
dead pixels or view-dependent defects need more than a constant-offset model.


## Calibrated lens distortion

```python
import tomojax as tj
from tomojax.corrections import RadialDistortion, Stripes

scan = tj.load_frames(path).corrected(
    RadialDistortion(center=(1230.99, 1018.87), coefficients=(1.00016, -9.83e-7)),
    Stripes(),
)
```

Supply the full fitted **backward** polynomial, not forward-model coefficients.
Its convention matches [Algotom's distortion correction](https://algotom.readthedocs.io/en/latest/toc/api/algotom.prep.correction.html#algotom.prep.correction.unwarp_projection).
For an output pixel with displacement `d` from `center` and radius `r`, the
measured source coordinate is `center + d * sum(c_k * r**k)`. Bilinear
interpolation acts on flat/dark-normalized transmission before the log; it is
not equivalent to warping the line integrals or separately warping raw counts
and calibration fields. The detector shape and geometry are retained, with
nearest-edge extension beyond the measured image and no Jacobian factor.
Distortion correction does not infer data outside the acquired detector or
find the rotation axis; estimate the axis on the corrected scan.

When combining spatial transmission corrections, remove zingers first, then
unwarp the lens, then retrieve phase with `Paganin`. The phase filter assumes
uniformly spaced pixels; lens distortion makes their spacing nonuniform in
the measured image. These operations do not commute. Steps within the
transmission domain run in the order supplied; TomoJAX does not reorder them.

Calibration coordinates must describe the frames being corrected. After a
crop starting at `(column0, row0)`, subtract that pair from the centre. After
mean binning by `f`, use `(center - (f - 1)/2)/f` and coefficients
`c_k * f**k`: a binned pixel lies at the mean of its source pixel centres.
Correct before cropping if the map requires source pixels outside the crop,
or retain a sufficient source halo and crop the corrected result afterwards.

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

- `tests/test_corrections.py` covers loading frames (NXtomo, other HDF5
  layouts, APS Data Exchange, Nikon, TIFF), flat interpolation, each step,
  view selection and cropping, and the saved records.
- `tests/test_correction_medians.py` checks the zinger and stripe medians
  against SciPy and NumPy references.
- `tests/test_correction_batch_memory.py` covers automatic batch sizing for
  phase retrieval.
- `tests/test_correction_output_storage.py` checks caller-owned array/memmap
  storage, bounded view compaction, and input/output separation.


- `tests/test_radial_distortion.py` checks identity, bilinear interpolation
  against SciPy, crop/binning coordinates, domain ordering and saved records.
