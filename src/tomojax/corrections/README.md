# tomojax.corrections

## Purpose

`tomojax.corrections` turns detector counts into line integrals and corrects
them. `tomojax.Frames.corrected(*steps)` runs any steps on counts, then
`(I - D) / (F - D)` with each view's flat interpolated between the flat sets
taken around it, then steps on transmission, the `-log`, and steps on line
integrals. `tomojax.Scan.corrected(*steps)` runs line-integral steps on a scan
that is already corrected. Views are corrected on the device a batch at a time.
Each step is recorded in `Scan.corrections` and saved with the scan.
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

`tomojax preprocess` is a thin layer over these: its options build the same
steps.

## Public API

Steps, each a frozen dataclass of its settings:

| Step | Acts on | Settings | `tomojax preprocess` option |
| --- | --- | --- | --- |
| `Zingers(threshold=0.1, size=3)` | transmission | specks brighter than their neighbours' median by `threshold` (a fraction of the open beam) | `--zingers [THRESHOLD]` |
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
