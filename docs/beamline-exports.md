# First slices from beamline exports

These recipes use the public Python API with an [installed TomoJAX environment](installation.md).
Run Python and `tomojax` from that environment; a source checkout is not required.
If your environment is `.venv`, activate it with `source .venv/bin/activate`.
Start with a single detector row so the preview does not allocate a full volume.
The examples are **raw counts**, not already-log-transformed projections.

The calibration numbers below belong to the named acquisitions. For another
scan, use its own recorded calibration; do not copy a plausible pixel size or
rotation centre. File recognition and physical calibration are separate: an
otherwise recognised file can still need `detector=` or `angles=`.

## ESRF NXtomo export

For the bamboo scan exported from Bliss by nxtomomill, use nabu's
[public test file](http://www.silx.org/pub/nabu/data/bamboo_reduced_bliss.nx).
This server currently has an invalid TLS certificate; the supplied link is HTTP.
Do not disable certificate verification for other downloads.

```python
import tomojax as tj

frames = tj.load_frames("bamboo_reduced_bliss.nx")
```

The reader finds the group declaring `NXtomo`, regardless of the entry's name,
and reads its sample angles, image keys, flat/dark fields and unit-labelled
pixel pitches. If the axial centre is absent, loading estimates it from
corrected near-opposite views and records that it was inferred; an explicit
calibrated `detector=` overrides this estimate.

Continue with the [one-row preview](#correct-reconstruct-and-inspect) below.
An ESRF NXtomo export is not a raw Bliss master or an EDF folder. Those have
[different metadata requirements](quickstart.md#raw-esrf-acquisitions);
complete real-scan validation currently covers the NXtomo exports, not every
native ESRF layout.

## PSI TOMCAT Data Exchange export

The [TomoBank catalogue](https://tomobank.readthedocs.io/) identifies the
`tomo_00069` skin dataset (file `tomo_00069.h5`). It contains counts, flats, darks and
`/exchange/theta`, but no pixel pitch. Its accompanying calibration gives
3.7 µm pixels and rotation-axis column 515.5 on the 1024-column detector.
Supply that calibration explicitly, here in millimetres:

```python
import tomojax as tj

pitch = 3.7e-3  # millimetres; this value is not in the HDF5 file
axis_column = 515.5
frames = tj.load_frames(
    "tomo_00069.h5",
    detector=tj.Detector(
        1024, 428, pitch, pitch,
        center=((1023 / 2 - axis_column) * pitch, 0),
    ),
)
```

`Detector.center` is a physical offset, **not** an axis column index. For an
axial detector, its horizontal offset is `(nu - 1) / 2 - axis_column`, multiplied
by the pitch. The explicit detector is retained rather than automatically
re-estimated. A metadata-only inspection of the original file cannot supply
its missing pitch; inspect the calibrated preview saved below instead.

This reads the Data Exchange conversion, not TOMCAT's native GigaFRoST format.

## Elettra SYRMEP Data Exchange export

The [TomoBank catalogue](https://tomobank.readthedocs.io/) identifies the
`tomo_00024` polystyrene dataset (file `tomo_00024.h5`). It records its 2.04 µm pixel pitch.
The accompanying calibration gives rotation-axis column 972 on 2048 columns:

```python
import tomojax as tj

pitch = 2.04e-3  # millimetres
frames = tj.load_frames(
    "tomo_00024.h5",
    detector=tj.Detector(
        2048, 2048, pitch, pitch,
        center=((2047 / 2 - 972) * pitch, 0),
    ),
)
```

The reader uses the recorded `/exchange/theta` array, not the inconsistent
`angular_step` summary in this export. This is local tomography: the sample
extends beyond the field of view. A successful preview or centre estimate
does not remove truncation artefacts or establish quantitative attenuation.
This recipe does not establish support for every native SYRMEP layout.

## Correct, reconstruct and inspect

Save one of the three `frames = ...` snippets above in `preview.py` and append
the following block. It retains one native detector row, applies flat/dark correction and the logarithm,
then uses the default axial reconstruction:

```python
row = frames.counts.shape[1] // 2
scan = frames.cropped(slice(row, row + 1), slice(None)).corrected()
result = tj.reconstruct(scan)
tj.save("preview.nxs", result)
```

Use a new output path, then inspect it from the same environment:

```bash
python preview.py
tomojax inspect preview.nxs --preview preview-images
```

The saved file retains the calibrated geometry, corrected projections and
volume. The PNGs are display-scaled previews; use the floating-point data for
quantitative work. The Python round trip is:

```python
restored = tj.load_reconstruction("preview.nxs")
print(restored.volume.shape, restored.scan.grid, restored.scan.detector)
```

For rings, phase retrieval, larger row windows and explicit reconstruction
methods, continue with [the quickstart](quickstart.md). Do not apply another
logarithm to the saved corrected projections.

## Diamond I13 NXtomo acquisition

The unpublished Diamond I13 scan 159815 is a GDA NXtomo file,
`diamond-i13-mg31912-1/raw/159815.nxs`, linked to `pco1-159815.hdf` beside it.
Keep both files together. Loading discovers its 3001 sample views, 200 flats in
before/after sets, 100 darks and recorded angles without a TIFF export.
The detector pixel pitch is not recorded, so `tj.load` reports the missing
calibration: supply `detector=` through `tj.load_frames`.

This preview uses **pixel units**, not millimetres or micrometres. The Savu
processing record supplies the crop 441:1872 and axis column 718.5 within that
crop: original axis column 1159.5, hence detector offset 120. These are values
for this scan, not I13-wide defaults; replace them with your scan's calibration.
The acquisition's recorded angle signs are preserved. Negate them explicitly
only when comparing to a pipeline using the opposite rotation convention.

```python
import tomojax as tj

frames = tj.load_frames(
    "diamond-i13-mg31912-1/raw/159815.nxs",
    detector=tj.Detector(2560, 2160, 1, 1, center=(120, 0)),
)
scan = frames.cropped(slice(1080, 1081), slice(441, 1872)).corrected()
result = tj.reconstruct(scan)
tj.save("i13-preview.nxs", result)
```

Save this block as `i13_preview.py` and run it:

```bash
python i13_preview.py
tomojax inspect i13-preview.nxs --preview i13-preview-images
```

This is the default reconstruction, not a reproduction of Savu's
ring-removal/filtering pipeline. For TIFF-only acquisitions without NXtomo
metadata, see [preparing TIFF data](real-laminography.md#prepare-tiff-data).
