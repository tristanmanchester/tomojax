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

## Diamond I13 mixed TIFF folder

This example uses an unpublished Diamond I13 TIFF export (scan 159815), not a
publicly downloadable test dataset. It has LZW-compressed TIFFs in one `raw/` directory:
100 darks, 100 flats, 3001 projections, then 100 flats. `savu_processed.nxs`
records the projection angles and the processing calibration. Generic TIFFs
alone do not identify which images are flats or supply angles and geometry.
For another such export, replace the filename ranges, angle dataset and sign,
detector dimensions/centre, and row/column crop with its own processing record;
the numbers below are not I13-wide defaults.

This Linux example creates temporary **symlinks**, not copies of the images.
Keep the temporary directories alive until correction has materialised the
selected row. The recorded Savu crop is columns 441:1872, with axis column
718.5 in that crop: original axis column 1159.5, hence detector offset 120.
Savu's angle convention is opposite to TomoJAX's world-from-object rotations.
Neither these TIFFs nor this Savu export establishes a physical pixel pitch;
this preview deliberately uses **pixel units**, not millimetres or micrometres.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import tomojax as tj

root = Path("diamond-i13-159815").resolve()
with h5py.File(root / "savu_processed.nxs") as source:
    angles = -np.asarray(source["entry/input_data/tomo/rotation_angle"])

with TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    selections = {
        "projections": range(200, 3201),
        "flats": [*range(100, 200), *range(3201, 3301)],
        "darks": range(100),
    }
    for name, indices in selections.items():
        folder = temporary / name
        folder.mkdir()
        for index in indices:
            (folder / f"proj_{index}.tif").symlink_to(root / "raw" / f"proj_{index}.tif")
    frames = tj.load_frames(
        temporary / "projections",
        angles=angles,
        detector=tj.Detector(2560, 2160, 1, 1, center=(120, 0)),
        flats=temporary / "flats",
        darks=temporary / "darks",
    )
    scan = frames.cropped(slice(1080, 1081), slice(441, 1872)).corrected()
    result = tj.reconstruct(scan)
    tj.save("i13-preview.nxs", result)
```

Save this complete block as `i13_preview.py` and run it:

```bash
python i13_preview.py
tomojax inspect i13-preview.nxs --preview i13-preview-images
```

This is the default reconstruction, not a
reproduction of Savu's ring-removal/filtering pipeline. Keep those distinctions
when comparing images. For ordinary separated TIFF folders and angle sidecars,
see [preparing TIFF data](real-laminography.md#prepare-tiff-data).
