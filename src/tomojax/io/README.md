# tomojax.io

`tomojax.io` reads and writes TomoJAX datasets, reads Nikon `.xtekct` scans and
TIFF stacks, and validates and inspects datasets. Raw detector frames are read
with `tomojax.load_frames` and corrected with
[`tomojax.corrections`](../corrections/README.md).

Import from here, not from `tomojax._data`.

## HDF5 acquisition metadata

NXtomo angles are read from `/entry/sample/rotation_angle`, or from TomoJAX's
`/entry/sample/transformations/rotation_angle`; these take priority over
similarly named datasets of other detectors. Inspection and validation use the
same angle paths and units as loading, and an inspected angle range covers
only sample frames.

Both raw-frame and processed NXtomo loading convert rotation angles labelled
`rad`, `radian` or `radians` to degrees. `deg`, `degree`, `degrees` and missing
units mean degrees. Angles must be a real one-dimensional vector with one
entry per frame and finite sample-view angles (calibration frames may have
NaN angles); unknown units fail rather than being guessed. An
`image_key` must be an integer vector with one entry per frame and labels in
`{0, 1, 2, 3}` (sample, flat, dark, other); labels are checked before conversion.
For other HDF5 layouts, per-frame shape disambiguates named angle/key datasets
from different detectors. A sole named dataset with an invalid shape is rejected,
not treated as absent. Ambiguous candidates require an explicit metadata path.

`tomojax.load` of a processed file leaves out frames that `image_key` marks as
other than sample views. Saving a scan retains every pose update, including an
update around a cone-segmented acquisition whose segments have their own;
`Scan.poses` is their complete composition in the detector translation frame,
and loading with `poses=False` removes all of them.

## TIFF angle sidecars

TIFF frame and projection loading reads every top-level grayscale page of a
file, including separate series. Folders are naturally sorted by filename,
then pages retain their file order. All pages must be two-dimensional and the
same size; RGB/RGBA pages are not interpreted as extra detector views.

`tomojax.io.api.load_angles` reads a one-dimensional NPY vector or the first
column of text/CSV in degrees, preserving acquisition order. Ingestion and TIFF
preprocessing use this same parser. It accepts leading headers, blank lines,
and comments; empty/nonfinite data and malformed rows after numeric data starts
raise `ValueError`. The caller checks that angle count matches projection count.

`tomojax import` records a centred detector; `tomojax align --mode cor`
estimates the offset. Python `Detector.center` stores physical lengths, so an
offset in pixels is multiplied by `du`/`dv`. See the
[real scan guide](../../../docs/real-laminography.md) for measured geometry and
for the distinction between raw intensities and corrected absorption.
