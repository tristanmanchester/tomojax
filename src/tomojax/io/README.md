# tomojax.io

`tomojax.io` handles dataset loading/saving, NXtomo validation, TIFF/NX
preprocessing, inspection reports, quicklooks, and metadata conversion.

Import from here, not from `tomojax._data`.

## TIFF angle sidecars

`tomojax.io.api.load_angles` reads a one-dimensional NPY vector or the first
column of text/CSV in degrees, preserving acquisition order. Ingestion and TIFF
preprocessing use this same parser. It accepts leading headers, blank lines,
and comments; empty/nonfinite data and malformed rows after numeric data starts
raise `ValueError`. The caller checks that angle count matches projection count.

The ingestion CLI accepts detector-centre offsets in pixels and converts them
with `du`/`dv`. Python `Detector.center` stores physical lengths. See the
[real scan guide](../../../docs/real-laminography.md) for measured geometry and
for the distinction between raw intensities and corrected absorption.
