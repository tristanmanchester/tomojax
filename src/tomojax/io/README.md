# tomojax.io

`tomojax.io` reads and writes TomoJAX datasets, exports registered volumes to
interchange formats (ImageJ TIFF, OME-TIFF, TIFF slices, raw binary, MetaImage,
and AmiraMesh via `tj.export`; see [export](../../../docs/export.md)), reads
Nikon, SkyScan, Zeiss and ESRF acquisitions, TIFF stacks and PSI cSAXS
ptychographic tomography `.mat` files, and validates and inspects datasets.
Detector frames are read with `tomojax.load_frames` and corrected with
[`tomojax.corrections`](../corrections/README.md).

Zeiss Xradia acquisitions are recognized by OLE content, with lazy raw counts,
embedded or explicitly selected XRM reference images and physical calibration;
fractional image shifts stay in the geometry rather than resampling the data. Per-view cone arrangements use
matched CGLS under automatic reconstruction, not single-arrangement FDK.
Use `tomojax correct scan.txrm --darks 0 -o scan.nxs`, or
`tj.load_frames("scan.txrm", darks=0).corrected(...)`; the
[lab-CT guide](../../../docs/lab-ct.md#zeiss-xradia-preview) gives a binned CPU preview.
Pass separate flat or dark images with `flats="reference_Front.xrm"` or `darks=...`.
Sample and flat exposure durations normalize dark-subtracted counts before
timestamp interpolation of multiple reference sets. Large OLE images retain only
allocation metadata and read the requested rows, without reopening their directory.

Import from here, not from `tomojax._data`.

For explicit HDF5 crops, `tomojax.io.api.read_hdf5_array(dataset, slices)`
preserves dtype and axes while decoding eligible gzip chunks in bounded parallel
batches; the caller retains ownership of the open dataset.

## Saved NeXus files

The root attribute `tomojax_format = 2` selects the native reader and
`default = "entry"` opens the NeXus entry. Its default plot is the volume,
working image, phantom or projection stack stored by the object.

| HDF5 path under `/entry` | Contents |
| --- | --- |
| `record` | Scalar UTF-8 JSON document |
| `title`, `sample/name` | Sample name |
| `instrument/detector` | Projection data, image keys and detector pitches |
| `instrument/source` | Recorded radiation source and probe |
| `sample/rotation_angle` | Rotation angles in degrees |
| `data` | Projection NXdata with links to detector data, rotation angles and image keys, plus detector coordinates `v` and `u` |
| `volume`, `image`, `phantom` | NXdata with a `data` signal and voxel-center coordinates `z`, `y`, `x` |
| `tomojax` | Effective geometry parameters, reported alignment poses, simulation truth poses and angle offsets when available |

An axis starts at the center of element zero. Recorded spatial units apply to
grid and detector coordinates. Uncalibrated coordinates use `scan_length`.
Projection datasets written by `tj.save` contain corrected line integrals.
Readers that import raw NXtomo counts must skip their flat-field and log steps.
A low-level save of detector counts labels the signal `counts`.

| Record field | Meaning |
| --- | --- |
| `format`, `kind`, `length_unit` | Layout version, saved object type and spatial unit |
| `geometry` | Effective grid, detector, geometry type, corrections and acquisition metadata |
| `arrays` | Member paths, shape, dtype, axes, units and measured quantity |
| `tables` | Paths to angles, frame labels, poses and offsets, with units and pose columns |
| `alignment`, `calibration`, `simulation` | Pose gauge, geometry calibration and simulation settings |
| `sample`, `source`, `frame` | Sample name, radiation source and coordinate frame |
| `provenance`, `history` | Writer versions and time, available command or worker context and provenance of earlier saves |

`tables.alignment_poses` gives each fitted view's detector-frame result, with
rotations in degrees. A center-only fit reports zero per-view corrections.
`tables.poses` holds the parameters applied to the scan geometry, with
rotations in radians and translations in the recorded frame.

When a reconstruction and its attached scan use different voxel grids,
`geometry.grid` describes the scan and `geometry.meta.volume_grid` describes
the reconstructed volume. The volume's NeXus axes use its own grid.

Readers skip additional members and record fields of a supported format.
An unrecognized format number reports the writing version. Re-save rc6 datasets with the format-2 migration release (build `4378e691`) before upgrading. To migrate a reconstruction while
preserving its optional projections, load it once and pass
`include_scan=result.scan is not None` to `tj.save`.

## HDF5 acquisition metadata

NXtomo angles are read from `/entry/sample/rotation_angle`, or from TomoJAX's
`/entry/sample/transformations/rotation_angle`; these take priority over
similarly named datasets of other detectors. When a GDA NXtomo entry omits its
angles, TomoJAX reads the SmarAct `zrot` or `psi` rotation command for the same
frame dataset, records the angle source, and warns if that recognized command
carries the positioner's `"mm"` unit label (reading it in degrees). Inspection and validation use the
same angle paths and units as loading, and an inspected angle range covers
only sample frames. `inspect_dataset`, previews and `validate_dataset` also use
raw-frame discovery for APS Data Exchange, Diamond/GDA and uniquely named HDF5
stacks. Optional NXtomo labels are not needed; `validate_nxtomo` remains the
strict NXtomo schema check.

Validation of recognized raw acquisitions reports missing detector/axis metadata
and flat/dark references, using the same requirements as frame loading.
`tj.inspect(path)` includes a `loading` section (`LoadingReport`: `full_turn`,
`axis_to_edge_px`, `field_of_view_px`, `overlap_px`, `central_gap_px`,
`length_unit`, `center`, `geometry`, `frame_records`, or `message` naming a
missing argument)
describing what loading reads from the file alone; given a checkpoint directory,
`tj.inspect(DIR)` reports the stopped run under `checkpoint` instead. A
parallel-beam file without a recorded pixel size loads in pixel units (pitch 1,
`length_unit: "pixel"`); pass `pixel_size=`, `detector=`, or `--pixel-size` for
physical units. A file without a recorded axial center loads with the center at
the middle of the detector and `detector_center_status="not_recorded"`; fit it
with `tj.align(scan, fit="center")` or supply `detector=`. A file that records
no rotation axis or pixel scale warns on load (`geometry_status="not_recorded"`)
and is refused by reconstruction and alignment until geometry is declared
(`geometry=` in `tj.load`, `--geometry parallel` on `tomojax correct`, or
`geometry: {"kind": "parallel"}` on worker or MCP `open`). CLI errors name
`--flats`, `--darks`, `--angles`, or `--pixel-size` where those flags supply the
missing input.

`data_path`, `image_key_path`, and `angles_path` apply to saved HDF5 scans as
well as raw acquisitions across eager and lazy loads, CLI, worker, and MCP.
Bliss accepts `data_path` to select its sequence; fixed-layout formats (TIFF,
MATLAB phase, Nikon, SkyScan, Zeiss, and EDF) reject these paths. Supplying
`flats` or `darks` on an unlabeled projection stack identifies its values as
detector counts in any storage dtype (including floating-point counts); files
with a recorded log correction or PSI phase measurement reject flat/dark
calibration, and exposure normalization requires flat frames rather than a
scalar white level. Geometry overrides on load preserve recorded
`view_positions` (acquisition gaps and restarts).

Both raw-frame and processed NXtomo loading convert rotation angles labeled
`rad`, `radian`, or `radians` to degrees. `deg`, `degree`, `degrees`, and
missing units mean degrees at format-defined NXtomo and Data Exchange angular
paths; unitless generic motors (or an explicit `angles_path=` pointing to a
unitless dataset) require an explicit `angles=` override unless structural
discovery finds matching angular units on same-named fields in the same entry.
Angles must be a real one-dimensional vector with finite sample-view angles
(calibration frames may have NaN angles); unknown units raise an error instead
of being guessed. An `image_key` must be a numeric one-dimensional vector with
exact labels in `{0, 1, 2, 3}` (sample, flat, dark, other); labels are checked
before conversion, including floating-point exports.

Raw NXtomo entries may be nested and link to external detector files; keep those
files beside the metadata file. Repeated links to one dataset are not ambiguous.
After fixed NXtomo, Data Exchange, and GDA paths, discovery follows the NeXus
structure: frames come from the default plot (root and entry `@default` to an
`NXdata` `@signal`), `image_key` comes from the `NXdetector` holding those
frames, and rotation angles come from the `NXdata` coordinate along the frames
(`@axes`, `<name>_indices`) with angular units. When that coordinate has no
units attribute, discovery uses the units recorded on same-named fields
elsewhere in the same entry if they agree (as GDA records each positioner once
per scan) and stores the source in `angle_units_path`. Multiple angular
coordinates resolve to the scan axis (`@axes`) or ask for `angles_path`. If
structure does not locate a dataset, per-frame shape disambiguates named angle
and key datasets from different detectors; a sole named dataset with an invalid
shape is rejected, and ambiguous candidates require an explicit metadata path.

When an `image_key` or angle dataset is paired with the frames by path or
structure, it may be longer or shorter than the frame stack (for example when a
scan stops early and separate writer streams end at different counts). TomoJAX
reads the common prefix from frame 0 covered by all of them, logs a warning with
the counts, and records the counts in `frame_records` for both standard NXtomo paths
and structurally discovered NeXus scans. A key or angle dataset
found only by name is paired by shape and must match the stack length exactly.

`tomojax.load` of a processed file leaves out frames that `image_key` marks as
other than sample views. Passing another scan's HDF5/NeXus file to `flats=` or
`darks=` (`--flats`, `--darks`) reads only that file's flat (`image_key == 1`)
or dark (`image_key == 2`) frames lazily as one set; a stack that labels no
frame kinds contributes every frame. Bracketing multi-frame flat sets are
weighted from a sample of the views by default (`flat_weights="views"`, or
`"position"` for linear position interpolation; see
[flat fields](../corrections/README.md#flat-fields)). Saving a scan retains every pose update, including an
update around a cone-segmented acquisition whose segments have their own;
`Scan.poses` is their complete composition in the detector translation frame,
and `scan.with_poses(None)` removes those updates. Raw Zeiss shifts are converted
once, on import. The `zeiss` metadata describes the original file, including its
per-view shifts in original detector pixels. Combined scans keep their input
metadata under `original_acquisitions`. A saved scan reloads from its effective
detectors, beam arrangements and pose table. Load the original Zeiss acquisition
with `apply_poses=False` to omit its shifts and center correction.

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

`tomojax correct` takes what a file lacks as flags under
`tj.load`'s names: `--pixel-size`, `--detector-center U V`, `--geometry` with
`--tilt` for laminography, and `--source-to-axis`, `--source-to-detector` and the
detector's `--detector-roll`, `--detector-pitch` and `--detector-yaw` for cone data. Python `Detector.center` stores physical lengths, so an
offset in pixels is multiplied by `du`/`dv`. See the
[real scan guide](../../../docs/real-laminography.md) for measured geometry and
for the distinction between raw intensities and corrected absorption.

## SkyScan corrected acquisitions

`tomojax.load_frames` and `tomojax.load` recognize a SkyScan acquisition log by
content, either explicitly or in its folder. Numbered sample TIFFs are read
lazily; scout images and the separate drift-reference acquisition are excluded.
Distances, camera calibration, and the vertical optical-axis line come from
the log in millimeters, with detector v pointing up. Geometric and flat-field
corrections must be enabled in the export. Camera rotation, aspect, and
random-movement settings stay in source metadata without being reapplied.

The log does not establish a horizontal rotation center: supply `detector=`
(or a complete `geometry=`). Missing binning/resize metadata remains unknown.
Keep the log beside its numbered TIFFs. Supply measured flats/white level and
darks explicitly: storage bit depth does not establish calibration. The example
below chooses white 65536 and dark 0; use the actual levels for your export.
For a modest preview, bin detector counts and independently choose a smaller
reconstruction grid, preserving the metadata's physical extent:

```python
import tomojax as tj

detector = tj.Detector(nu, nv, du, dv, (center_u, center_v))  # measured calibration
scan = tj.load_frames("skyscan/", detector=detector, flats=65536, darks=0).corrected(binning=4)
grid = scan.grid.resampled(shape=(256, 256, 256))
result = tj.reconstruct(scan, method="fbp", grid=grid)  # FDK for this cone geometry
tj.save("grains-recon.nxs", result)
```

Alternatively, the equivalent CLI workflow writes the corrected scan, then the
volume and a central-slice preview (use fresh output names):

```sh
tomojax correct skyscan/ --flats 65536 --darks 0 --binning 4 --pixel-size ... \
  --detector-center U V -o grains.nxs
tomojax reconstruct grains.nxs --method fbp --grid 256 256 256 \
  -o grains-recon.nxs --preview grains.png
```

External NXtomo and Data Exchange raw exports are content-discovered, including
renamed files and NXtomo entries other than `/entry`. Recorded physical pixel
sizes are retained in millimeters; an absent axial rotation center defaults to
the detector middle (`detector_center_status="not_recorded"`). Supply
`detector=`/`geometry=` or fit it with `tj.align(scan, fit="center")`. The
support matrix lists metadata paths.
