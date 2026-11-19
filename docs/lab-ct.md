# Lab cone-beam CT

This guide takes a lab CT scan (a point source, a flat detector and a sample on
a turntable) from the scanner's files to a reconstruction: import, calibrate
the rotation axis, reconstruct with FDK or an iterative solver, and correct
per-view motion. Every step is one `tomojax` command, run with
`uv run --no-sync` from the checkout root as in the other guides; the Python
calls are at the end.

## Import

A Nikon (X-Tek) scan is an `.xtekct` parameter file beside its projection
TIFFs. `tomojax preprocess` reads the source and detector distances, the detector
pixels and offsets, vendor reconstruction metadata, the white level and nominal
angles from `InitialAngle` and `AngularStep`; `.ang` and `_ctdata.txt` sidecars
are not used. `tomojax inspect` describes and checks the result.
Only TIFFs named `<Name>` plus the input separator and a number are
projections; flat fields, thumbnails and reconstructions sharing the prefix are
skipped. If a directory holds several parameter files, name the intended
calibration file explicitly. The examples choose zero dark level; use that
only for already dark-corrected counts, otherwise supply measured darks:

```bash
uv run --no-sync tomojax preprocess scan/part.xtekct --darks 0 -o scan.nxs
uv run --no-sync tomojax inspect scan.nxs
```

Projections become line integrals, `-log((I - D) / (WhiteLevel - D))`, with
both TIFF rows and columns reversed from Nikon's top-right panel origin.
Rotation angles and horizontal calibration offsets are converted consistently
into TomoJAX's detector and object-angle frame.
Lengths stay in the scanner's units, millimetres for Nikon. The calibrated
rotation axis is read from `ObjectOffsetX`, `ObjectRoll` and `ObjectTilt`, or
the older `CentreOfRotationTop/Bottom` detector-plane offsets. The declared
view count excludes an extra closing frame on a full turn, with a warning.
Supply a measured schedule through explicit `angles=` in `load_frames`
(or `--angles` in preprocessing), already expressed in TomoJAX's object-angle
frame rather than the vendor's stage convention.

Zeiss Xradia projection acquisitions can be corrected directly, without a TIFF
export; see the [Zeiss preview](#zeiss-xradia-preview) below. For scanners without
a reader, provide the measured `ConeGeometry` to `tj.load_frames(...,
geometry=..., flats=..., darks=...)`. Already-corrected TIFF line integrals can
be packaged on the command line; `calibration.toml` supplies the measured
`detector_center = [u, v]` in physical length units:

```bash
uv run --no-sync tomojax import projections/ --angles angles.csv --geometry cone \
  --source-to-axis 120.5 --source-to-detector 980.0 --pixel-size 0.2 \
  --config calibration.toml -o scan.nxs
```

`--flats` and `--darks` take TIFF frames, or one level for every pixel
(`--flats 60000 --darks 0`) when the scanner calibrates its flat field.
Use one length unit throughout (the pixel size and both distances).
`--pixel-size` takes one value for square pixels, or the u then v sizes.

`tomojax preprocess` also corrects lab-CT artefacts.
Beam-hardening correction uses an explicit polynomial or spectrum through the
Python `BeamHardening` step; the
[corrections reference](../src/tomojax/corrections/README.md#beam-hardening)
describes these models. Vendor settings are retained unapplied: their intensity
domains and scaling conventions are not established. `--stripes 9` removes rings:
it subtracts each detector pixel's constant offset, comparing its values sorted
over views with those of its 9 neighbouring columns. Structure symmetric about the rotation axis also
looks stationary, so a narrow centred feature can be altered: use the smallest
width that covers the defect. A faint offset on a steep gradient can remain.
`--zingers` replaces bright specks.
In Python these are steps of `tomojax.corrections` (`BeamHardening`,
`Stripes`, `Zingers`) passed to `tj.load_frames(path).corrected(...)`.
`tomojax import --axis-offset` and TOML keys `detector_roll`,
`detector_pitch`, `detector_yaw` set the detector's orientation; pitch and
yaw apply to cone data only. They are
right-handed rotations about the detector's own axes, the normal (pointing back
towards the source), u and v, in that order: positive roll turns u towards v,
positive pitch tilts the +v edge towards the source and positive yaw swings the +u
edge away from it. Scanners do not report these. Nikon's `ObjectRoll` and
`ObjectTilt` describe the rotation axis and are read by the loader, not typed into
these keys. A TIFF import
records no grid: cone datasets without one reconstruct one voxel per detector
pixel at the axis, the pixel size divided by the magnification
`source_to_detector / source_to_axis`, and `tomojax recon --grid NX NY NZ`
sets the number of voxels while preserving the selected physical field of view
(adjusting voxel sizes). For a small whole-volume preview, bin the detector
with `preprocess` and use `recon --grid 64 64 64 --region off`. In Python the same
grid is `scan.grid.resampled((64, 64, 64))`, passed as `grid=` to reconstruction.

### Zeiss Xradia preview

Use the **projection acquisition** (usually `.txrm`), not a reconstructed `.txm`
volume or a single scout image. TomoJAX reads the embedded reference image,
angles, detector calibration and source distances. The file's per-view image
shifts (`alignment/x-shifts`, `y-shifts`) become per-view detector centres, so
the measured pixels are never resampled. They are measured from the centre that
a Zeiss reconstruction setup records as `AutoRecon/CenterShift`. A raw scan without
it loads with the centre at the middle of the detector, the shifts added to it, and
`detector_center_status` "not_recorded": include `"center"` in `tj.align`'s `fit=`, or
pass `detector=tj.Detector(nu, nv, pitch, pitch, (u, 0))` for a known centre, with
`pitch` the recorded pixel size times the magnification. An explicit `geometry=` is used as given,
without the shifts, and the loader warns how many views it dropped. Because the
shifts move the detector, `tj.align` leaves them out of `result.poses`; a pose
translation moves the object instead, so the same correction written as a pose is
the shift times the recorded (object-plane) pixel size, with the opposite sign. For a
separate reference image, pass `--flats reference_Front.xrm` or
`load_frames("scan.txrm", flats="reference_Front.xrm")`.
Sample and reference exposure times are used to compare dark-subtracted count
rates. Multiple embedded references are interpolated at their acquisition
timestamps. Raw pixels remain unchanged. If supplying calibration arrays rather
than XRM files, also pass their `flat_exposure_times=` in seconds.
If no flat is available, supply the measured open-beam level as `flats=`; one is not inferred
from the detector's bit depth. A file that records
`AcquisitionSettings/CameraStaticBackground` = 1 had its dark removed by the camera
and needs no dark; otherwise, if no dark reference is stored, supply measured darks
with `--darks` or `load_frames(..., darks=...)`: absent dark calibration is an
error, not an assumed zero.

The camera states the top of its linear range
(`ConfigureBackup/ConfigCamera/Camera 1/MaxLinearRange`, 45 000 counts on the scans
measured here). Zeiss holds the references inside it but not the views, so a long
exposure can put the bright parts of a view, usually the air beside the sample, above
it or into saturation. The `flat_dark` correction record gives the range, the fraction
of counts above it, and `linear_limit_line_integral`, the line integral where the range
ends; `tj.load` warns when any count lies above it. Line integrals below that value are
unreliable, and saturated air reads as absorbing: on CuO_2 (6 s views, 2.8 s
references, 52% of counts above the range) air corrects to about 0.33 instead of 0,
which the reconstruction carries as a false background around the sample. The values
are not changed.

This is a coarse, whole-volume CPU preview. It keeps every eighth view, bins
16 x 16 detector pixels and resamples the original physical volume to 64³ voxels.
Replace `801` below with the acquisition's number of sample views; CLI ranges
require both start and stop indices:

```bash
printf 'binning = 16\n' > preview.toml
uv run --no-sync tomojax preprocess scan.txrm --darks 0 -o corrected.nxs \
  --views 0:801:8 --config preview.toml
uv run --no-sync tomojax recon corrected.nxs -o preview.nxs \
  --method cgls --iterations 10 --grid 64 64 64 --region off
uv run --no-sync tomojax inspect preview.nxs --preview previews
```

The equivalent Python route is:

```python
import tomojax as tj

scan = tj.load_frames("scan.txrm").selected(slice(None, None, 8)).corrected(binning=16)
grid = scan.grid.resampled((64, 64, 64))
recon = tj.reconstruct(scan, "cgls", iterations=10, grid=grid)
tj.save("preview.nxs", recon)
```

Per-view detector shifts can make this a `ConeSegments` geometry. Automatic
reconstruction then selects matched CGLS, not the single-arrangement FDK path.
Binning changes the detector sampling, **not** the recorded reconstruction grid;
keep the explicit grid for a small preview. Inspect all three slice planes and
increase view sampling, resolution and iterations for a final reconstruction.
A finite volume or lower residual alone does not establish its accuracy.

## Calibrate the detector centre and setup angles

Scanners record the source and detector distances, but their detector centre and orientation may
still need correction. A wrong centre doubles or blurs features; roll leaves a height-dependent
error. Fit both shared corrections with:

```bash
uv run --no-sync tomojax align scan.nxs -o calibrated.nxs --fit center roll
```

For a cone beam, `"center"` fits detector-u and detector-v offsets in the detector frame.
These are not object translations or a replacement for the recorded `ConeBeam.axis_offset`.
`"roll"` fits one detector angle; add `"tilt"` if the nominal rotation-axis direction is wrong.
The returned scan carries the calibrated geometry. Source distances, detector pitch and yaw
remain as declared. Inspect multiple heights and compare with an unaligned reconstruction;
a low residual alone does not establish physical calibration.

## Reconstruct

The default `--method auto` selects FDK for a single cone-beam arrangement;
`--method fbp` selects it explicitly:

```bash
uv run --no-sync tomojax recon calibrated.nxs -o fdk.nxs
```

It weights full turns, offset-detector full turns (the axis projecting near
one edge of the detector, which widens the field of view) and short scans of
at least 180° plus the fan angle (Parker weights). Like every FDK, it is exact
only in the orbit plane, with cone artefacts growing away from it.

The iterative solvers model the cone beam exactly, with any per-view poses.
CGLS suits low-noise data, and FISTA with total variation suits noisy or
sparse scans:

```bash
uv run --no-sync tomojax recon calibrated.nxs -o cgls.nxs --method cgls --iterations 20
uv run --no-sync tomojax recon calibrated.nxs -o tv.nxs --method fista --iterations 50 --tv-weight 0.002
```

Expert settings are fields of the explicitly selected method's configuration
class in a `--config` TOML file: `views_per_batch = 16`, say, or FISTA's
`regularizer = "huber_tv"` (see the
[quickstart](quickstart.md#reconstruct-and-inspect-slices)); in Python, pass
the class, `tj.reconstruct(scan, "fista", config=FistaConfig(...))`.

Their backprojection is the exact transpose of the projector, so CGLS stays
stable as it iterates; ASTRA's CGLS, whose backprojector is approximate,
diverges after about 20 iterations on the same scan (see
[measurements](measurements.md#cone-beam-projection-and-fdk)).

When the volume would not fit in GPU memory, FDK switches to z-slabs on the
host by itself; see [large scans and several GPUs](large-scans.md).

## Export

Volume viewers and analysis packages read slice stacks or raw volumes:

```bash
uv run --no-sync tomojax export fdk.nxs -o fdk_slices/                 # 32-bit TIFF per z slice
uv run --no-sync tomojax export fdk.nxs -o fdk.raw --dtype uint16      # one raw file
```

An output ending in `.raw` is written as one raw file; any other output is a
directory of TIFF slices. TIFF slices have y rows and x columns, numbered from
the bottom of the volume; raw files are little-endian and z-major. A JSON
sidecar (`fdk.json` beside the raw file, `slice.json` in the slice directory)
records the shape, the voxel size and, for `uint16`, the value range
mapped to 0–65535 (`--range`, by default the 0.1 and 99.9 percentiles). The
export reads one slice at a time.

## Correct per-view motion

Sample drift, stage wobble and thermal motion move the sample a little in every view.
The default fits three lab-frame translations for cone data, including along-beam motion that
changes magnification. Add per-view rotations when the data require them:

```bash
uv run --no-sync tomojax align scan.nxs -o aligned.nxs --fit translations rotations
uv run --no-sync tomojax recon aligned.nxs -o recon.nxs --method cgls
```

The corrections are fitted on top of any input poses, including imported and multi-orbit scans.
`tomojax recon` applies the calibrated geometry and poses saved in `aligned.nxs`; `--no-apply-poses`
removes per-view corrections when loading. See [align a scan](align-a-scan.md) for parameter
selection and gauges.

## Speed

On a 256³, 360-view scan on a laptop RTX 4070, with every library taking and
returning host arrays, FDK takes 0.092 s (ASTRA 0.26 s, TIGRE 0.50 s) and the
forward projection 0.13 s (ASTRA 0.17 s). The exact transpose takes 0.12 s,
slower than ASTRA's approximate backprojector (0.073 s). The
[measurements](measurements.md#cone-beam-projection-and-fdk) give the protocol,
errors, device-resident times and the 1024³ FDK.

## Python

```python
import tomojax as tj

scan = tj.load_frames("scan/part.xtekct", darks=0).corrected()
calibrated = tj.align(scan, fit=("center", "roll")).scan
recon = tj.reconstruct(calibrated, "cgls")
tj.save("part-recon.nxs", recon)
```

The `tomojax` commands hold the projections and the volume in host memory.
For scans larger than that, `fdk_host` reconstructs from a projection memmap
into a volume memmap; see [large scans and several GPUs](large-scans.md).

## Coming from ASTRA

`tj.Scan.from_astra` takes projections and geometries exactly as an ASTRA
Toolbox script holds them: data in ASTRA's `(rows, views, columns)` layout, a
`cone` or `cone_vec` projection geometry and a 3-D volume geometry, as
`astra.create_proj_geom` and `astra.create_vol_geom` make them.

```python
import astra
import tomojax as tj

proj_geom = astra.create_proj_geom("cone_vec", rows, cols, vectors)
vol_geom = astra.create_vol_geom(501, 501, 501)  # with its window set as usual
scan = tj.Scan.from_astra(data, proj_geom, vol_geom)
volume = tj.reconstruct(scan).volume              # (z, y, x), shared with ASTRA
data, proj_geom, vol_geom = scan.to_astra()      # and back
```

ASTRA moves the source and detector around a fixed object; TomoJAX fixes them
and moves the object, so each ASTRA view becomes a rigid object pose. The
conversion fits a circular orbit (`scan.geometry`, with `scan.angles` running
the opposite way to ASTRA's) and keeps the scan's departure from it, such as a
tilted axis or per-view corrections, in `scan.poses`; a source raised above the
volume moves the grid instead. The source and detector must keep one
arrangement for runs of views: where the arrangement changes partway through,
as in a multi-orbit scan, each run becomes a segment of a `ConeSegments` scan
(see the next section), and vectors whose arrangement changes from view to view
are refused. Projections of the converted scan agree with ASTRA's to 0.03% on
tilted, offset and jittered geometries.

## Try one FIPS walnut orbit on a CPU

Download and unpack a WalnutN directory from the FIPS collection linked in
[the examples](../examples/README.md#bring-a-real-multi-orbit-scan-into-register).
The importer handles its portrait TIFF orientation, reversed acquisition order,
dark/flat correction and ASTRA vector geometry. It does not require ASTRA.
Its defaults keep every twentieth view of the middle orbit, bin the detector
12 × 12 and use 64³ voxels over the authors' 50.1 mm cube:

```bash
uv run --no-sync python examples/import_walnut.py ~/data/walnuts/Walnut1 -o orbit.nxs
uv run --no-sync tomojax recon orbit.nxs -o fdk.nxs --method fbp --region off
uv run --no-sync tomojax recon orbit.nxs -o cgls.nxs --method cgls --iterations 20 --region off
uv run --no-sync tomojax inspect cgls.nxs --preview walnut-previews
```

In Python, after the same import:

```python
import tomojax as tj

scan = tj.load("orbit.nxs")
fdk = tj.reconstruct(scan, "fbp")
cgls = tj.reconstruct(scan, "cgls", iterations=20)
tj.save("walnut-cgls.nxs", cgls)
```

Open all three slice planes, not just the axial plane: a single circular orbit
has cone artefacts away from that plane. The input uses the scanner's original
geometry, not the authors' corrected geometry; this is a coarse preview, not a
reference-quality reconstruction. `--orbit 1` or `--orbit 3` selects another
orbit. Smaller `--every` and `--binning` and larger `--size` increase sampling,
resolution, memory and runtime; leave the physical volume extent unchanged.
For all three orbits together, use the alignment example below, not FDK.

## Multi-orbit scans

A scan whose source and detector arrangement changes partway through, such as
a multi-orbit scan with the source at several heights or a tall sample scanned
in stacked sections, has a `ConeSegments` geometry: each run of views with one
arrangement is a segment. `tj.Scan.from_astra` makes one from ASTRA
`cone_vec` vectors, and `tj.Scan.combine([a, b, c])` joins separate scans.
Views that repeat an angle under one arrangement (several turns) stay in one
segment.

The iterative methods reconstruct all the segments together. FDK reconstructs
one arrangement and refuses segmented scans; `warm_start=True` starts an
iterative method from the view-count-weighted mean of each segment's FDK,
which needs every segment to cover an arc FDK accepts.

Recorded orbit heights are often slightly wrong, which doubles and blurs
edges. `tj.align(scan)` fits their translations jointly as one scan. Only the orbits'
heights relative to one another are observable, so the whole object can sit away from where
the record puts it. The earlier FIPS walnut comparison found a 0.38 mm displacement;
that is a historical gauge observation, not a qualification of the new solver.

[`examples/align_walnut_orbits.py`](../examples/align_walnut_orbits.py) runs
the whole workflow on a FIPS walnut, three orbits recorded with the
uncorrected geometry: it reads them with `from_astra`, bins the detector 2 × 2
(`scan.binned(2)`), aligns and reconstructs. It needs a CUDA GPU with 8 GB and
about six minutes. On that walnut `tj.align` recovers the orbits' heights to
within 0.04 mm of the authors' correction; the
[measurements](measurements.md#fips-walnuts-real-lab-cone-beam-ct) give the
details and the comparison with ASTRA.

## Limitations

TomoJAX models a circular source orbit (no helical scans) and a flat
detector. Beam-hardening correction is a polynomial with given coefficients,
and scatter is not corrected.
[Known limitations](known-limitations.md) lists the rest.
