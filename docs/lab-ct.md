# Lab cone-beam CT

This guide takes a lab CT scan (a point source, a flat detector and a sample on
a turntable) from the scanner's files to a reconstruction: import, calibrate
the rotation axis, reconstruct with FDK or an iterative solver, and correct
per-view motion. Every step is one `tomojax` command, run with
`uv run --no-sync` from the checkout root as in the other guides; the Python
calls are at the end.

## Import

A Nikon (X-Tek) scan is an `.xtekct` parameter file beside its projection
TIFFs. `tomojax import` reads the source and detector distances, the detector
pixels and offsets, the reconstruction volume, the white level and the angles
(from `<Name>_ctdata.txt` when present), and `tomojax inspect` describes and checks
the result. Only TIFFs named `<Name>` plus the input separator and a number are
projections; flat fields, thumbnails and reconstructions sharing the prefix are
skipped:

```bash
uv run --no-sync tomojax import scan/part.xtekct -o scan.nxs
uv run --no-sync tomojax inspect scan.nxs
```

Projections become line integrals, `-log(I / WhiteLevel)` (`--transmission`
keeps intensities), with image rows flipped so that the detector's v axis points up.
Lengths stay in the scanner's units, millimetres for Nikon. The axis offset and
detector roll are left at zero for the next step. TomoJAX has not yet been
checked against a real Nikon scan: confirm the handedness of the first
reconstruction on a known sample, and use `--reverse-angles` if the stage
turns the other way.

For other scanners, import the TIFF stack with the angles and the geometry,
then flat- and dark-correct it:

```bash
uv run --no-sync tomojax import projections/ --angles angles.csv --geometry cone \
  --source-to-axis 120.5 --source-to-detector 980.0 --pixel-size 0.2 \
  -o raw.nxs
uv run --no-sync tomojax preprocess raw.nxs -o scan.nxs --flats flats/ --darks darks/
```

`--flats` and `--darks` take TIFF frames, or one level for every pixel
(`--flats 60000 --darks 0`) when the scanner calibrates its flat field.
Use one length unit throughout (the pixel size and both distances).
`--pixel-size` takes one value for square pixels, or the u then v sizes.

`tomojax preprocess` also corrects lab-CT artefacts.
`--beam-hardening 1,0.05` linearises beam hardening, mapping each line
integral p to `p + 0.05 p²` (calibrate the coefficients on a single-material
sample). `--remove-stripes 9` removes rings: it subtracts each detector
pixel's constant offset, comparing its values sorted over views with those of
its 9 neighbouring columns. Structure symmetric about the rotation axis also
looks stationary, so a narrow centred feature can be altered: use the smallest
width that covers the defect. A faint offset on a steep gradient can remain.
`--zingers` replaces bright specks.
In Python these are steps of `tomojax.corrections` (`BeamHardening`,
`Stripes`, `Zingers`) passed to `tj.load_frames(path).corrected(...)`.
`tomojax import --detector-roll`, `--detector-pitch`, `--detector-yaw` and
`--axis-offset` take values the scanner reports. A TIFF import
records no grid: cone datasets without one reconstruct one voxel per detector
pixel at the axis, the pixel size divided by the magnification
`source_to_detector / source_to_axis`, and `tomojax recon --grid NX NY NZ`
sets the number of voxels.

## Calibrate the rotation axis

Scanners record the source and detector distances well but rarely the exact
position of the rotation axis. An axis offset from the central ray doubles
and blurs every slice; a rolled detector makes that offset change with height.

```bash
uv run --no-sync tomojax align scan.nxs -o calibrated.nxs --mode cor
```

On cone data `--mode cor` estimates the axis offset (`ConeBeam.axis_offset`,
the lab-CT centre of rotation) and the detector roll. It reconstructs thin
FDK slabs near the centre, top and bottom of the volume for trial offsets,
coarse to fine on binned data, and keeps the sharpest; the slabs' offsets
change linearly with height in proportion to the roll. The output records the
calibrated beam, so later commands on `calibrated.nxs` use it. On simulated
scans it recovers offsets to 0.08 voxels and rolls to 0.04°
([measurements](measurements.md#cone-beam-scans)). It needs some in-plane structure at the slab heights and a
scan of at least 180° plus the fan angle. Detector pitch and yaw, the axis
direction and the source distances are not estimated.

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
`regulariser = "huber_tv"` (see the
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

Sample drift, stage wobble and thermal motion move the sample a little in
every view. Cone-beam alignment estimates all six pose parameters per view,
including `dy` along the beam, which changes the magnification:

```bash
uv run --no-sync tomojax align scan.nxs -o aligned.nxs --mode cor-then-pose
uv run --no-sync tomojax recon aligned.nxs -o recon.nxs --method cgls
```

`cor-then-pose` calibrates the axis as `--mode cor` does, then aligns the
poses; `--mode pose` aligns the poses alone, and also takes scans that already
carry poses (an ASTRA import, say, or an earlier alignment), which it corrects
on top of, and multi-orbit scans, whose orbits it brings into register (see
[multi-orbit scans](#multi-orbit-scans)). `tomojax recon` applies the
poses saved in `aligned.nxs`; with either command, `--no-poses` starts from the
nominal geometry instead. See
[cone-beam alignment](alignment-guide.md#align-cone-beam-scans-in-six-degrees-of-freedom)
for its accuracy and gauges.

## Speed

On a 256³, 360-view scan on a laptop RTX 4070, FDK takes 0.048 s (ASTRA
0.29 s, TIGRE 0.51 s) and the forward projection 0.093 s (ASTRA 0.17 s). The
exact transpose takes 0.18 s, slower than ASTRA's approximate backprojector
(0.075 s). The [measurements](measurements.md#cone-beam-projection-and-fdk)
give the protocol, errors and the 1024³ FDK.

## Python

```python
import tomojax as tj

scan = tj.load("scan/part.xtekct")
calibrated = tj.align(scan, mode="cor").scan      # axis offset and detector roll
recon = tj.reconstruct(calibrated)                # FDK
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
volume = tj.reconstruct(scan).volume              # (x, y, z): ASTRA's (z, y, x) transposed
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
uv run --no-sync tomojax recon orbit.nxs -o fdk.nxs --method fbp --roi off
uv run --no-sync tomojax recon orbit.nxs -o cgls.nxs --method cgls --iterations 20 --roi off
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
edges. `tj.align(scan)` selects `pose` alignment for segments as one scan and
brings the orbits into register; the other modes refuse segmented scans. Only
the orbits' heights relative to one another are observable, so the result is
the least-motion estimate, and the whole object can sit away from where the
record puts it (0.38 mm, almost four voxels, on the FIPS walnut).

For a quick check without reconstructing, `orbit_heights` estimates each
orbit's height above the first from the consistency of the views' data. It is
good to about half a pixel at the axis, so treat it as a check or a starting
point, not a replacement for alignment:

```python
import numpy as np

from tomojax.alignment.api import orbit_heights

found = orbit_heights(scan.geometry, np.asarray(scan.projections))
print(found.heights)   # in the scan's length unit; the first orbit is 0
print(found.contrast)  # near 0: the data cannot decide that orbit's height
```

It returns `None` for a single orbit.

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
