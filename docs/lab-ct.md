# Lab cone-beam CT

This guide takes a lab CT scan (a point source, a flat detector and a sample on
a turntable) from the scanner's files to a reconstruction: import, calibrate
the rotation axis, reconstruct with FDK or an iterative solver, and correct
per-view motion. Every step is one `tomojax` command; the Python calls are at
the end.

## Import

A Nikon (X-Tek) scan is an `.xtekct` parameter file beside its projection
TIFFs. `tomojax import` reads the source and detector distances, the detector
pixels and offsets, the reconstruction volume, the white level and the angles
(from `_ctdata.txt` when present), and `tomojax inspect` describes and checks
the result:

```bash
tomojax import scan/part.xtekct -o scan.nxs
tomojax inspect scan.nxs
```

Projections become absorption, `-log(I / WhiteLevel)` (`--transmission` keeps
intensities), with image rows flipped so that the detector's v axis points up.
Lengths stay in the scanner's units, millimetres for Nikon. The axis offset and
detector roll are left at zero for the next step. TomoJAX has not yet been
checked against a real Nikon scan: confirm the handedness of the first
reconstruction on a known sample, and use `--reverse-angles` if the stage
turns the other way.

For other scanners, import the TIFF stack with the angles and the geometry,
then flat- and dark-correct it:

```bash
tomojax import projections/ --angles angles.csv --geometry cone \
  --source-to-axis 120.5 --source-to-detector 980.0 --pixel-size 0.2 \
  -o raw.nxs
tomojax preprocess raw.nxs -o scan.nxs --config flat.toml
```

where `flat.toml` gives the constant flat and dark levels (expert settings
such as these are keys of a TOML file passed with `--config`;
`tomojax preprocess --config-keys` lists them):

```toml
assume_flat_field = 60000
assume_dark_field = 0
```

Use one length unit throughout (the pixel size and both distances).
`--pixel-size` takes one value for square pixels, or the u then v sizes.

`tomojax preprocess` also corrects two lab-CT artefacts in absorption data.
`--beam-hardening 1,0.05` linearises beam hardening, mapping each value p to
`p + 0.05 p²` (calibrate the coefficients on a single-material sample).
`--remove-stripes 9` removes rings: it subtracts each detector pixel's
constant offset, comparing its values sorted over views with those of its 9
neighbouring columns, so data without defects pass unchanged (a faint offset
on a steep gradient can remain).
`tomojax import --detector-roll`, `--detector-pitch`, `--detector-yaw` and
`--axis-offset` take values the scanner reports. `tomojax preprocess` also
takes measured flat and dark fields (`--flats`, `--darks`), but only for TIFF
input, and that path records parallel geometry with unit pixels. A TIFF import
records no grid: cone datasets without one reconstruct one voxel per detector
pixel at the axis, the pixel size divided by the magnification
`source_to_detector / source_to_axis`, and `tomojax recon --grid NX NY NZ`
sets the number of voxels.

## Calibrate the rotation axis

Scanners record the source and detector distances well but rarely the exact
position of the rotation axis. An axis offset from the central ray doubles
and blurs every slice; a rolled detector makes that offset change with height.

```bash
tomojax align scan.nxs -o calibrated.nxs --mode cor
```

On cone data `--mode cor` estimates the axis offset (`ConeBeam.axis_offset`,
the lab-CT centre of rotation) and the detector roll. It reconstructs thin
FDK slabs near the centre, top and bottom of the volume for trial offsets,
coarse to fine on binned data, and keeps the sharpest; the slabs' offsets
change linearly with height in proportion to the roll. The output records the
calibrated beam, so later commands on `calibrated.nxs` use it. On simulated
128³ and 256³ scans it recovers offsets to 0.08 voxels and rolls to 0.04°, in
about 4 s at 256³. It needs some in-plane structure at the slab heights and a
scan of at least 180° plus the fan angle. Detector pitch and yaw, the axis
direction and the source distances are not estimated.

## Reconstruct

FDK (the default `--method fbp` runs FDK on cone data) is the fast first
reconstruction:

```bash
tomojax recon calibrated.nxs -o fdk.nxs
```

It weights full turns, offset-detector full turns (the axis projecting near
one edge of the detector, which widens the field of view) and short scans of
at least 180° plus the fan angle (Parker weights). Like every FDK, it is exact
only in the orbit plane, with cone artefacts growing away from it.

The iterative solvers model the cone beam exactly, with any per-view poses.
CGLS suits low-noise data, and FISTA with total variation suits noisy or
sparse scans:

```bash
tomojax recon calibrated.nxs -o cgls.nxs --method cgls --iterations 20
tomojax recon calibrated.nxs -o tv.nxs --method fista --iterations 50 --tv-weight 0.002
```

Their backprojection is the exact transpose of the projector, so CGLS stays
stable as it iterates; ASTRA's CGLS, whose backprojector is approximate,
diverges after about 20 iterations on the same scan (see
[measurements](measurements.md#cone-beam-projection-and-fdk)).

When the volume would not fit in device memory, `--method fbp` reconstructs it
in z-slabs on the host (`fdk_host`), so a 2000³ volume needs host RAM for the
projections and the volume but only a few slabs on the GPU.

## Export

Volume viewers and analysis packages read slice stacks or raw volumes:

```bash
tomojax export fdk.nxs -o fdk_slices/                 # 32-bit TIFF per z slice
tomojax export fdk.nxs -o fdk.raw --dtype uint16      # one raw file
```

An output ending in `.raw` is written as one raw file; any other output is a
directory of TIFF slices. TIFF slices have y rows and x columns, numbered from
the bottom of the volume; raw files are little-endian and z-major. A JSON
sidecar records the shape, the voxel size and, for `uint16`, the value range
mapped to 0–65535 (`--range`, by default the 0.1 and 99.9 percentiles). The
export reads one slice at a time.

## Correct per-view motion

Sample drift, stage wobble and thermal motion move the sample a little in
every view. Cone-beam alignment estimates all six pose parameters per view,
including `dy` along the beam, which changes the magnification:

```bash
tomojax align scan.nxs -o aligned.nxs --mode cor-then-pose
tomojax recon aligned.nxs -o recon.nxs --method cgls
```

`cor-then-pose` calibrates the axis as `--mode cor` does, then aligns the
poses; `--mode pose` aligns the poses alone. `tomojax recon` applies the
alignment saved in `aligned.nxs`; `--ignore-alignment` reconstructs with the
nominal geometry instead. See
[cone-beam alignment](alignment-guide.md#align-cone-beam-scans-in-six-degrees-of-freedom)
for its accuracy and gauges.

## Speed

On a 256³, 360-view scan with a 384² detector on a laptop RTX 4070, forward
projection takes 0.12 s (ASTRA 0.17 s, TIGRE 0.48 s), the exact transpose
0.18 s (ASTRA's approximate backprojector 0.075 s) and FDK 0.10 s (ASTRA
0.29 s, TIGRE 0.52 s). A tilted axis or per-view poses use the general
kernels: 0.17 s forward and 0.29 s transpose. A 1024³ FDK of 1024 views takes
11.7 s in RAM (ASTRA 20.3 s, TIGRE 30.7 s). Details are in the
[measurements](measurements.md#cone-beam-projection-and-fdk).

## Python

```python
import tomojax as tj

scan = tj.load("scan/part.xtekct")
calibrated = tj.align(scan, mode="cor").scan      # axis offset and detector roll
recon = tj.reconstruct(calibrated)                # FDK
tj.save("part-recon.nxs", recon)
```

The `tomojax` commands hold the projections and the volume in host memory.
For scans larger than that, `fdk_host` reconstructs in z-slabs from a
projection memmap into a volume memmap, reading only the detector rows each
slab needs.

## Limitations

TomoJAX models a circular source orbit (no helical scans) and a flat
detector. Beam-hardening correction is a polynomial with given coefficients,
and scatter is not corrected.
[Known limitations](known-limitations.md) lists the rest.
