# Reconstruct your first scan

This guide starts from an installed checkout; follow [installation](installation.md)
first. If you do not have data yet, run the [synthetic example](../README.md#first-reconstruction).
Commands use `uv run --no-sync` from the checkout root. Replace example input
paths with your files and use new output paths.

## Inspect the data and geometry

```bash
uv run --no-sync tomojax inspect scan.nxs
```

Check the projection count and shape, angles, detector pitch, voxel pitch, and
geometry type. `inspect` also checks the dataset contract: it ends with
`Valid: yes`, or lists the issues and exits with status 1. That check cannot
prove that metadata matches the instrument. `--json` prints the report as JSON
instead. It also lists the corrections that made the projections from
detector frames, when the file records them.

Reconstruction expects line integrals (log attenuation). Raw detector counts,
normalized transmission, and already-corrected attenuation are different
inputs: establish which you have. TomoJAX corrects raw frames itself when the
file says they are raw (an `image_key` marking flats or darks, or a Nikon
scan); it refuses integer counts with no flats rather than guess.

## Correct raw detector frames

For an HDF5 scan containing sample, flat, and dark frames:

```bash
uv run --no-sync tomojax preprocess raw.nxs -o corrected.nxs
uv run --no-sync tomojax inspect corrected.nxs
```

This writes the line integrals `-log((I - D) / (F - D))` of the sample views,
each view's flat interpolated between the flat sets taken around it, and
records the correction. The frames, `image_key` and angles are found at their
NXtomo paths or as the file's only datasets of those names; `--data-path`,
`--image-key-path` and `--angles-path` (expert settings, `--config-keys`) name
them otherwise. Options add corrections: `--zingers`, `--remove-stripes 9`
(rings), `--reject-outliers` (views that jump from their neighbours, such as a
closed shutter), `--beam-hardening 1,0.05`; `--select-views`, `--reject-views`
and `--crop` keep part of the scan. Already-corrected data should skip this
step; applying the logarithm again changes the data.

In Python, `tj.load("raw.nxs")` makes the same correction, and records it in
`scan.corrections`. For other corrections, load the frames and pass steps from
`tomojax.corrections`:

```python
import tomojax as tj
from tomojax.corrections import RejectViews, Stripes

frames = tj.load_frames("raw.nxs")  # or a Nikon .xtekct, or TIFFs with angles=
scan = frames.cropped(slice(100, 900), slice(None)).corrected(RejectViews(), Stripes(9))
print(scan)  # Scan('sample': ..., corrections: flat_dark(...), log(...), reject_views(...), ...)
tj.save("corrected.nxs", scan)
```

The views are corrected on the GPU a batch at a time, read from the file as
they are needed. `batch_views` must be a positive integer and `epsilon` finite
and positive. `frames.selected(...)` and `scan.selected(...)` keep a nonempty
selection in acquisition order: a forward slice, increasing integer indices,
or a Boolean mask. Selection keeps each view's geometry and its original
position between the flat fields.

For TIFF frames, use the [TIFF and measured-geometry instructions](real-laminography.md#prepare-tiff-data).
`tomojax import` packages a stack; it does not correct frames.

## Reconstruct and inspect slices

For parallel or laminography data, start with FBP:

```bash
uv run --no-sync tomojax recon corrected.nxs -o recon.nxs \
  --roi off --manifest recon-manifest.json
uv run --no-sync tomojax inspect recon.nxs --preview previews
```

FBP is the default `--method`. `recon.nxs` stores the volume and copies the
projections. The manifest records reconstruction settings. `previews/` contains
PNGs of the central projection (`projection.png`) and the volume's three central
planes (`slice_z.png`, `slice_y.png`, `slice_x.png`). PNG contrast is scaled for
display; read the stored floating-point volume for quantitative work.
`tomojax export recon.nxs -o slices/` writes the volume as TIFF slices for other
software.

`--roi off` preserves the recorded grid. The default is `--roi auto`, which may
crop to the detector field of view. `--grid NX NY NZ` changes the dimensions but
keeps the input voxel spacing. Check both field of view and physical units before
comparing reconstructions. `--roi cyl` also zeroes the volume outside the
cylinder every view sees.

Expert settings are fields of the method's configuration class
(`tomojax.recon.FBPConfig`, `CGLSConfig`, `FistaConfig` or `SPDHGConfig`),
set in a TOML file passed with `--config`; `tomojax recon --config-keys` lists
them. Each replaces that field of the class's defaults, and a setting the
method does not take fails with the ones it does. For FISTA with Huber total
variation:

```bash
printf 'regulariser = "huber_tv"\nhuber_delta = 0.01\n' > huber.toml
uv run --no-sync tomojax recon corrected.nxs -o tv.nxs --method fista --config huber.toml
```

In Python this is
`tj.reconstruct(scan, "fista", config=FistaConfig(regulariser="huber_tv", huber_delta=0.01))`,
with `from tomojax.recon import FistaConfig`.
The keywords, like the command's options, replace the config's fields of the
same names: `iterations=100` wins over the config's `iterations`.

`tomojax align --mode cor` estimates the detector centre (the centre of
rotation). To compare trial centres by eye instead, shift the detector in
Python; the centre is in the geometry's length unit, so pixels times `du`:

```python
from dataclasses import replace

import tomojax as tj

scan = tj.load("corrected.nxs", poses=False)
d = scan.detector
for u_px in (-4, -2, 0, 2, 4):
    detector = replace(d, center=(u_px * d.du, d.center[1]))
    trial = replace(scan, geometry=replace(scan.geometry, detector=detector))
    tj.save(f"cor{u_px:+d}.nxs", tj.reconstruct(trial))
```

For tilted or irregular-angle data, start with the iterative workflow in the
[real scan guide](real-laminography.md#reconstruct-with-the-recorded-geometry).
If motion remains plausible, use the [alignment guide](alignment-guide.md) after
checking geometry and preprocessing. A visually sharper aligned image does not
by itself establish accurate recovered poses.
