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
instead. Arbitrary HDF5 layouts may need explicit paths during preprocessing,
set as `data_path`, `angles_path` and `image_key_path` in a `--config` TOML
file; `tomojax preprocess --config-keys` lists them.

Reconstruction expects absorption/log-attenuation projections. Raw detector
intensities, normalized transmission, and already-corrected attenuation are
different inputs. Establish which you have before preprocessing.

## Correct raw detector frames

For an NXtomo scan containing sample, flat, and dark frames:

```bash
uv run --no-sync tomojax preprocess raw.nxs -o corrected.nxs
uv run --no-sync tomojax inspect corrected.nxs
```

By default this applies flat/dark correction and the negative logarithm, then
writes sample-only absorption projections with preprocessing provenance.
`--transmission` writes normalized transmission instead and is not the input
domain expected by the reconstruction commands. Already-corrected absorption
data should bypass this step; applying the logarithm again changes the data.

In Python, `tj.load("raw.nxs")` makes the same correction when the file's
`image_key` marks flats or darks, and records it in `scan.corrections`. For
other corrections, load the frames and pass steps from `tomojax.corrections`:

```python
import tomojax as tj
from tomojax.corrections import BeamHardening

frames = tj.load_frames("raw.nxs")  # or a Nikon .xtekct, or TIFFs with angles=
scan = frames.corrected(BeamHardening((1.0, 0.05)))
print(scan)  # Scan('sample': ..., corrections: flat_dark(...), log(...), beam_hardening(...))
```

Flats taken before and after the scan are interpolated by position, and the
views are corrected on the GPU a batch at a time.

For TIFF data, use the [TIFF and measured-geometry instructions](real-laminography.md#prepare-tiff-data).
`tomojax import` packages a stack; it does not perform flat/dark correction.

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
