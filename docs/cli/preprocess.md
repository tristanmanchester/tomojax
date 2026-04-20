# preprocess

The `tomojax-preprocess` command corrects raw NXtomo sample/flat/dark
frames into sample-only projections ready for reconstruction. It uses
`image_key` values (0=sample, 1=flat, 2=dark) to identify frame types
and writes normalised transmission by default. Add `--log` to write
absorption (`-log` transmission) for reconstruction workflows.

```
tomojax-preprocess <raw.nxs> <corrected.nxs> \
  [--log] [--epsilon 1e-6] [--clip-min 1e-6] \
  [--select-views SPEC] [--reject-views SPEC] \
  [--auto-reject off|nonfinite|outliers|both] \
  [--crop y0:y1,x0:x1]
```

## View filtering

View indices refer to sample-view indices after `image_key == 0`
filtering, not raw acquisition frame numbers. This distinction
matters when the raw file interleaves flat and dark frames between
sample exposures.

Specs accept comma- or whitespace-separated integers and half-open
ranges like `0:90` or `120:180:2`. Files use the same syntax and may
include `#` comments for annotation.

Selection and rejection are applied in a fixed order:

1. `--select-views` / `--select-views-file` keeps only the listed
   views (if specified).
2. `--reject-views` / `--reject-views-file` removes the listed views
   from whatever remains.
3. `--auto-reject` drops views based on the corrected data.

Output view order always follows the original sample-view order.

The automatic rejection modes detect common data-quality problems
after flat/dark correction:

| Mode | Description |
|------|-------------|
| `nonfinite` | Drops views containing any NaN or Inf values |
| `outliers` | Drops views whose median intensity is a robust-MAD outlier (controlled by `--outlier-z-threshold`, default 6.0) |
| `both` | Applies both checks |

## Detector crop

The `--crop y0:y1,x0:x1` flag crops projection axes before
correction. The crop coordinates use the same half-open range
convention as Python slicing.

Cropping updates the output detector metadata to reflect the new
`nv` and `nu` dimensions. The `det_center` value is shifted so that
cropped pixel coordinates remain physically aligned with the
original detector geometry. This means downstream reconstruction and
alignment commands can use the cropped file without manual center
adjustments.

## Path overrides

If your NXtomo file uses a non-standard HDF5 layout, you can
override the default dataset paths. This is common with beamline
data that stores frames, angles, or image keys under custom groups.

| Flag | Description |
|------|-------------|
| `--data-path` | HDF5 path to the raw frame stack `[n_frames, nv, nu]` |
| `--angles-path` | HDF5 path to the rotation angles `[n_frames]` |
| `--image-key-path` | HDF5 path to the image key array `[n_frames]` |
| `--assume-dark-field` | Use a constant dark field value (e.g. `0`) when no dark frames are present |
| `--assume-flat-field` | Use a constant flat field value when no flat frames are present |

> [!TIP]
> If your raw file has no dark frames, pass `--assume-dark-field 0`
> to use a zero dark field instead of failing.

## Examples

The examples below use `uv run` to invoke the console script. You
can substitute `python -m tomojax.cli.preprocess` if you prefer.

### Basic transmission output

This produces normalised transmission projections with default
settings.

```bash
uv run tomojax-preprocess raw.nxs corrected_transmission.nxs
```

### Absorption output with log

Adding `--log` writes `-log(transmission)` projections, which is
what most reconstruction algorithms expect as input.

```bash
uv run tomojax-preprocess raw.nxs corrected_absorption.nxs \
  --log --epsilon 1e-6 --clip-min 1e-6
```

### Detector crop

Crop the detector to a region of interest before correction. The
output metadata is updated to match the cropped geometry.

```bash
uv run tomojax-preprocess raw.nxs corrected_cropped.nxs \
  --log --crop 120:900,64:960
```

### Reject known bad views

Remove specific sample views by index or range.

```bash
uv run tomojax-preprocess raw.nxs corrected_rejected.nxs \
  --reject-views 12,57:61
```

### Combined rejection with auto-reject

Load a file listing bad views, then apply automatic non-finite and
outlier rejection on top.

```bash
uv run tomojax-preprocess raw.nxs corrected_robust.nxs \
  --reject-views-file bad_views.txt \
  --auto-reject both --outlier-z-threshold 6
```

### Path overrides for non-standard layouts

Override the default HDF5 paths when working with beamline data
that doesn't follow the standard NXtomo layout.

```bash
uv run tomojax-preprocess raw.nxs corrected.nxs \
  --data-path /entry/imaging/data \
  --angles-path /entry/imaging_sum/smaract_zrot_value_set \
  --image-key-path /entry/instrument/EtherCAT/image_key
```

## Notes

A few things to keep in mind when using this command.

- The output `image_key` values are all `0` because the corrected
  file contains sample projections only. Flat and dark frames are
  consumed during correction and not written to the output.
- Provenance metadata is written to
  `/entry/processing/tomojax/preprocess` in the output file. This
  includes the source dataset paths, frame counts, view filtering
  configuration, crop bounds, angular coverage before and after
  filtering, correction options, warning counts, and mean flat/dark
  fields.
- Stripe and ring correction are not yet included in this command.
  Apply them as a separate post-processing step if needed.

---

See also: [data format](../reference/data-format.md),
[recon](recon.md).
