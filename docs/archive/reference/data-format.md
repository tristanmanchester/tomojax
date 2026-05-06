# Data format (NXtomo)

TomoJAX stores all datasets in HDF5 files using the NeXus NXtomo
convention with additional TomoJAX-specific metadata. The default file
extension is `.nxs`.

## Required structure

Every valid NXtomo file must contain these paths:

```
/entry (NXentry)
  definition = "NXtomo"
  /instrument (NXinstrument)
    /detector (NXdetector)
      data              float32 (n_views, nv, nu), chunked (1, 256, 256)
      image_key         int32 (n_views) — 0=sample, 1=flat, 2=dark
      x_pixel_size      float, units: pixel
      y_pixel_size      float, units: pixel
  /sample (NXsample)
    /transformations (NXtransformations)
      rotation_angle    float32 (n_views), units: degree
      rotation_axis     float32 (3,), default [0, 0, 1]
      depends_on = "rotation_angle"
  /data (NXdata)
    projections       → link to /entry/instrument/detector/data
    signal = "projections"
```

The `data` dataset stores absorption projections (`-log`
transmission). The `image_key` field identifies frame types using
NXtomo conventions.

## TomoJAX extras

TomoJAX writes additional metadata beyond the NXtomo standard:

### Geometry

```
/entry/geometry (NXcollection)
  type = "parallel" | "lamino"
  @geometry_meta_json   (optional) JSON, for example:
                        { "tilt_deg": 35.0, "tilt_about": "x" }
```

### Grid and detector

```
/entry/@grid_meta_json            JSON-serialized Grid:
                                  { nx, ny, nz, vx, vy, vz }

/entry/instrument/detector/@detector_meta_json
                                  JSON-serialized Detector:
                                  { nu, nv, du, dv, det_center }
```

### Processing results

```
/entry/processing (NXprocess)
  /tomojax (NXcollection)
    volume            (optional) reconstruction or ground truth
    @volume_axes_order  "zyx" (on-disk layout)
    @frame              "sample" | "lab"

    /align
      thetas          float32 (n_views, 5)
                      columns: [alpha, beta, phi, dx, dz]
      @gauge_fix_json (optional) JSON describing gauge projection

    /preprocess       (optional) provenance from tomojax-preprocess
```

> [!NOTE]
> Volumes are stored on disk in `(nz, ny, nx)` order with
> `@volume_axes_order="zyx"`. The `load_nxtomo()` function
> transposes them to internal `(nx, ny, nz)` order automatically.

### Preprocessing provenance

The `preprocess` collection records correction metadata:

- Source paths (`input_path`, `data_path`, `angles_path`)
- Frame counts and view filtering details
- Correction options (`log`, `epsilon`, `clip_min`)
- Mean flat/dark fields (compressed with `lzf`)
- Crop bounds and angular coverage before/after filtering

## Units and conventions

| Item | Convention |
|------|-----------|
| Angles | Stored in degrees (NX convention); internal math uses radians |
| Voxel/pixel sizes | Units `pixel` for simulated data |
| Rotation axis | Default `[0, 0, 1]` (+z); laminography tilt described in `/entry/geometry` |
| Compression | `lzf` by default; `gzip(level=4)` for smaller files |
| Array dtype | `float32` for interoperability and performance |

## Reading data in Python

```python
from tomojax.data.io_hdf5 import load_nxtomo

result = load_nxtomo("data/scan.nxs")
projections = result.projections   # (n_views, nv, nu), float32
grid = result.metadata.grid        # Grid instance or None
thetas = result.metadata.thetas_deg  # (n_views,), degrees
volume = result.metadata.volume    # (nx, ny, nz) or None
```

The returned volume (if present) is always in internal `(nx, ny, nz)`
order.

## Validating files

Use the `validate` CLI or the Python function:

```bash
uv run tomojax-validate data/scan.nxs
```

```python
from tomojax.data.io_hdf5 import validate_nxtomo

report = validate_nxtomo("data/scan.nxs")
if not report["issues"]:
    print("File is valid")
```

## Notes

- Alignment traces may be gauge-fixed. The `gauge_fix_json` attribute
  records whether `mean_translation` was applied to saved parameters.
- Use local SSD for writing to avoid NFS locking issues.
- Reproducibility manifests from `--save-manifest` are JSON sidecars,
  not embedded in the NXtomo file.
- Set `TOMOJAX_AXES_SILENCE=1` to suppress heuristic load-time
  axis-order warnings when processing large batches.

## Next steps

- [Python API reference](api.md) — `load_nxtomo()` and
  `save_nxtomo()` signatures
- [validate CLI](../cli/validate.md) — command-line validation
- [preprocess CLI](../cli/preprocess.md) — correcting raw frames
