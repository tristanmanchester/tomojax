# inspect

The `tomojax-inspect` command reads metadata and projection
statistics from an NXtomo/HDF5 file without running JAX or
reconstructing a volume. It's useful for checking data shape,
angle coverage, NaN/Inf counts, and memory estimates before
committing to a reconstruction.

```
tomojax-inspect <scan.nxs> [--json <report.json>] \
  [--quicklook <preview.png>]
```

## Output fields

The command prints a structured summary to stdout. The following
table lists every field in the report.

| Field | Description |
|-------|-------------|
| Projection shape | `[n_views, nv, nu]` array dimensions |
| Dtype | Storage data type of the projection dataset |
| Views | Number of projection views |
| Detector shape | `{nv, nu}` detector pixel counts |
| Stats: min | Minimum finite voxel value |
| Stats: p01 | 1st percentile of finite values |
| Stats: mean | Mean of finite values |
| Stats: p50 | Median (50th percentile) of finite values |
| Stats: p99 | 99th percentile of finite values |
| Stats: max | Maximum finite voxel value |
| NaN count | Number of NaN elements in projections |
| +Inf count | Number of positive infinity elements |
| -Inf count | Number of negative infinity elements |
| Inf total | Sum of +Inf and -Inf counts |
| Angle coverage | Total angular range in degrees (max - min) |
| Angle min/max | Smallest and largest rotation angles |
| Angle count | Number of angle entries |
| Angle units | Unit string from the angle dataset attributes |
| Geometry type | `parallel` or `lamino` (from `/entry/geometry`) |
| Geometry metadata | Whether the JSON geometry blob is present |
| Detector metadata | `nu`, `nv`, `du`, `dv`, `det_center` |
| Flats/darks | Flat-field and dark-field frame counts |
| Alignment parameters | Shape of saved alignment params, plus flags for angle offset, misalignment spec, and gauge fix |
| Memory estimates | Heuristic fp32 working-set sizes for `fbp_fp32`, `fista_tv_fp32`, and `spdhg_tv_fp32` in bytes, based on the inferred reconstruction grid |

## JSON output

The `--json report.json` flag writes a stable, machine-readable
version of the same report. Keys are snake_case and nested by
section (e.g., `projection.stats.mean`,
`memory_estimates.modes.fbp_fp32.estimated_working_set_bytes`).

JSON output doesn't suppress stdout -- you get both the
human-readable summary and the JSON file.

## Quicklook

The `--quicklook preview.png` flag writes the central projection
view as a percentile-scaled PNG. This is a fast way to visually
check that the data loaded correctly before running a full
reconstruction.

## Examples

Basic inspection:

```bash
tomojax-inspect data/scan.nxs
```

JSON report with a quicklook preview:

```bash
tomojax-inspect data/scan.nxs \
  --json runs/scan.inspect.json \
  --quicklook runs/scan.projection.png
```

Sample output:

```
TomoJAX inspection: data/scan.nxs
Projection shape: [180, 256, 256]
Dtype: float32
Views: 180
Detector shape: {'nv': 256, 'nu': 256}
Stats: min=0, p01=0, mean=0.12, p50=0.08, p99=1.7, max=2.3
NaN/Inf counts: nan=0, +inf=0, -inf=0, inf_total=0
Angle coverage: 179 deg (min=0, max=179, count=180, units=degree)
Geometry type: parallel
Geometry metadata: not found
Detector metadata: nu=256, nv=256, du=1.0, dv=1.0, det_center=[0.0, 0.0]
Flats/darks: not found (image_key present; no flat/dark frames)
Alignment parameters: not found
Memory estimates: grid=[256, 256, 256], fbp_fp32=181403648 bytes, fista_tv_fp32=382730240 bytes, spdhg_tv_fp32=382730240 bytes
```

## Notes

Missing optional metadata is reported as "not found". Valid
projection data are enough for a successful inspection -- the
command doesn't require geometry metadata, alignment parameters,
or flat/dark fields to be present.

---

See also: [data format](../reference/data-format.md)
| [CLI overview](index.md)
