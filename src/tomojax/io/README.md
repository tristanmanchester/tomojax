# tomojax.io

## Purpose

`tomojax.io` owns the public input/output boundary for measured projection
datasets plus typed helpers for serialising TomoJAX metadata, manifests, and
artifact schemas.

## Public API

- `JsonValue`
- `ProjectionDataset`
- `ValidationReport`
- `load_projection_payload(path)`
- `save_projection_payload(path, projections=..., metadata=...)`
- `build_geometry_from_dataset_metadata(...)`
- `convert_dataset(input_path, output_path)`
- `inspect_dataset(path)`
- `format_inspection_report(report)`
- `save_projection_quicklook(path, output_path)`
- `load_dataset(path)`
- `load_tiff_stack(path, angles_deg=...)`
- `PreprocessConfig`
- `PreprocessResult`
- `preprocess_nxtomo(input_path, output_path, config=None)`
- `save_dataset(path, dataset)`
- `validate_dataset(path)`
- `normalize_json(...)`
- `drop_none(...)`

## Dependencies

This module may depend on `tomojax.core` data structures and transitional
low-level data readers while the v2 IO boundary is being consolidated. It must
not depend on alignment, reconstruction, benchmark, or CLI implementation
modules.

## Invariants

- Public helpers return strict JSON-compatible values.
- Real input datasets cross into TomoJAX through `ProjectionDataset`.
- Solver-heavy commands use `ProjectionDataset.geometry_inputs()` and
  `ProjectionDataset.copy_metadata()` rather than importing legacy data payloads
  directly.
- TIFF stack loading requires explicit angle metadata.
- Raw NXtomo flat/dark correction crosses through the public IO facade before
  producing corrected projection datasets.
- Non-finite floats are converted to strings so callers can use
  `json.dump(..., allow_nan=False)`.
- Optional NumPy and JAX arrays are converted without making either library a
  module boundary dependency for callers.

## Tests

Covered by `tests/test_json_utils.py`, `tests/test_io_public_dataset.py`, and
downstream manifest/checkpoint tests.
