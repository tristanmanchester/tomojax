# tomojax.io

## Purpose

`tomojax.io` owns the public input/output boundary for measured projection
datasets plus typed helpers for serialising TomoJAX metadata, manifests, and
artifact schemas.

## Public API

- `JsonValue`
- `LoadedNXTomo`
- `NXTomoMetadata`
- `ProjectionDataset`
- `RealLaminographyInput`
- `ValidationReport`
- `absorption_to_transmission(...)`
- `load_projection_payload(path)`
- `load_nxtomo(path)`
- `save_projection_payload(path, projections=..., metadata=...)`
- `save_nxtomo(path, projections=..., metadata=...)`
- `build_geometry_from_dataset_metadata(...)`
- `convert_dataset(input_path, output_path)`
- `inspect_dataset(path)`
- `format_inspection_report(report)`
- `flat_dark_to_absorption(...)`
- `flat_dark_to_transmission(...)`
- `save_projection_quicklook(path, output_path)`
- `load_dataset(path)`
- `load_real_laminography_input(path, flip_u=False, flip_v=False, transpose_detector=False)`
- `load_tiff_stack(path, angles_deg=...)`
- `PreprocessConfig`
- `PreprocessResult`
- `preprocess_nxtomo(input_path, output_path, config=None)`
- `preprocess_tiff_stack(projections_path, flats_path, darks_path, angles_path, output_path, config=None)`
- `save_dataset(path, dataset)`
- `validate_dataset(path)`
- `validate_nxtomo(path)`
- `transmission_to_absorption(...)`
- `normalize_json(...)`
- `drop_none(...)`

## Dependencies

This module may depend on `tomojax.core` data structures and low-level dataset
readers owned by the IO layer. It must not depend on alignment, reconstruction,
benchmark, or CLI implementation modules.

## Invariants

- Public helpers return strict JSON-compatible values.
- Real input datasets cross into TomoJAX through `ProjectionDataset`.
- Solver-heavy commands use `ProjectionDataset.geometry_inputs()` and
  `ProjectionDataset.copy_metadata()` rather than importing lower-level data
  payloads directly.
- TIFF stack loading requires explicit angle metadata.
- Raw NXtomo flat/dark correction crosses through the public IO facade before
  producing corrected projection datasets.
- TIFF stack preprocessing is explicit: callers provide separate projection,
  flat, dark, and angle sidecars rather than relying on broad dispatch magic.
- Real laminography NX/HDF5 loading is explicit: callers use
  `load_real_laminography_input(...)` for the measured beamline layout and pass
  detector orientation transforms at the IO boundary.
- Preprocessing writes reconstruction-ready absorption/log-attenuation
  projections by default. `PreprocessConfig(output_domain="transmission")` is
  available for workflows that explicitly need normalized transmission.
- Preprocessing provenance records the correction formula, epsilon/clip policy,
  frame counts, selected/rejected views where applicable, crop bounds, source
  paths, output domain, and dark/flat override status.
- Non-finite floats are converted to strings so callers can use
  `json.dump(..., allow_nan=False)`.
- Optional NumPy and JAX arrays are converted without making either library a
  module boundary dependency for callers.

## Tests

Covered by `tests/test_json_utils.py`, `tests/test_io_public_dataset.py`,
`tests/test_real_laminography_io.py`, and downstream manifest/checkpoint tests.
