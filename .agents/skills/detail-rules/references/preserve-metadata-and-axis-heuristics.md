# Preserve Metadata and Axis Heuristics

Verify that CLI overrides and IO operations preserve essential volume metadata, alignment parameters, and axis order heuristics, preventing silent configuration loss or spatial shifts.

## What to look for

- **Grid Metadata Preservation:** When overriding volume dimensions (e.g., via `--grid` in CLIs), ensure that `vol_origin` and `vol_center` are preserved. Use `dataclasses.replace` on an existing `Grid` object instead of manual reconstruction.
- **Centralized Geometry Construction:** CLI modules should use `build_geometry_from_meta` (or `build_nominal_geometry_from_meta`) from `src/tomojax/data/geometry_meta.py`. Avoid manual instantiation of `ParallelGeometry` or `LaminographyGeometry` in CLI entry points.
- **Alignment Application:** Reconstruction workflows (`recon.py`) must build geometry with `apply_saved_alignment=True` to ensure `align_params` and `angle_offset_deg` are used. Conversely, alignment workflows (`align.py`) typically use `apply_saved_alignment=False` for the starting point.
- **ROI vs. Grid Overrides:** If an explicit `--grid` override is provided, it must take precedence over ROI-derived masks. Specifically, ensure `apply_cyl_mask` is set to `False` if `grid_override` is not `None`.
- **Robust IO Metadata Handling:**
  - **HDF5:** Use `_attr_to_str` helper for all HDF5 attribute reads to handle varying dtypes (bytes, numpy scalars). Wrap `json.loads` calls in `try...except` to prevent crashes on malformed metadata.
  - **NPZ:** Extracted metadata objects (like dicts stored in NPZ) may be returned as 0-D numpy arrays; use `.item()` to retrieve the underlying Python object.
- **Axis Order Heuristics:** `infer_disk_axes` in `src/tomojax/utils/axes.py` should return `None` (ambiguous) if `grid` metadata is missing, rather than guessing based on shape alone (which fails for non-cubic volumes).

## Violation examples

### Dropping Grid Metadata in CLI
```python
# VIOLATION: Manual construction drops vol_origin and vol_center
if args.grid is not None:
    NX, NY, NZ = map(int, args.grid)
    recon_grid = Grid(nx=NX, ny=NY, nz=NZ, vx=grid.vx, vy=grid.vy, vz=grid.vz)
```

### Manual Geometry Build in CLI
```python
# VIOLATION: Manual build ignores saved alignment and angle offsets
if meta.get("geometry_type") == "parallel":
    geom = ParallelGeometry(grid=recon_grid, detector=detector, thetas_deg=meta["thetas_deg"])
```

### Unprotected JSON Parsing in IO
```python
# VIOLATION: Crashes on malformed metadata JSON
out["grid"] = json.loads(entry.attrs["grid_meta_json"])
```

### Masking Overrides Data
```python
# VIOLATION: Cylindrical mask persists even with explicit grid override
if grid_override is not None:
    NX, NY, NZ = map(int, grid_override)
    recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)
    # Missing: apply_cyl_mask = False
```

## Correct patterns

### Preserving Metadata via replace()
```python
# CORRECT: Preserves all metadata except the overridden dimensions
from dataclasses import replace
if args.grid is not None:
    NX, NY, NZ = map(int, args.grid)
    recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)
```

### Using Centralized Geometry Builder
```python
# CORRECT: Handles alignment, angle offsets, and grid overrides safely
from ..data.geometry_meta import build_geometry_from_meta
grid, detector, geom = build_geometry_from_meta(
    meta, 
    grid_override=grid_override, 
    apply_saved_alignment=True
)
```

### Robust Attribute Reading
```python
# CORRECT: Uses helper and error handling
grid_meta = entry.attrs.get("grid_meta_json")
if grid_meta is not None:
    s = _attr_to_str(grid_meta)
    if s:
        try:
            out["grid"] = json.loads(s)
        except Exception:
            pass
```

## Scope

- `src/tomojax/cli/*.py`: CLI entry points and geometry resolution logic.
- `src/tomojax/data/io_hdf5.py`: Metadata serialization and deserialization.
- `src/tomojax/utils/axes.py`: Axis order inference and transpositions.
- `src/tomojax/recon/multires.py`: Geometry scaling across resolution levels.