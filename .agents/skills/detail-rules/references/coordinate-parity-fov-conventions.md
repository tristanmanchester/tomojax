# Coordinate, Parity, and FOV Conventions

Maintain center-indexed coordinate conventions, preserve dimension parity in ROI grids, and ensure FOV utilities respect detector boundaries and physical rotations.

## What to look for

- **Center-Indexed Voxel Origin**: TomoJAX convention treats `vol_origin` as the physical location of voxel `(0, 0, 0)`'s **centre**.
- **Default Centered Origin**: The default `vol_origin` for an axis of length `n` and voxel size `v` must be `-((n / 2.0) - 0.5) * v`. Using `- (n / 2.0) * v` introduces a half-voxel shift.
- **Projector Index Mapping**: Mapping from physical position `pos` to voxel index must be `(pos - vol_origin) / v`. Integer indices must land exactly on voxel centers. There should be NO `-0.5` offset in this mapping when `vol_origin` is center-indexed.
- **ROI Parity Preservation**: When cropping grids to an ROI, the output grid dimensions should match the original grid's parity (odd/even) whenever possible to avoid off-by-half voxel shifts in the centered-origin convention. Use `_choose_shared_side` or `match_parity=True` in FOV helpers.
- **FOV Bound Enforcement**: All axes in the rotation plane (typically `x` and `y` for parallel-beam) must be bounded by the detector's horizontal FOV radius `r_u`. Helpers like `grid_from_detector_fov_cube` must not ignore the `y` axis (depth) fit.
- **Data Validation**: `Grid.vol_origin` and `Grid.vol_center` must be validated for length 3. `Detector.det_center` must be validated for length 2. Use `_coerce_fixed_tuple` in `__post_init__`.
- **Rotated Cube Extents**: The maximum XY extent for a 3D rotated cube of side `size` (worst-case arbitrary 3D rotation) is `size * sqrt(3) / 2`.

## Violation examples

- **Incorrect default origin (half-voxel shift)**:
  ```python
  ox = - (nx / 2.0) * vx
  ```
- **Double offset in projector mapping**:
  ```python
  ix0 = (q0[0] - vol_origin[0]) * inv_vx - 0.5
  ```
- **ROI crop ignoring Y-axis depth fit**:
  ```python
  side = min(int(info.nx_roi), int(info.nz_roi), int(grid.ny))
  ```
- **Parity flip from raw min() across dimensions**:
  ```python
  side = min(nx_fit, ny_fit)
  ```
- **Incorrect phantom extent (2D instead of 3D)**:
  ```python
  max_xy_extent = size * np.sqrt(2) / 2.0
  ```
- **Unvalidated Grid origin in CLI override**:
  ```python
  recon_grid = Grid(nx=NX, ny=NY, nz=NZ, vx=grid.vx, vy=grid.vy, vz=grid.vz)
  # Fails to copy/preserve vol_origin or vol_center
  ```

## Correct patterns

- **Center-indexed default origin**:
  ```python
  ox = -((grid.nx / 2.0) - 0.5) * grid.vx
  ```
- **Mapping to integer voxel indices**:
  ```python
  ix0 = (q0[0] - vol_origin[0]) * inv_vx
  ```
- **Parity-preserving shared side**:
  ```python
  side = _choose_shared_side(side_max, int(grid.nx), int(grid.ny))
  ```
- **3D rotated cube extent**:
  ```python
  max_xy_extent = size * np.sqrt(3) / 2.0
  ```
- **Runtime validation in Grid/Detector**:
  ```python
  object.__setattr__(self, "vol_origin", _coerce_fixed_tuple("vol_origin", self.vol_origin, 3))
  ```

## Scope

- `src/tomojax/core/projector.py`: Default origin and index mapping.
- `src/tomojax/core/geometry/`: Base classes and ray models.
- `src/tomojax/utils/fov.py`: ROI and FOV cropping utilities.
- `src/tomojax/data/phantoms.py`: Synthetic data generation.
- `src/tomojax/cli/`: CLI entry points resolving grids and ROIs.
