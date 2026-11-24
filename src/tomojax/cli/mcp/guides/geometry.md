# Inspect ray geometry

Detector pitch, centre and grid dimensions share the acquisition's length unit.
Keep the origin fixed when reproducing a saved grid. Inspect the per-view
transform and rays to check a declared geometry.

```python
from tomojax import geometry
grid = tj.Grid.from_dict(scan.grid.to_dict())
detector = tj.Detector.from_dict(scan.detector.to_dict())
coarse = grid.resampled(shape=tuple(max(1, n // 2) for n in grid.shape))
origin = geometry.grid_volume_origin(grid)
g: geometry.ScanGeometry = scan.geometry
registered = (g.grid.shape, g.detector.nu, len(g.angles))
pose = g.pose_matrix(0)
rays = g.rays_for_view(0)
```

For a cone scan, inspect the beam and detector plane or convert it to ASTRA's
(rows, views, columns) order for an external tool.

```python
beam = geometry.beam_of(scan.geometry)
saved = tj.ConeBeam.from_dict(beam.to_dict())
source = saved.source()
frame = saved.detector_frame(scan.detector)
magnification = saved.magnification
sinogram, projection_geometry, volume_geometry = scan.to_astra()
restored = tj.Scan.from_astra(sinogram, projection_geometry, volume_geometry)
```

For a multi-orbit acquisition whose two known cone geometries share a grid,
inspect the combined transforms alongside the single-orbit metadata.

```python
g = geometry.ConeSegments((first_orbit, second_orbit))
poses = g.stack_poses(len(g.angles), np.float32)
axis = first_orbit.rotation_axis()
metadata = first_orbit.geometry_metadata()
first_poses = first_orbit.pose_matrices()
```
