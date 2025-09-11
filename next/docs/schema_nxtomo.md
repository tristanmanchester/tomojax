% NXtomo Schema for TomoJAX v2

Primary file type: HDF5 with NeXus NXtomo conventions. Default extension: `.nxs`.

## Required Structure
- `/entry (NXentry)`
  - `definition = "NXtomo"`
  - `/instrument (NXinstrument)`
    - `/detector (NXdetector)`
      - `data` (float32, shape `(n_views, nv, nu)`, chunks `(1,256,256)`) — projections
      - `x_pixel_size` (float, units: `pixel`)
      - `y_pixel_size` (float, units: `pixel`)
  - `/sample (NXsample)`
    - `/transformations (NXtransformations)`
      - `rotation_angle` (float32, shape `(n_views,)`, units `degree`)
      - `rotation_axis` (float32, shape `(3,)`, default `[0,0,1]`)
      - `depends_on = "rotation_angle"`
  - `/data (NXdata)`
    - `projections` -> link to `/entry/instrument/detector/data`
    - `signal = "projections"`

## TomoJAX Extras
- `/entry/geometry/type = "parallel" | "lamino" | "custom"`
- `/entry/@grid_meta_json`: JSON-serialized Grid
  - `{ nx, ny, nz, vx, vy, vz, vol_origin?, vol_center? }`
- `/entry/instrument/detector/@detector_meta_json`: JSON-serialized Detector
  - `{ nu, nv, du, dv, det_center }`
- `/entry/processing (NXprocess)/tomojax_next (NXcollection)`
  - `volume` (optional GT, shape `(nz,ny,nx)`, compression `lzf`)
  - `align/thetas` (optional, shape `(n_views,5)`, columns=`[alpha,beta,phi,dx,dz]`)

## Units & Conventions
- Angles stored in degrees (NX convention). Internal math uses radians.
- Voxel/detector sizes: units `pixel` for simulated data unless specified.
- Rotation axis defaults to world +z; laminography adds tilt via geometry metadata.

## Notes
- Use `lzf` compression by default; `gzip(level=4)` for smaller files if needed.
- Prefer local SSD for writing to avoid NFS locking issues.
- Keep arrays float32 for interoperability and performance.
