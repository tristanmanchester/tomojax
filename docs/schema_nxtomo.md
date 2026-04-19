% NXtomo Schema for TomoJAX v2

Primary file type: HDF5 with NeXus NXtomo conventions. Default extension: `.nxs`.

## Required Structure
- `/entry (NXentry)`
  - `definition = "NXtomo"`
  - `/instrument (NXinstrument)`
    - `/detector (NXdetector)`
      - `data` (float32, shape `(n_views, nv, nu)`, chunks `(1,256,256)`) — absorption projections (`-log` transmission)
      - `image_key` (int32, shape `(n_views,)`) — NXtomo frame kind; TomoJAX uses `0=sample/projection`, `1=flat`, `2=dark`
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
- `/entry/geometry (NXcollection)/type = "parallel" | "lamino"`
- `/entry/geometry/@geometry_meta_json` (optional): JSON with geometry‑specific metadata, e.g., for laminography `{ "tilt_deg": <float>, "tilt_about": "x"|"z" }`
- `/entry/@grid_meta_json`: JSON-serialized Grid
  - `{ nx, ny, nz, vx, vy, vz, vol_origin?, vol_center? }`
- `/entry/instrument/detector/@detector_meta_json`: JSON-serialized Detector
  - `{ nu, nv, du, dv, det_center }`
- `/entry/processing (NXprocess)/tomojax (NXcollection)`
  - `volume` (optional GT or reconstruction, stored in-memory as `(nx,ny,nz)`; written on disk as `(nz,ny,nx)` when `@volume_axes_order="zyx"`, compression `lzf`)
  - `@volume_axes_order` (attr on `tomojax`, default `"zyx"`; loaders always return internal `xyz` volume order)
  - `@frame` (optional attr on `tomojax`): `"sample"|"lab"` indicates the frame of the saved volume (default `sample`)
  - `align/thetas` (optional, shape `(n_views,5)`, columns=`[alpha,beta,phi,dx,dz]`)
  - `preprocess` (optional NXcollection): provenance for `tomojax-preprocess`
    - attrs: `schema_version`, `input_path`, `data_path`, `angles_path`, `image_key_path`, `frame_counts`, `output_domain`, `epsilon`, `clip_min`, `output_dtype`, `correction_formula`, `log`, `assume_dark_field`, `assume_flat_field`, `warning_counts`
    - view/ROI attrs when applicable: `select_views`, `reject_views`, `select_views_file`, `reject_views_file`, `view_selection`, `final_sample_view_indices`, `final_raw_frame_indices`, `auto_reject`, `crop`, `crop_bounds`, `original_projection_shape`, `cropped_projection_shape`, `final_projection_shape`, `angular_coverage_before`, `angular_coverage_after`
    - `flat_mean` and `dark_mean`: mean correction fields used for output, compressed with `lzf`
- Reproducibility manifests from `tomojax-recon --save-manifest` and `tomojax-align --save-manifest` are JSON sidecars; they are not embedded in the NXtomo file.

## Units & Conventions
- Angles stored in degrees (NX convention). Internal math uses radians.
- Voxel/detector sizes: units `pixel` for simulated data unless specified.
- Transformations: `rotation_angle` is stored in degrees; a default `rotation_axis=[0,0,1]` is written for NX compatibility. The exact geometry (e.g., laminography tilt) is described in `/entry/geometry` (tagged `NXcollection`) and used by TomoJAX.

## Notes
- Use `lzf` compression by default; `gzip(level=4)` for smaller files if needed.
- Prefer local SSD for writing to avoid NFS locking issues.
- Keep arrays float32 for interoperability and performance.
- Set `TOMOJAX_AXES_SILENCE=1` to silence heuristic load-time warnings when processing large batches.
- Raw NXtomo preprocessing expects mixed sample/flat/dark frames in the same detector stack.
  `tomojax-preprocess` writes sample-only output with all output `image_key` values reset to `0`.
- `tomojax-preprocess` view selection/rejection uses sample-view indices after `image_key == 0`
  filtering. Detector ROI crop specs use projection axis order `y0:y1,x0:x1`; cropped output
  updates detector `nv`, `nu`, and `det_center`.
