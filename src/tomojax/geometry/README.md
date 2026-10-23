# tomojax.geometry

`tomojax.geometry` describes a scan: the reconstruction `Grid`, the `Detector`,
the geometry classes `ParallelGeometry`, `LaminographyGeometry`,
`ConeGeometry` (with `ConeBeam`) and `ConeSegments`, and field-of-view helpers
such as `compute_roi`, `grid_from_detector_fov` and `cylindrical_mask_xy`.

`tomojax.geometry.api` holds the expert helpers alignment, IO and the solvers
build on: volume axis order constants and `transpose_volume`, calibration
state (`CalibrationState`, `GeometryState`, `validate_calibration_gauges`),
axis poses and detector grids.

Import from these two modules, not from private implementation files.

## Frames and positive angles

Physical coordinates are `(x, y, z)`, independent of array storage order.
In the fixed detector frame, the unrotated detector columns (`u`) point along
`+x`, rows (`v`) along `+z`, and the beam travels towards `+y`. The cone source
is at `(0, -source_to_axis, source_height)` before detector/axis offsets.
`Grid(nx, ny, nz)` names physical dimensions; projection arrays are
`(view, v, u)`.

Angles are in degrees and rotate the **object**, not the source/detector.
For an untilted axis, angle zero aligns object and detector-frame axes; positive
angles use the right-hand rule about `+z`: `+x` turns towards `+y`. Thus an
object point `(x, y, z)` has detector-frame horizontal coordinate
`x*cos(angle) - y*sin(angle)` before cone magnification and offsets. Looking
from `+z` towards the origin, positive rotation is counterclockwise.

For laminography or a tilted cone axis, angle zero includes the fixed alignment
`S` that maps the object's `+z` to the declared lab-frame rotation axis. The
world-from-object rotation is `S @ Rz(angle)`. Detector roll, pitch and yaw
change the detector axes, not the definition or sign of the view angle.

ASTRA and TIGRE keep the object fixed and move the source/detector. Exported
rays therefore use the inverse object pose: an object turn of `+angle` gives
a beam turn of `-angle`. `Scan.to_astra()` supplies the physical vectors; the
benchmark TIGRE adapter represents those same vectors as TIGRE's source,
detector offsets and intrinsic ZYZ angles. Do not negate stored scan angles
or mirror a sinogram in addition to this conversion.
