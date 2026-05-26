# tomojax.align

`tomojax.align` owns the product alignment workflow. The package root is kept
deliberately small and exposes the stable Python alignment surface.

- `AlignConfig`
- `align`
- `align_multires`

The default Python posture uses per-view 5-DOF pose correction. Smooth pose
models, setup stages, and mixed setup and pose schedules remain available when
callers request them explicitly.

Product code should import through `tomojax.align` or `tomojax.align.api`.
Private stage modules remain implementation details.
