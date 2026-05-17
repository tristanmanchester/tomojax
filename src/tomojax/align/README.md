# tomojax.align

`tomojax.align` owns the product alignment workflow. The package root is kept deliberately small:

- `AlignConfig`
- `align`
- `align_multires`

`tomojax.align.api` exposes the typed configuration, schedule, loss, profile, geometry-state, and checkpoint-adjacent helpers needed by the product CLI. Historical alternating benchmark runners, Schur diagnostics, continuation probes, and v1-parity artifact writers have been removed from the product spine and archived outside the package.

Product code should import through `tomojax.align` or `tomojax.align.api`. Private stage modules remain implementation details.
