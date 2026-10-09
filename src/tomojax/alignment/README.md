# tomojax.alignment

`tomojax.alignment` provides the building blocks behind `tj.align` and
`tomojax align`.

- `AlignConfig`: every solver setting. Its fields are also the keys of a
  `tomojax align --config` TOML file.
- `alignment_plan(mode, grid, quality=..., levels=..., freeze=...)`: the
  configuration and coarse-to-fine levels `tj.align` runs for a mode. Change a
  field with `dataclasses.replace(plan.config, ...)` and pass the result as
  `tj.align(scan, mode=..., config=...)`.
- `align_multires`: the array-level solver.

`tj.align(scan, checkpoint="align.ckpt")` saves progress after each outer
iteration and resumes an interrupted run from it. Defaults to per-view pose
correction (five parameters, six in a cone beam). Smooth pose models, setup
stages and mixed schedules are available when requested explicitly. See the
[alignment guide](../../../docs/alignment-guide.md).

Import from `tomojax.alignment` or `tomojax.alignment.api`.
