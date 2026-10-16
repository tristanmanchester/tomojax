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
iteration and resumes an interrupted run from it. Without expert overrides,
`tj.align` chooses centre estimation for axial parallel scans and guarded
setup, support and motion selection for unposed laminography; its result records
conditional outcomes and limitations. Explicit `mode="pose"` selects per-view
corrections (five parameters, six in a cone beam). Smooth pose models, setup
stages and mixed schedules remain available explicitly. See the
[alignment guide](../../../docs/alignment-guide.md).

Joint alignment retries without its derivative cache before skipping a level
that does not fit GPU memory.

Import from `tomojax.alignment` or `tomojax.alignment.api`.
