# tomojax.align

`tomojax.align` owns the product alignment workflow. The package root is kept deliberately small:

- `AlignConfig`
- `align`
- `align_multires`


Product code should import through `tomojax.align` or `tomojax.align.api`. Private stage modules remain implementation details.
