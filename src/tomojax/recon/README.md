# tomojax.recon

`tomojax.recon` owns the supported reconstruction routines:

- `fbp`
- `fista_tv`
- `spdhg_tv`

The public facade also exports the associated config types, regulariser type, support-mask helper, and chunked backprojection helper used by the product CLI. Historical reference-FISTA diagnostics, trace recomputation utilities, scout-support builders, and v1-parity reconstruction helpers have been moved out of the product package.
