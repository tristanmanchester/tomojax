# Known Limitations

The product spine is intentionally narrower than the full research repository.
The shipped package focuses on the public CLI/API workflows listed in
[`support-matrix.md`](support-matrix.md).

- Broad, truth-free laminography alignment remains a research area rather than a
  broad product claim.
- Detector-centre/COR alignment is exposed through the product CLI/profile path;
  fixed-truth and scan-specific diagnostics are not product workflows.
- Object-frame drift recovery, nuisance fitting as a headline workflow, and
  combined bad-view/jump handling are not promoted as public product workflows.
- Pallas paths remain implementation/backend support rather than the default
  correctness claim. Reference/JAX paths are the baseline for public smoke tests.
- The test suite is intentionally small. It proves import boundaries, CLI shape,
  a CPU reconstruction smoke workflow, and numerical correctness of select math
  primitives (loss kernels, geometry transforms).
