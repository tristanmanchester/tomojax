# Known Limitations

The product spine is narrower than the full research repository. The shipped
package focuses on the public CLI/API workflows listed in
[`support-matrix.md`](support-matrix.md), including 5-DOF pose alignment,
detector-centre/COR alignment, and expert mixed setup and pose correction.

## Alignment limitations

TomoJAX can recover useful reconstructions from many pose and setup errors, but
some correction paths remain expert workflows or need scan-specific review.

- Pose-only correction can absorb some setup errors. Treat those results as
  reconstruction-quality corrections unless you also run a setup-specific mode
  such as `--mode cor`.
- Mixed setup and pose correction has gauge ambiguity. Use `--mode auto` only
  with an explicit `--gauge-policy`, such as `anchor_mean`.
- Detector-v or sample-elevation reference shifts are physically ambiguous.
  Don't treat `det_v_px` as a normal recoverable alignment target.
- Abrupt jumps and short bursts of bad views need more robust diagnostics or
  specialized workflows before they become headline product claims.
- Large combined setup and pose errors can still need staged initialization,
  stronger priors, or manual review.

## Implementation limitations

These implementation constraints shape the current support level and test
coverage.

- Pallas paths remain implementation/backend support rather than the default
  correctness claim. Reference/JAX paths are the baseline for public smoke
  tests.
- The test suite is intentionally small. It proves import boundaries, CLI
  shape, a CPU reconstruction smoke workflow, and numerical correctness of
  select math primitives, including loss kernels and geometry transforms.

## Next steps

Use [`alignment-guide.md`](alignment-guide.md) to choose a public alignment
path. Use this page when you need to decide whether a scan result is a
supported product workflow, an expert diagnostic, or a research case.
