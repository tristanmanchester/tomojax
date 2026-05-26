# Known Limitations

The shipped package covers the workflows listed in
[`support-matrix.md`](support-matrix.md), including 5-DOF pose alignment,
detector-centre/COR alignment, and mixed setup and pose correction.

## Alignment limitations

TomoJAX handles many pose and setup errors, but some cases remain expert
workflows or need scan-specific review.

- Pose-only correction can absorb some setup errors. The reconstruction may
  look good, but the recovered parameters won't reflect true geometry unless
  you also run `--mode cor`.
- Mixed setup and pose correction has gauge ambiguity. Use `--mode auto` only
  with an explicit `--gauge-policy`, such as `anchor_mean`.
- Detector-v or sample-elevation reference shifts are physically ambiguous
  and not reliably recoverable.
- Abrupt jumps and short bursts of bad views need more robust diagnostics or
  specialized workflows.
- Large combined setup and pose errors can still need staged initialization,
  stronger priors, or manual review.

## Implementation limitations

- Pallas paths are optional accelerator backends. Reference/JAX paths are the
  baseline for correctness tests.
- The test suite covers import boundaries, CLI routing, CPU reconstruction
  smoke tests, and numerical correctness of loss kernels and geometry
  transforms.

## Next steps

See [`alignment-guide.md`](alignment-guide.md) to choose an alignment mode.
