# Alignment Inner Solver 64³ Smoke Benchmark

This note records a small CPU smoke benchmark comparing FISTA-TV and SPDHG-TV as
the inner reconstruction solver for joint alignment. It is not a production
performance claim; it is a reproducibility note for the first SPDHG alignment
integration.

## Environment

- Date: 2026-04-19
- Backend: JAX CPU
- Device: `TFRT_CPU_0`
- Volume shape: `64 x 64 x 64`
- Detector shape: `64 x 64`
- Phantom: centered cube
- Misalignment: translation-only `dx,dz`, Gaussian sigma about `0.35 px`
- Alignment optimizer: Gauss-Newton
- Active DOFs: `dx,dz`
- Alignment loss: `l2`
- TV weight: `0.001`

## Results

Equal tiny inner iteration counts are not a fair comparison because one FISTA
iteration is a full-gradient reconstruction iteration, while one SPDHG iteration
is one stochastic subset update. This pilot used `12` views, `outer_iters=1`,
`recon_iters=2`, and `views_per_batch=2` for both solvers:

| Solver | Wall time | Initial trans RMSE | Final trans RMSE | Alignment loss | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| FISTA | `7.59s` | `0.2997` | `0.2277` | `557957.6 -> 551694.8` | Moved alignment |
| SPDHG | `10.29s` | `0.2997` | `0.2997` | `3142472.8 -> 3142472.8` | Finite, but no pose movement |

With more appropriate inner work for SPDHG, both solvers improved the alignment.
These runs used `12` views, `outer_iters=2`, `views_per_batch=2`, FISTA
`recon_iters=5`, and SPDHG `recon_iters=30`.

| Seed | Solver | Wall time | Recon time total | Initial RMSE | Final RMSE | RMSE ratio | Total loss reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | FISTA | `15.22s` | `11.27s` | `0.2997` | `0.1547` | `0.516` | `75.1%` |
| 0 | SPDHG | `31.67s` | `26.31s` | `0.2997` | `0.1433` | `0.478` | `84.8%` |
| 1 | FISTA | `13.05s` | `11.13s` | `0.2197` | `0.0780` | `0.355` | `73.7%` |
| 1 | SPDHG | `25.80s` | `24.11s` | `0.2197` | `0.0895` | `0.408` | `78.7%` |

Average over the two `12`-view cases:

| Solver | Average wall time | Average final RMSE | Average RMSE ratio | Average loss reduction |
| --- | ---: | ---: | ---: | ---: |
| FISTA | `14.14s` | `0.1163` | `0.436` | `74.4%` |
| SPDHG | `28.74s` | `0.1164` | `0.443` | `81.8%` |

A slightly larger view-count smoke used `24` views, `outer_iters=1`,
`views_per_batch=4`, FISTA `recon_iters=3`, and SPDHG `recon_iters=30`:

| Solver | Wall time | Recon time | Initial RMSE | Final RMSE | RMSE ratio | Loss reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FISTA | `12.40s` | `8.97s` | `0.3323` | `0.1600` | `0.481` | `4.6%` |
| SPDHG | `24.94s` | `23.33s` | `0.3323` | `0.0895` | `0.269` | `37.0%` |

## Interpretation

SPDHG alignment works on these `64³` smoke cases: it runs end-to-end, returns
finite volumes and alignment parameters, records SPDHG-specific stats, and
improves both loss and translation RMSE when given enough subset updates.

FISTA remains the better default for small CPU/debug runs. It moves poses with
very few inner iterations and was faster in these CPU tests.

SPDHG should be treated as a scalability path rather than a same-iteration-count
replacement. It needs more inner iterations because each iteration updates a
subset of views. In the `24`-view smoke test, SPDHG gave better alignment quality
than the short FISTA run, but it was slower on CPU.

The current SPDHG path also pays setup cost for automatic step-size/operator-norm
estimation. If SPDHG alignment becomes a primary production path, caching or
reusing that estimate, or exposing explicit step-size controls for alignment,
would be worth evaluating.
