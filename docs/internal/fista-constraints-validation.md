# FISTA Constraint Validation, 64^3

This note records smoke validation for optional positivity/box constraints and
regulariser selection in the FISTA reconstruction path. It is a correctness and
regression check, not a production benchmark.

## Regression Commands

The focused solver, CLI, config, and manifest tests passed with:

```bash
uv run pytest -q \
  tests/test_recon.py \
  tests/test_recon_math_fixes.py \
  tests/test_cli_config.py \
  tests/test_cli_entrypoints.py
```

Result:

```text
63 passed in 26.60s
```

The broader FISTA-focused regression set passed with:

```bash
uv run pytest -q \
  tests/test_recon.py \
  tests/test_integration.py \
  tests/test_multires.py
```

Result:

```text
29 passed in 17.91s
```

The combined post-review check passed with:

```bash
uv run pytest -q \
  tests/test_recon.py \
  tests/test_recon_math_fixes.py \
  tests/test_cli_config.py \
  tests/test_cli_entrypoints.py \
  tests/test_integration.py \
  tests/test_multires.py
```

Result:

```text
85 passed in 39.63s
```

`git diff --check` was clean.

After adding Huber-TV, the expanded solver/config regression set passed with:

```bash
uv run pytest -q \
  tests/test_recon_math_fixes.py \
  tests/test_cli_entrypoints.py \
  tests/test_tv_ops.py \
  tests/test_recon.py \
  tests/test_spdhg.py \
  tests/test_cli_config.py
```

Result:

```text
78 passed in 48.37s
```

## 64^3 Constraint Comparison

This is the original FISTA-TV constraint smoke run.

Environment:

- Backend: CPU
- Volume: synthetic nonnegative `64 x 64 x 64` phantom
- Geometry: parallel beam
- Views: `32`
- Detector: `64 x 64`
- FISTA iterations: `10`
- TV weight: `lambda_tv=0.001`
- TV prox iterations: `5`
- Shared fixed Lipschitz estimate: `L=1957.8265380859375`

Setup timings:

| Stage | Time |
| --- | ---: |
| Projection generation | `2.242s` |
| Power-method estimate | `4.121s` |

Results:

| Variant | Min | Max | Negative voxels | Lower violations | Upper violations | Final loss | RMSE vs GT | PSNR vs GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | `-0.065069` | `1.095137` | `62391` | n/a | n/a | `10633.34` | `0.04865` | `27.05` |
| positivity | `0.000000` | `1.087733` | `0` | `0` | n/a | `10035.64` | `0.04733` | `27.23` |
| lower `0.05` | `0.050000` | `1.038984` | `0` | `0` | n/a | `471346.28` | `0.07238` | `23.14` |
| upper `0.80` | `-0.063358` | `0.800000` | `35952` | n/a | `0` | `94472.83` | `0.06786` | `23.37` |
| box `0.00..0.80` | `0.000000` | `0.800000` | `0` | `0` | `0` | `94803.38` | `0.06715` | `23.46` |
| positivity + box `0.05..0.80` | `0.050000` | `0.800000` | `0` | `0` | `0` | `522207.91` | `0.07901` | `22.05` |

## Notes

- The unconstrained default still permits negative voxels, preserving previous
  FISTA behavior when no constraint options are set.
- `positivity=True` removed all negative voxels and slightly improved RMSE/PSNR
  on this nonnegative phantom.
- Upper-only constraints intentionally do not imply positivity; the upper-only
  run still had negative voxels while reporting zero upper-bound violations.
- Box-constrained runs had zero lower and upper violations.
- The tighter/lower-shifted boxes scored worse against this phantom because they
  imposed constraints that do not match the phantom background and intensity
  distribution.

## 64^3 TV vs Huber-TV Constraint Comparison

This follow-up smoke run checks that the new Huber-TV regulariser remains finite
and respects the same FISTA constraint projections on the same problem class.
The run intentionally uses a mild regularisation weight, so TV and Huber-TV are
expected to be close; unit tests cover the Huber-TV small-delta TV-like limit and
quadratic near-zero-gradient behavior.

Environment:

- Backend: CPU (`TFRT_CPU_0`)
- Volume: synthetic nonnegative `64 x 64 x 64` phantom
- Geometry: parallel beam
- Views: `32`
- Detector: `64 x 64`
- FISTA iterations: `10`
- Regularisation weight: `lambda_tv=0.001`
- TV prox iterations: `5`
- Huber transition radius: `huber_delta=0.05`
- Data Lipschitz estimate: `L_data=1957.70556640625`
- Huber smooth-step Lipschitz estimate: `L_huber=1957.94556640625`

Setup timings:

| Stage | Time |
| --- | ---: |
| Projection generation | `6.952s` |
| Power-method estimate | `10.353s` |

Results:

| Regulariser | Variant | Min | Max | Negative voxels | Lower violations | Upper violations | Final loss | RMSE vs GT | PSNR vs GT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tv` | none | `-0.06030` | `1.06495` | `72819` | n/a | n/a | `9045.52` | `0.04425` | `27.63` |
| `tv` | positivity | `0.00000` | `1.06005` | `0` | n/a | n/a | `8609.88` | `0.04307` | `27.82` |
| `tv` | lower `0.05` | `0.05000` | `1.01602` | `0` | `0` | n/a | `453558.66` | `0.06932` | `23.32` |
| `tv` | upper `0.80` | `-0.05638` | `0.80000` | `52989` | n/a | `0` | `40768.12` | `0.05294` | `25.52` |
| `tv` | box `0.00..0.80` | `0.00000` | `0.80000` | `0` | `0` | `0` | `40960.88` | `0.05206` | `25.67` |
| `tv` | positivity + box `0.05..0.80` | `0.05000` | `0.80000` | `0` | `0` | `0` | `467828.66` | `0.07084` | `22.99` |
| `huber_tv` | none | `-0.06031` | `1.06496` | `71416` | n/a | n/a | `9045.68` | `0.04425` | `27.63` |
| `huber_tv` | positivity | `0.00000` | `1.06007` | `0` | n/a | n/a | `8611.23` | `0.04307` | `27.82` |
| `huber_tv` | lower `0.05` | `0.05000` | `1.01603` | `0` | `0` | n/a | `453559.78` | `0.06932` | `23.32` |
| `huber_tv` | upper `0.80` | `-0.05638` | `0.80000` | `51598` | n/a | `0` | `40767.36` | `0.05294` | `25.52` |
| `huber_tv` | box `0.00..0.80` | `0.00000` | `0.80000` | `0` | `0` | `0` | `40961.31` | `0.05206` | `25.67` |
| `huber_tv` | positivity + box `0.05..0.80` | `0.05000` | `0.80000` | `0` | `0` | `0` | `467829.69` | `0.07084` | `22.99` |

Notes:

- Huber-TV completed all constraint variants with finite losses and volumes.
- The unconstrained default still permits negative voxels for both regularisers.
- Positivity removed all negative voxels and slightly improved RMSE/PSNR on this
  nonnegative phantom for both regularisers.
- Upper-only constraints capped maxima but did not imply positivity; negative
  voxels remained in upper-only runs.
- Box-constrained runs had zero lower and upper violations for both
  regularisers.
- The `0.05` lower floor worsened RMSE/PSNR because the phantom background is
  truly zero, so that constraint is intentionally mismatched.
