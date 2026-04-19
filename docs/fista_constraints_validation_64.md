# FISTA Constraint Validation, 64^3

This note records the smoke validation used when adding optional positivity and
box constraints to the FISTA-TV reconstruction path. It is a correctness and
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

## 64^3 Constraint Comparison

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
