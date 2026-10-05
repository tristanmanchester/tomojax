# Public Huber-TV screen, 2026-10-04

All six scheduled cells completed one cold and one warm call. All 12 calls failed the fixed rotation gate. This fixed-weight variant is rejected; no runtime defaults change, and this line of tuning stops.

The only configuration change from the [pose-eliminated run](public-free-voxel-schur-2026-10-04.md) is `lambda_tv=0.005`, the existing library default, replacing zero TV. This tests an image prior motivated by the [local noise analysis](pilot-noise-2026-10-04.md). Fixtures, zero-volume/nominal-pose initialization, positivity, exact projector, central Jacobian, damping, iteration budgets, and acceptance gates remain fixed.

| Cell | Accepted calls | Cold attempt s | Warm attempt s | Cold image relative L2 | Cold rotation RMSE ° | Cold shift-vector RMSE px | Peak process GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parallel-clean | 0/2 | 56.475 | 51.478 | 0.001964 | 0.012021 | 0.000065 | 320 |
| parallel-noisy | 0/2 | 56.434 | 51.351 | 0.003796 | 0.014170 | 0.000241 | 320 |
| anisotropic-clean | 0/2 | 49.222 | 44.122 | 0.001708 | 0.059487 | 0.000712 | 320 |
| anisotropic-noisy | 0/2 | 48.975 | 44.372 | 0.002698 | 0.063693 | 0.000924 | 320 |
| lamino-clean | 0/2 | 66.382 | 61.469 | 0.055341 | 0.024644 | 0.000207 | 320 |
| lamino-noisy | 0/2 | 66.501 | 61.108 | 0.055339 | 0.023630 | 0.000290 | 320 |

The gates remain rotation RMSE ≤0.01°, shift-vector RMSE ≤0.05 pixels, and full-volume relative L2 ≤0.10 for parallel/anisotropic or ≤0.20 for laminography, scored in one common rigid object frame. These are failed-attempt costs, not accepted-result timings. The screen has one warm repeat per cell and is not eligible as a headline timing baseline.

Cold time includes process startup, imports, loading, setup, compilation, transfers, solving, and verification. Warm calls restart from the original initialization. Peak GPU memory is sampled per worker PID at requested 10 ms intervals and can miss short peaks. Hardware is the same RTX 4070 Laptop GPU as the preceding comparison.

The prior improves some image errors but biases pose recovery past the gate in every cell. This does not reject all image priors or prove a universal noise limit. It rejects this preselected weight under the frozen protocol. No accepted-result speedup is defined, and the 27-cell reconstruction matrix is unaffected.

## Provenance

The run used the unchanged frozen pose-elimination source snapshot, SHA-256 `5ff8791482a48008c71a01ae8a0a8a6b57e03b6e9b403eb95eb6b00254696415`. The [raw manifest](../../bench/reference/public-free-voxel-v1-tv-screen.json.gz) retains every completed call, quality result, configuration, history, fixture hash, source metadata, and the launch/resume scripts. An interrupted anisotropic-noisy worker is retained separately in that manifest; only unfinished cells were restarted. It is not counted among the 12 completed calls.
