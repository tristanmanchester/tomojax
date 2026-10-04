# Whole-system persistent-cache screen, 2026-10-04

All 135 isolated workers completed: four cache conditions for every one of the 27 frozen reconstruction and six public alignment cells, plus three changed-data alignment controls. Every worker performs one initial call/search and one warm call from the original initialization. This is an instrumented screen with one repeat, not a new headline performance baseline.

## Decision

Cache reuse works for all ten JAX-based reconstruction selections and all six
alignment cells. With a populated cache, lowering the threshold from one second
to zero reduces diagnostic reconstruction time by factors of 1.06–1.43 in those
JAX paths and alignment time by 1.04–1.29. These single-repeat ratios are
observations, not headline speed claims. The 17 Fourier/CuPy selections show no
JAX cache activity and no consistent benefit.

Both original failures remain: sharp anisotropic 64 reconstruction and noisy
anisotropic alignment. The complete reconstruction aggregate and successful
whole-pilot alignment baseline remain unavailable. No goal gate or library
default changes. This subset improvement does not justify more cache tuning;
this line stops here.

Populated zero-threshold caches reduce sampled alignment memory from 320 MiB
to 192–196 MiB in these calls. They do not establish the required half-memory
advantage. Reusing a clean-data cache on noisy data preserves the expected
quality outcomes but still recompiles `jit_loss` and `jit_update`; 83–84 cache-hit events are logged for other compiled calls. The experiment therefore does not establish
compilation-free startup on a new acquisition.

Fresh-process minus warm time includes extra budget-search work and differences
in stopping, so it cannot isolate the sub-second startup target. The earlier
[whole-matrix profile](system-profile-2026-10-04.md) still identifies iterative
solve time and conditioning as major costs in large tilted scans. Further work
must address those costs alongside the two quality failures; saving compiled
code cannot change the underlying information or reconstruction error.

## What was tested

The experiment compares JAX’s default one-second minimum compilation time for caching against zero. Each setting has its own empty cache, followed by a fresh process reusing that populated cache. CuPy and CUDA driver cache policies stay unchanged: “empty” here refers only to the isolated JAX persistent cache. See [JAX’s cache configuration](https://docs.jax.dev/en/latest/persistent_compilation_cache.html#caching-thresholds).

The reconstruction method in each cell is its previously cold-fastest accepted TomoJAX workflow; the uncovered sharp anisotropic 64 cell uses general Joseph CGLS. This does not search for a different winning algorithm. The independent fixture definitions, budget ladder, quality gates, and numerical source stay fixed. The alignment path uses independent free voxels, the exact projector, central pose differences, pose elimination, and zero TV.

Compiler logging is enabled in every condition. A logged XLA compilation interval can include loading a cached executable; it is not proof of fresh compilation. Cache-hit names and keys are retained in the raw record, including solver kernels. Peak memory is sampled per PID at requested 10 ms intervals; short peaks can be missed.

## Reconstruction results

Times are seconds. The initial time uses the existing protocol: process startup plus all verified solve calls through the first accepted budget, or all attempted budgets on failure. It excludes benchmark result-file writes between those calls. Warm time measures the selected budget, or the final failed budget. Failed attempts are never accepted-result speedups. Relative L2 is full-volume error for the final initial attempt and the warm call, in that order.

### Cache threshold 1 s; initially empty

| Suite / cell | Status | Initial / warm s | Initial / warm image L2 | Peak GPU MiB | Cache hits / misses |
| --- | --- | ---: | ---: | ---: | ---: |
| gaussian-v1 / lamino-64-180 | accepted | 2.683 / 0.613 | 0.090326 / 0.090326 | 214 | 0 / 13 |
| structured-noisy-v1 / lamino-64-180 | accepted | 1.549 / 0.027 | 0.268277 / 0.268277 | 214 | 0 / 13 |
| structured-v1 / lamino-64-180 | accepted | 1.516 / 0.026 | 0.268145 / 0.268145 | 214 | 0 / 13 |
| gaussian-v1 / lamino-128-180 | accepted | 5.548 / 0.705 | 0.090988 / 0.090988 | 406 | 0 / 21 |
| structured-noisy-v1 / lamino-128-180 | accepted | 1.698 / 0.104 | 0.319490 / 0.319490 | 398 | 0 / 13 |
| structured-v1 / lamino-128-180 | accepted | 1.898 / 0.175 | 0.257526 / 0.257526 | 398 | 0 / 13 |
| gaussian-v1 / lamino-256-180 | accepted | 16.329 / 5.266 | 0.091453 / 0.091453 | 1174 | 0 / 21 |
| structured-noisy-v1 / lamino-256-180 | accepted | 3.309 / 0.803 | 0.317843 / 0.317843 | 1166 | 0 / 13 |
| structured-v1 / lamino-256-180 | accepted | 4.729 / 1.440 | 0.254374 / 0.254374 | 1166 | 0 / 13 |
| gaussian-v1 / anisotropic-64-180 | accepted | 0.576 / 0.002 | 0.013551 / 0.013551 | 146 | 0 / 0 |
| structured-noisy-v1 / anisotropic-64-180 | accepted | 0.573 / 0.002 | 0.178906 / 0.178906 | 146 | 0 / 0 |
| structured-v1 / anisotropic-64-180 | target_not_reached | 2.084 / 0.278 | 0.359895 / 0.359895 | 212 | 0 / 12 |
| gaussian-v1 / anisotropic-128-180 | accepted | 0.597 / 0.008 | 0.009877 / 0.009877 | 170 | 0 / 0 |
| structured-noisy-v1 / anisotropic-128-180 | accepted | 0.590 / 0.008 | 0.129038 / 0.129038 | 170 | 0 / 0 |
| structured-v1 / anisotropic-128-180 | accepted | 0.585 / 0.008 | 0.124626 / 0.124626 | 170 | 0 / 0 |
| gaussian-v1 / anisotropic-256-180 | accepted | 0.649 / 0.042 | 0.013707 / 0.013707 | 222 | 0 / 0 |
| structured-noisy-v1 / anisotropic-256-180 | accepted | 0.644 / 0.043 | 0.102425 / 0.102425 | 222 | 0 / 0 |
| structured-v1 / anisotropic-256-180 | accepted | 0.664 / 0.043 | 0.088213 / 0.088213 | 222 | 0 / 0 |
| gaussian-v1 / parallel-64-180 | accepted | 0.572 / 0.003 | 0.000115 / 0.000115 | 146 | 0 / 0 |
| structured-noisy-v1 / parallel-64-180 | accepted | 0.580 / 0.003 | 0.137725 / 0.137725 | 146 | 0 / 0 |
| structured-v1 / parallel-64-180 | accepted | 0.590 / 0.004 | 0.135290 / 0.135290 | 146 | 0 / 0 |
| gaussian-v1 / parallel-128-180 | accepted | 0.624 / 0.013 | 0.000134 / 0.000134 | 168 | 0 / 0 |
| structured-noisy-v1 / parallel-128-180 | accepted | 0.600 / 0.013 | 0.105897 / 0.105897 | 168 | 0 / 0 |
| structured-v1 / parallel-128-180 | accepted | 0.606 / 0.013 | 0.096056 / 0.096056 | 168 | 0 / 0 |
| gaussian-v1 / parallel-256-180 | accepted | 0.690 / 0.069 | 0.000149 / 0.000149 | 244 | 0 / 0 |
| structured-noisy-v1 / parallel-256-180 | accepted | 0.709 / 0.065 | 0.097765 / 0.097765 | 244 | 0 / 0 |
| structured-v1 / parallel-256-180 | accepted | 0.708 / 0.073 | 0.068552 / 0.068552 | 244 | 0 / 0 |

### Cache threshold 1 s; populated before process start

| Suite / cell | Status | Initial / warm s | Initial / warm image L2 | Peak GPU MiB | Cache hits / misses |
| --- | --- | ---: | ---: | ---: | ---: |
| gaussian-v1 / lamino-64-180 | accepted | 2.625 / 0.614 | 0.090326 / 0.090326 | 174 | 0 / 13 |
| structured-noisy-v1 / lamino-64-180 | accepted | 1.438 / 0.026 | 0.268277 / 0.268277 | 174 | 0 / 13 |
| structured-v1 / lamino-64-180 | accepted | 1.447 / 0.027 | 0.268145 / 0.268145 | 174 | 0 / 13 |
| gaussian-v1 / lamino-128-180 | accepted | 4.313 / 0.706 | 0.090988 / 0.090988 | 334 | 1 / 20 |
| structured-noisy-v1 / lamino-128-180 | accepted | 1.615 / 0.102 | 0.319490 / 0.319490 | 334 | 0 / 13 |
| structured-v1 / lamino-128-180 | accepted | 1.781 / 0.173 | 0.257526 / 0.257526 | 334 | 0 / 13 |
| gaussian-v1 / lamino-256-180 | accepted | 15.181 / 5.301 | 0.091453 / 0.091453 | 1174 | 1 / 20 |
| structured-noisy-v1 / lamino-256-180 | accepted | 3.142 / 0.810 | 0.317843 / 0.317843 | 1166 | 0 / 13 |
| structured-v1 / lamino-256-180 | accepted | 4.571 / 1.444 | 0.254374 / 0.254374 | 1166 | 0 / 13 |
| gaussian-v1 / anisotropic-64-180 | accepted | 0.584 / 0.002 | 0.013551 / 0.013551 | 146 | 0 / 0 |
| structured-noisy-v1 / anisotropic-64-180 | accepted | 0.561 / 0.002 | 0.178906 / 0.178906 | 146 | 0 / 0 |
| structured-v1 / anisotropic-64-180 | target_not_reached | 2.016 / 0.279 | 0.359895 / 0.359895 | 164 | 0 / 12 |
| gaussian-v1 / anisotropic-128-180 | accepted | 0.608 / 0.008 | 0.009877 / 0.009877 | 170 | 0 / 0 |
| structured-noisy-v1 / anisotropic-128-180 | accepted | 0.603 / 0.008 | 0.129038 / 0.129038 | 170 | 0 / 0 |
| structured-v1 / anisotropic-128-180 | accepted | 0.604 / 0.008 | 0.124626 / 0.124626 | 170 | 0 / 0 |
| gaussian-v1 / anisotropic-256-180 | accepted | 0.677 / 0.041 | 0.013707 / 0.013707 | 222 | 0 / 0 |
| structured-noisy-v1 / anisotropic-256-180 | accepted | 0.650 / 0.043 | 0.102425 / 0.102425 | 222 | 0 / 0 |
| structured-v1 / anisotropic-256-180 | accepted | 0.642 / 0.043 | 0.088213 / 0.088213 | 222 | 0 / 0 |
| gaussian-v1 / parallel-64-180 | accepted | 0.579 / 0.003 | 0.000115 / 0.000115 | 146 | 0 / 0 |
| structured-noisy-v1 / parallel-64-180 | accepted | 0.581 / 0.004 | 0.137725 / 0.137725 | 146 | 0 / 0 |
| structured-v1 / parallel-64-180 | accepted | 0.588 / 0.004 | 0.135290 / 0.135290 | 146 | 0 / 0 |
| gaussian-v1 / parallel-128-180 | accepted | 0.594 / 0.013 | 0.000134 / 0.000134 | 168 | 0 / 0 |
| structured-noisy-v1 / parallel-128-180 | accepted | 0.621 / 0.013 | 0.105897 / 0.105897 | 168 | 0 / 0 |
| structured-v1 / parallel-128-180 | accepted | 0.600 / 0.013 | 0.096056 / 0.096056 | 168 | 0 / 0 |
| gaussian-v1 / parallel-256-180 | accepted | 0.706 / 0.074 | 0.000149 / 0.000149 | 244 | 0 / 0 |
| structured-noisy-v1 / parallel-256-180 | accepted | 0.691 / 0.076 | 0.097765 / 0.097765 | 244 | 0 / 0 |
| structured-v1 / parallel-256-180 | accepted | 0.700 / 0.060 | 0.068552 / 0.068552 | 244 | 0 / 0 |

### Cache threshold 0 s; initially empty

| Suite / cell | Status | Initial / warm s | Initial / warm image L2 | Peak GPU MiB | Cache hits / misses |
| --- | --- | ---: | ---: | ---: | ---: |
| gaussian-v1 / lamino-64-180 | accepted | 2.696 / 0.615 | 0.090249 / 0.090249 | 214 | 0 / 13 |
| structured-noisy-v1 / lamino-64-180 | accepted | 1.511 / 0.027 | 0.268277 / 0.268277 | 214 | 0 / 13 |
| structured-v1 / lamino-64-180 | accepted | 1.561 / 0.028 | 0.268145 / 0.268145 | 214 | 0 / 13 |
| gaussian-v1 / lamino-128-180 | accepted | 5.541 / 0.702 | 0.090988 / 0.090988 | 406 | 0 / 21 |
| structured-noisy-v1 / lamino-128-180 | accepted | 1.688 / 0.102 | 0.319490 / 0.319490 | 398 | 0 / 13 |
| structured-v1 / lamino-128-180 | accepted | 1.872 / 0.172 | 0.257526 / 0.257526 | 398 | 0 / 13 |
| gaussian-v1 / lamino-256-180 | accepted | 16.607 / 5.316 | 0.091455 / 0.091455 | 1174 | 0 / 21 |
| structured-noisy-v1 / lamino-256-180 | accepted | 3.276 / 0.805 | 0.317843 / 0.317843 | 1166 | 0 / 13 |
| structured-v1 / lamino-256-180 | accepted | 4.686 / 1.438 | 0.254374 / 0.254374 | 1166 | 0 / 13 |
| gaussian-v1 / anisotropic-64-180 | accepted | 0.564 / 0.002 | 0.013551 / 0.013551 | 146 | 0 / 0 |
| structured-noisy-v1 / anisotropic-64-180 | accepted | 0.566 / 0.002 | 0.178906 / 0.178906 | 146 | 0 / 0 |
| structured-v1 / anisotropic-64-180 | target_not_reached | 2.128 / 0.279 | 0.359877 / 0.359877 | 212 | 0 / 12 |
| gaussian-v1 / anisotropic-128-180 | accepted | 0.589 / 0.008 | 0.009877 / 0.009877 | 170 | 0 / 0 |
| structured-noisy-v1 / anisotropic-128-180 | accepted | 0.596 / 0.008 | 0.129038 / 0.129038 | 170 | 0 / 0 |
| structured-v1 / anisotropic-128-180 | accepted | 0.611 / 0.008 | 0.124626 / 0.124626 | 170 | 0 / 0 |
| gaussian-v1 / anisotropic-256-180 | accepted | 0.669 / 0.043 | 0.013707 / 0.013707 | 222 | 0 / 0 |
| structured-noisy-v1 / anisotropic-256-180 | accepted | 0.666 / 0.045 | 0.102425 / 0.102425 | 222 | 0 / 0 |
| structured-v1 / anisotropic-256-180 | accepted | 0.667 / 0.041 | 0.088213 / 0.088213 | 222 | 0 / 0 |
| gaussian-v1 / parallel-64-180 | accepted | 0.567 / 0.003 | 0.000115 / 0.000115 | 146 | 0 / 0 |
| structured-noisy-v1 / parallel-64-180 | accepted | 0.590 / 0.003 | 0.137725 / 0.137725 | 146 | 0 / 0 |
| structured-v1 / parallel-64-180 | accepted | 0.577 / 0.004 | 0.135290 / 0.135290 | 146 | 0 / 0 |
| gaussian-v1 / parallel-128-180 | accepted | 0.615 / 0.013 | 0.000134 / 0.000134 | 168 | 0 / 0 |
| structured-noisy-v1 / parallel-128-180 | accepted | 0.635 / 0.013 | 0.105897 / 0.105897 | 168 | 0 / 0 |
| structured-v1 / parallel-128-180 | accepted | 0.603 / 0.013 | 0.096056 / 0.096056 | 168 | 0 / 0 |
| gaussian-v1 / parallel-256-180 | accepted | 0.709 / 0.077 | 0.000149 / 0.000149 | 244 | 0 / 0 |
| structured-noisy-v1 / parallel-256-180 | accepted | 0.703 / 0.069 | 0.097765 / 0.097765 | 244 | 0 / 0 |
| structured-v1 / parallel-256-180 | accepted | 0.701 / 0.076 | 0.068552 / 0.068552 | 244 | 0 / 0 |

### Cache threshold 0 s; populated before process start

| Suite / cell | Status | Initial / warm s | Initial / warm image L2 | Peak GPU MiB | Cache hits / misses |
| --- | --- | ---: | ---: | ---: | ---: |
| gaussian-v1 / lamino-64-180 | accepted | 2.224 / 0.618 | 0.090249 / 0.090249 | 166 | 13 / 0 |
| structured-noisy-v1 / lamino-64-180 | accepted | 1.043 / 0.027 | 0.268277 / 0.268277 | 166 | 13 / 0 |
| structured-v1 / lamino-64-180 | accepted | 1.013 / 0.026 | 0.268145 / 0.268145 | 166 | 13 / 0 |
| gaussian-v1 / lamino-128-180 | accepted | 3.328 / 0.701 | 0.090988 / 0.090988 | 326 | 21 / 0 |
| structured-noisy-v1 / lamino-128-180 | accepted | 1.185 / 0.102 | 0.319490 / 0.319490 | 326 | 13 / 0 |
| structured-v1 / lamino-128-180 | accepted | 1.355 / 0.174 | 0.257526 / 0.257526 | 326 | 13 / 0 |
| gaussian-v1 / lamino-256-180 | accepted | 14.267 / 5.332 | 0.091455 / 0.091455 | 1168 | 21 / 0 |
| structured-noisy-v1 / lamino-256-180 | accepted | 2.728 / 0.799 | 0.317843 / 0.317843 | 1158 | 13 / 0 |
| structured-v1 / lamino-256-180 | accepted | 4.132 / 1.434 | 0.254374 / 0.254374 | 1158 | 13 / 0 |
| gaussian-v1 / anisotropic-64-180 | accepted | 0.574 / 0.002 | 0.013551 / 0.013551 | 146 | 0 / 0 |
| structured-noisy-v1 / anisotropic-64-180 | accepted | 0.580 / 0.002 | 0.178906 / 0.178906 | 146 | 0 / 0 |
| structured-v1 / anisotropic-64-180 | target_not_reached | 1.571 / 0.279 | 0.359877 / 0.359877 | 156 | 12 / 0 |
| gaussian-v1 / anisotropic-128-180 | accepted | 0.605 / 0.008 | 0.009877 / 0.009877 | 170 | 0 / 0 |
| structured-noisy-v1 / anisotropic-128-180 | accepted | 0.592 / 0.008 | 0.129038 / 0.129038 | 170 | 0 / 0 |
| structured-v1 / anisotropic-128-180 | accepted | 0.600 / 0.008 | 0.124626 / 0.124626 | 170 | 0 / 0 |
| gaussian-v1 / anisotropic-256-180 | accepted | 0.681 / 0.042 | 0.013707 / 0.013707 | 222 | 0 / 0 |
| structured-noisy-v1 / anisotropic-256-180 | accepted | 0.641 / 0.043 | 0.102425 / 0.102425 | 222 | 0 / 0 |
| structured-v1 / anisotropic-256-180 | accepted | 0.639 / 0.053 | 0.088213 / 0.088213 | 222 | 0 / 0 |
| gaussian-v1 / parallel-64-180 | accepted | 0.584 / 0.003 | 0.000115 / 0.000115 | 146 | 0 / 0 |
| structured-noisy-v1 / parallel-64-180 | accepted | 0.573 / 0.003 | 0.137725 / 0.137725 | 146 | 0 / 0 |
| structured-v1 / parallel-64-180 | accepted | 0.578 / 0.003 | 0.135290 / 0.135290 | 146 | 0 / 0 |
| gaussian-v1 / parallel-128-180 | accepted | 0.596 / 0.013 | 0.000134 / 0.000134 | 168 | 0 / 0 |
| structured-noisy-v1 / parallel-128-180 | accepted | 0.609 / 0.013 | 0.105897 / 0.105897 | 168 | 0 / 0 |
| structured-v1 / parallel-128-180 | accepted | 0.604 / 0.013 | 0.096056 / 0.096056 | 168 | 0 / 0 |
| gaussian-v1 / parallel-256-180 | accepted | 0.692 / 0.078 | 0.000149 / 0.000149 | 244 | 0 / 0 |
| structured-noisy-v1 / parallel-256-180 | accepted | 0.709 / 0.092 | 0.097765 / 0.097765 | 244 | 0 / 0 |
| structured-v1 / parallel-256-180 | accepted | 0.691 / 0.072 | 0.068552 / 0.068552 | 244 | 0 / 0 |

## External comparison context

The external workflow below is the same cold-fastest accepted ASTRA/TIGRE workflow used by the frozen comparison, with that workflow’s sampled memory. These historical measurements were not rerun as part of the cache screen. Populated-cache TomoJAX timings must not be substituted into the uncached headline comparison. The complete matrix aggregate remains undefined while a cell has no accepted result.

| Suite / cell | External workflow | Historical cold / warm s | Peak GPU MiB |
| --- | --- | ---: | ---: |
| gaussian-v1 / lamino-64-180 | astra_cgls | 1.655 / 0.674 | 158 |
| structured-noisy-v1 / lamino-64-180 | astra_cgls | 0.328 / 0.029 | 158 |
| structured-v1 / lamino-64-180 | astra_cgls | 0.338 / 0.029 | 158 |
| gaussian-v1 / lamino-128-180 | astra_cgls | 6.953 / 3.191 | 202 |
| structured-noisy-v1 / lamino-128-180 | astra_cgls | 0.492 / 0.099 | 202 |
| structured-v1 / lamino-128-180 | astra_fbp_cgls | 0.633 / 0.107 | 374 |
| gaussian-v1 / lamino-256-180 | astra_cgls | 39.277 / 17.920 | 524 |
| structured-noisy-v1 / lamino-256-180 | astra_cgls | 2.110 / 0.713 | 524 |
| structured-v1 / lamino-256-180 | astra_cgls | 3.068 / 0.988 | 524 |
| gaussian-v1 / anisotropic-64-180 | astra_cgls | 0.334 / 0.030 | 150 |
| structured-noisy-v1 / anisotropic-64-180 | astra_fbp_cgls | 0.355 / 0.008 | 182 |
| structured-v1 / anisotropic-64-180 | No accepted workflow | — | — |
| gaussian-v1 / anisotropic-128-180 | astra_fbp3d_cupy | 0.362 / 0.010 | 242 |
| structured-noisy-v1 / anisotropic-128-180 | astra_fbp3d_cupy | 0.363 / 0.010 | 242 |
| structured-v1 / anisotropic-128-180 | astra_fbp3d_cupy | 0.372 / 0.010 | 242 |
| gaussian-v1 / anisotropic-256-180 | astra_fbp3d_cupy | 0.463 / 0.077 | 544 |
| structured-noisy-v1 / anisotropic-256-180 | astra_fbp3d_cupy | 0.455 / 0.074 | 544 |
| structured-v1 / anisotropic-256-180 | astra_fbp3d_cupy | 0.465 / 0.076 | 544 |
| gaussian-v1 / parallel-64-180 | astra_fbp2d | 0.291 / 0.024 | 140 |
| structured-noisy-v1 / parallel-64-180 | astra_fbp2d | 0.296 / 0.027 | 140 |
| structured-v1 / parallel-64-180 | astra_fbp2d | 0.310 / 0.027 | 140 |
| gaussian-v1 / parallel-128-180 | astra_fbp2d | 0.318 / 0.059 | 140 |
| structured-noisy-v1 / parallel-128-180 | astra_fbp2d | 0.326 / 0.060 | 140 |
| structured-v1 / parallel-128-180 | astra_fbp2d | 0.341 / 0.061 | 140 |
| gaussian-v1 / parallel-256-180 | astra_fbp2d | 0.596 / 0.298 | 140 |
| structured-noisy-v1 / parallel-256-180 | astra_fbp2d | 0.587 / 0.297 | 140 |
| structured-v1 / parallel-256-180 | astra_fbp2d | 0.607 / 0.301 | 140 |

## Public free-voxel alignment

All cold calls and warm repeats begin from zero voxels and nominal poses. The image, rotation and shift gates and common-frame verification are unchanged. Initial/warm differences can include differing outer-iteration counts; subtracting them does not isolate startup time.

| Cell | Threshold s / cache phase | Status | Initial / warm s | Initial / warm image L2 | Initial / warm rotation RMSE ° | Initial / warm shift RMSE px | Peak GPU MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lamino-clean | 1 / empty | accepted | 20.779 / 13.234 | 0.082013 / 0.085785 | 0.006687 / 0.009878 | 0.000098 / 0.000140 | 320 |
| lamino-clean | 1 / populated | accepted | 16.377 / 12.288 | 0.085834 / 0.086744 | 0.009661 / 0.009874 | 0.000143 / 0.000140 | 202 |
| lamino-clean | 0 / empty | accepted | 18.645 / 13.048 | 0.086614 / 0.084712 | 0.009942 / 0.008206 | 0.000132 / 0.000127 | 320 |
| lamino-clean | 0 / populated | accepted | 14.748 / 12.967 | 0.085767 / 0.084412 | 0.009525 / 0.008179 | 0.000138 / 0.000123 | 196 |
| lamino-noisy | 1 / empty | accepted | 21.206 / 14.191 | 0.082077 / 0.080415 | 0.009609 / 0.009250 | 0.000246 / 0.000241 | 320 |
| lamino-noisy | 1 / populated | accepted | 16.378 / 11.367 | 0.081617 / 0.084909 | 0.008223 / 0.009525 | 0.000233 / 0.000247 | 202 |
| lamino-noisy | 0 / empty | accepted | 17.650 / 12.980 | 0.084549 / 0.081955 | 0.009789 / 0.009355 | 0.000244 / 0.000241 | 320 |
| lamino-noisy | 0 / populated | accepted | 14.791 / 12.078 | 0.083053 / 0.083839 | 0.009556 / 0.009859 | 0.000245 / 0.000236 | 196 |
| anisotropic-clean | 1 / empty | accepted | 11.192 / 4.670 | 0.001508 / 0.001516 | 0.008795 / 0.008893 | 0.000117 / 0.000121 | 320 |
| anisotropic-clean | 1 / populated | accepted | 7.804 / 4.592 | 0.001517 / 0.001495 | 0.008841 / 0.008447 | 0.000113 / 0.000118 | 198 |
| anisotropic-clean | 0 / empty | accepted | 10.844 / 4.389 | 0.001484 / 0.001492 | 0.008752 / 0.008794 | 0.000111 / 0.000110 | 320 |
| anisotropic-clean | 0 / populated | accepted | 6.165 / 4.373 | 0.001486 / 0.001432 | 0.008979 / 0.008822 | 0.000110 / 0.000104 | 192 |
| anisotropic-noisy | 1 / empty | target_not_reached | 49.425 / 43.207 | 0.003484 / 0.003483 | 0.016270 / 0.016237 | 0.000383 / 0.000383 | 320 |
| anisotropic-noisy | 1 / populated | target_not_reached | 46.382 / 43.314 | 0.003484 / 0.003481 | 0.016248 / 0.016242 | 0.000384 / 0.000381 | 198 |
| anisotropic-noisy | 0 / empty | target_not_reached | 49.571 / 42.970 | 0.003484 / 0.003485 | 0.016387 / 0.016421 | 0.000383 / 0.000384 | 320 |
| anisotropic-noisy | 0 / populated | target_not_reached | 44.752 / 42.821 | 0.003483 / 0.003483 | 0.016152 / 0.016251 | 0.000381 / 0.000381 | 192 |
| parallel-clean | 1 / empty | accepted | 10.308 / 4.129 | 0.002319 / 0.002322 | 0.004046 / 0.003731 | 0.000112 / 0.000111 | 320 |
| parallel-clean | 1 / populated | accepted | 7.292 / 4.137 | 0.002323 / 0.002317 | 0.003975 / 0.003814 | 0.000112 / 0.000109 | 202 |
| parallel-clean | 0 / empty | accepted | 10.254 / 3.918 | 0.002318 / 0.002322 | 0.003804 / 0.003786 | 0.000113 / 0.000112 | 320 |
| parallel-clean | 0 / populated | accepted | 5.682 / 3.900 | 0.002313 / 0.002317 | 0.003745 / 0.003710 | 0.000110 / 0.000111 | 196 |
| parallel-noisy | 1 / empty | accepted | 10.260 / 4.153 | 0.005331 / 0.005338 | 0.007872 / 0.008261 | 0.000251 / 0.000247 | 320 |
| parallel-noisy | 1 / populated | accepted | 7.304 / 4.131 | 0.005341 / 0.005340 | 0.008203 / 0.007901 | 0.000248 / 0.000249 | 202 |
| parallel-noisy | 0 / empty | accepted | 10.305 / 3.934 | 0.005331 / 0.005337 | 0.007750 / 0.008313 | 0.000253 / 0.000246 | 320 |
| parallel-noisy | 0 / populated | accepted | 5.674 / 3.903 | 0.005341 / 0.005336 | 0.008139 / 0.008318 | 0.000247 / 0.000246 | 196 |
| lamino-noisy | 0 / noisy data, clean cache | accepted | 19.695 / 13.999 | 0.082156 / 0.081661 | 0.009814 / 0.009669 | 0.000245 / 0.000244 | 202 |
| anisotropic-noisy | 0 / noisy data, clean cache | target_not_reached | 46.088 / 43.088 | 0.003485 / 0.003481 | 0.016207 / 0.016242 | 0.000381 / 0.000381 | 198 |
| parallel-noisy | 0 / noisy data, clean cache | accepted | 6.876 / 3.922 | 0.005334 / 0.005335 | 0.008298 / 0.008081 | 0.000250 / 0.000246 | 202 |

## Provenance

Frozen numerical/driver source SHA-256: `5ff8791482a48008c71a01ae8a0a8a6b57e03b6e9b403eb95eb6b00254696415`. The runner verifies it is unchanged. The current checkout differs from that frozen library only in the subsequent IO cleanup; numerical implementation is unchanged. No other GPU job ran concurrently. A brief CPU-only evidence-archive cleanup overlapped early diagnostics; this further limits timing claims from the single-repeat screen.

The retained record contains the launch script, exact environment, fixture hashes, all solve histories and failures, compiler cache keys, cache sizes, and sampled peak process memory. The corresponding source archive is [the pose-elimination snapshot](../bench/reference/public-free-voxel-v1-schur-source.tar.gz).

Artifacts: [all runs and launch metadata](../bench/reference/system-cache-2026-10-04.json.gz),
[derived per-condition summary](../bench/reference/system-cache-2026-10-04-summary.json).
Read compressed records with the [evidence guide](../bench/reference/README.md).
