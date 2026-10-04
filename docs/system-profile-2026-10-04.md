# Whole-matrix diagnostic attribution, 2026-10-04

All 27 frozen cells were profiled: 46 workflows and 92 cold/warm phases. Each cell includes general Joseph CGLS and, if different, its cold-fastest accepted TomoJAX method. These are diagnostic selected-budget calls with cProfile and Nsight overhead, not replacements for the [uninstrumented accepted-result score](system-matrix-2026-10-04.md). They do not repeat the full budget search.

Nsight Systems 2026.5.1 produced readable traces. The earlier 2023.3 profiler returned exit zero after fatal CUDA UUID errors; those incomplete traces were discarded from attribution. The source snapshot matches the retained launch archive.

CUDA active time below is the union of traced kernels, memory operations and whole CUDA graph intervals. Internal graph nodes were not individually traced; these profiles cannot distinguish the graph's projector and adjoint kernels. Summing kernel events alone would incorrectly classify graph execution as idle time. Compilation is cProfile cumulative time in JAX backend_compile_and_load; it can overlap GPU activity, so the columns are not an additive wall-time decomposition.

[Raw attribution](../bench/reference/system-matrix-v2-profile-attribution.json) retains all 46 workflows, including failed solves.

| Suite / cell | Profiled method | Cold total / compile ms | Cold CUDA active ms | Warm total / CUDA active ms | Accepted in profile |
|---|---|---:|---:|---:|---|
| gaussian-v1 / parallel-64-180 | tomojax_fourier_cupy | 1005.3 / 0.0 | 1.4 | 5.5 / 0.6 | True |
| gaussian-v1 / anisotropic-64-180 | tomojax_fourier_cupy | 998.3 / 0.0 | 1.1 | 2.8 / 0.3 | True |
| gaussian-v1 / lamino-64-180 | tomojax_joseph_cgls_pallas | 4386.3 / 2044.7 | 605.9 | 616.2 / 600.1 | True |
| gaussian-v1 / parallel-128-180 | tomojax_fourier_cupy | 1001.6 / 0.0 | 3.9 | 15.3 / 3.1 | True |
| gaussian-v1 / anisotropic-128-180 | tomojax_fourier_cupy | 1094.4 / 0.0 | 2.4 | 10.1 / 1.7 | True |
| gaussian-v1 / lamino-128-180 | tomojax_multires_joseph_cgls_pallas | 8443.9 / 5195.5 | 745.7 | 710.5 / 679.8 | True |
| gaussian-v1 / parallel-256-180 | tomojax_fourier_cupy | 1146.9 / 0.0 | 21.4 | 70.8 / 20.6 | True |
| gaussian-v1 / anisotropic-256-180 | tomojax_fourier_cupy | 1086.2 / 0.0 | 11.4 | 49.9 / 10.3 | True |
| gaussian-v1 / lamino-256-180 | tomojax_multires_joseph_cgls_pallas | 13057.1 / 5298.3 | 5278.5 | 5277.9 / 5206.2 | True |
| structured-v1 / parallel-64-180 | tomojax_fourier_cupy | 884.3 / 0.0 | 1.4 | 5.0 / 0.6 | True |
| structured-v1 / anisotropic-64-180 | tomojax_joseph_cgls_pallas | 3967.9 / 2135.4 | 282.8 | 284.3 / 269.6 | False |
| structured-v1 / lamino-64-180 | tomojax_joseph_cgls_pallas | 3543.9 / 1947.0 | 26.1 | 32.2 / 19.0 | True |
| structured-v1 / parallel-128-180 | tomojax_fourier_cupy | 940.2 / 0.0 | 4.0 | 13.9 / 3.3 | True |
| structured-v1 / anisotropic-128-180 | tomojax_fourier_cupy | 945.5 / 0.0 | 2.5 | 9.1 / 1.7 | True |
| structured-v1 / lamino-128-180 | tomojax_joseph_cgls_pallas | 3746.2 / 1958.0 | 164.5 | 180.9 / 157.8 | True |
| structured-v1 / parallel-256-180 | tomojax_fourier_cupy | 1053.0 / 0.0 | 21.5 | 75.4 / 20.6 | True |
| structured-v1 / anisotropic-256-180 | tomojax_fourier_cupy | 992.9 / 0.0 | 11.3 | 42.7 / 10.2 | True |
| structured-v1 / lamino-256-180 | tomojax_joseph_cgls_pallas | 5173.1 / 2119.3 | 1346.7 | 1401.3 / 1325.5 | True |
| structured-noisy-v1 / parallel-64-180 | tomojax_fourier_cupy | 915.7 / 0.0 | 1.4 | 4.9 / 0.6 | True |
| structured-noisy-v1 / anisotropic-64-180 | tomojax_fourier_cupy | 946.8 / 0.0 | 1.1 | 2.7 / 0.4 | True |
| structured-noisy-v1 / lamino-64-180 | tomojax_joseph_cgls_pallas | 3603.0 / 1951.3 | 25.9 | 30.8 / 19.0 | True |
| structured-noisy-v1 / parallel-128-180 | tomojax_fourier_cupy | 946.3 / 0.0 | 3.9 | 14.1 / 3.2 | True |
| structured-noisy-v1 / anisotropic-128-180 | tomojax_fourier_cupy | 943.3 / 0.0 | 2.5 | 9.4 / 1.7 | True |
| structured-noisy-v1 / lamino-128-180 | tomojax_joseph_cgls_pallas | 3733.8 / 1973.0 | 93.8 | 108.3 / 85.8 | True |
| structured-noisy-v1 / parallel-256-180 | tomojax_fourier_cupy | 1071.3 / 0.0 | 21.6 | 79.3 / 20.9 | True |
| structured-noisy-v1 / anisotropic-256-180 | tomojax_fourier_cupy | 990.7 / 0.0 | 11.5 | 43.3 / 10.3 | True |
| structured-noisy-v1 / lamino-256-180 | tomojax_joseph_cgls_pallas | 4540.0 / 2084.2 | 732.7 | 788.7 / 718.5 | True |

The slow cold ratios and the long tilted solve have different limiting costs. In noisy tilted 64, the instrumented call takes 3603 ms with 1951 ms of backend compilation and 26 ms of CUDA activity. In Gaussian tilted 256, plain CGLS spends about 43.2 seconds executing CUDA work on a warm failed solve; its accepted coarse-to-fine alternative spends about 5.2 seconds executing CUDA work and incurs 5.3 seconds of backend compilation on the cold selected-budget call.

These measurements support reducing general solver setup/compilation and improving convergence across tilted and irregular geometry. They do not justify copy overlap, tile sizes, or gather tuning as the next whole-system change. The failed sharp anisotropic 64 gate still requires a quality improvement; startup changes alone cannot supply it.
