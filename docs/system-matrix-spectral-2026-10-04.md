# Shared spectral conditioning: complete frozen matrix, 2026-10-04

The spectral candidate passed 9/54 method/cell combinations and failed 45/54. At least one candidate passed in 8/27 cells, versus 25/27 for these two original workflows. Every worker completed its cold budget search and seven warm calls, including failures. It is rejected and withdrawn from the working library. The full-matrix speedup remains undefined; no passing-only aggregate is used.

Only the optional eight-probe geometry-derived positive Fourier preconditioner changed the ordinary and multiresolution Joseph CGLS workflows. Operators, zero initialization, existing coarse-to-fine policy, 180-view batch, budgets 1–256 and quality gates stayed fixed. Solves remain unconstrained and unregularized. The same method applies to all three data suites and geometries; there is no per-cell tuning. See the [independent algebra and alignment diagnostics](shared-conditioning-2026-10-04.md).

| Suite | Workflow | Previous accepted / 9 | Spectral accepted / 9 | Previously passing cells that now fail |
|---|---|---:|---:|---|
| gaussian-v1 | ordinary | 8 | 1 | lamino-64-180, parallel-64-180, lamino-128-180, anisotropic-128-180, parallel-128-180, anisotropic-256-180, parallel-256-180 |
| gaussian-v1 | multiresolution | 9 | 5 | lamino-64-180, lamino-128-180, lamino-256-180, anisotropic-256-180 |
| structured-v1 | ordinary | 8 | 0 | lamino-64-180, parallel-64-180, lamino-128-180, anisotropic-128-180, parallel-128-180, lamino-256-180, anisotropic-256-180, parallel-256-180 |
| structured-v1 | multiresolution | 8 | 1 | lamino-64-180, parallel-64-180, lamino-128-180, anisotropic-128-180, lamino-256-180, anisotropic-256-180, parallel-256-180 |
| structured-noisy-v1 | ordinary | 8 | 0 | lamino-64-180, parallel-64-180, lamino-128-180, anisotropic-128-180, parallel-128-180, lamino-256-180, anisotropic-256-180, parallel-256-180 |
| structured-noisy-v1 | multiresolution | 8 | 2 | lamino-64-180, lamino-128-180, anisotropic-128-180, lamino-256-180, anisotropic-256-180, parallel-256-180 |

Cold time sums process startup/loading, setup, transfers, compilation, every budget attempted and verification. Warm calls independently reconstruct the same data from the workflow's initial state; failed rows repeat the final 256-iteration attempt. GPU execution was serial and exclusive. Peak process GPU memory is sampled by PID at requested 10 ms intervals across imports, search and repeats; brief peaks may be missed. Raw records retain all quality values, stopping reasons, iteration counts and samples.

## gaussian-v1

S = ordinary spectral CGLS; M = existing multiresolution policy with spectral CGLS at each level. `FAIL` times are unsuccessful attempts, never accepted-result times. Quality is final cold-search relative L2; the best value across the entire search is also shown. Every failed row still includes seven warm attempts at the final budget.

| Cell | Method | Status | Budget | Cold search/attempt s | Warm median [min, max] s | Final / best search L2 | Gate | GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| lamino-64-180 | S | FAIL | 256 | 2.964 | 0.623 [0.622, 0.624] | 0.364921 / 0.364921 | 0.10 | 230 |
| lamino-64-180 | M | FAIL | 256 | 4.876 | 0.202 [0.200, 0.204] | 0.152588 / 0.152588 | 0.10 | 296 |
| anisotropic-64-180 | S | PASS | 256 | 2.347 | 0.299 [0.296, 0.300] | 0.0281322 / 0.0281322 | 0.03 | 228 |
| anisotropic-64-180 | M | PASS | 128 | 4.464 | 0.062 [0.062, 0.063] | 0.0190175 / 0.0190175 | 0.03 | 294 |
| parallel-64-180 | S | FAIL | 256 | 2.770 | 0.499 [0.498, 0.500] | 0.0307769 / 0.0307769 | 0.03 | 230 |
| parallel-64-180 | M | PASS | 32 | 4.251 | 0.042 [0.040, 0.043] | 0.0200083 / 0.0200083 | 0.03 | 296 |
| lamino-128-180 | S | FAIL | 256 | 12.310 | 5.059 [5.030, 5.076] | 2.84072 / 2.84072 | 0.10 | 414 |
| lamino-128-180 | M | FAIL | 256 | 7.328 | 0.824 [0.822, 0.827] | 0.152724 / 0.152724 | 0.10 | 416 |
| anisotropic-128-180 | S | FAIL | 256 | 6.490 | 2.198 [2.187, 2.199] | 0.0733682 / 0.0733682 | 0.03 | 302 |
| anisotropic-128-180 | M | PASS | 128 | 5.391 | 0.212 [0.211, 0.214] | 0.0207285 / 0.0207285 | 0.03 | 304 |
| parallel-128-180 | S | FAIL | 256 | 10.351 | 4.067 [4.050, 4.080] | 0.0478015 / 0.0478015 | 0.03 | 414 |
| parallel-128-180 | M | PASS | 64 | 5.811 | 0.242 [0.239, 0.244] | 0.023769 / 0.023769 | 0.03 | 416 |
| lamino-256-180 | S | FAIL | 256 | 95.954 | 47.748 [46.040, 47.956] | 6.52736 / 4.53252 | 0.10 | 1182 |
| lamino-256-180 | M | FAIL | 256 | 26.713 | 6.877 [6.797, 6.885] | 0.167063 / 0.167063 | 0.10 | 1184 |
| anisotropic-256-180 | S | FAIL | 256 | 43.003 | 19.721 [19.656, 19.757] | 0.824008 / 0.824008 | 0.03 | 926 |
| anisotropic-256-180 | M | FAIL | 256 | 13.447 | 2.691 [2.681, 2.726] | 0.0706405 / 0.0706405 | 0.03 | 928 |
| parallel-256-180 | S | FAIL | 256 | 82.561 | 39.559 [39.381, 39.726] | 1.02584 / 1.02584 | 0.03 | 1182 |
| parallel-256-180 | M | PASS | 256 | 21.858 | 5.425 [5.394, 5.449] | 0.0231875 / 0.0231875 | 0.03 | 1184 |

## structured-v1

S = ordinary spectral CGLS; M = existing multiresolution policy with spectral CGLS at each level. `FAIL` times are unsuccessful attempts, never accepted-result times. Quality is final cold-search relative L2; the best value across the entire search is also shown. Every failed row still includes seven warm attempts at the final budget.

| Cell | Method | Status | Budget | Cold search/attempt s | Warm median [min, max] s | Final / best search L2 | Gate | GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| lamino-64-180 | S | FAIL | 256 | 3.375 | 0.743 [0.725, 0.750] | 0.629414 / 0.629414 | 0.30 | 230 |
| lamino-64-180 | M | FAIL | 256 | 5.555 | 0.207 [0.201, 0.208] | 0.382009 / 0.352084 | 0.30 | 296 |
| anisotropic-64-180 | S | FAIL | 256 | 2.576 | 0.312 [0.304, 0.316] | 0.548768 / 0.379747 | 0.15 | 228 |
| anisotropic-64-180 | M | FAIL | 256 | 5.199 | 0.109 [0.108, 0.109] | 0.365262 / 0.308441 | 0.15 | 294 |
| parallel-64-180 | S | FAIL | 256 | 2.925 | 0.523 [0.507, 0.533] | 0.219921 / 0.203865 | 0.15 | 230 |
| parallel-64-180 | M | FAIL | 256 | 5.063 | 0.164 [0.162, 0.166] | 0.173614 / 0.158704 | 0.15 | 296 |
| lamino-128-180 | S | FAIL | 256 | 13.156 | 5.694 [5.566, 5.763] | 3.79958 / 3.26241 | 0.30 | 414 |
| lamino-128-180 | M | FAIL | 256 | 7.693 | 0.900 [0.891, 0.927] | 0.407605 / 0.396734 | 0.30 | 416 |
| anisotropic-128-180 | S | FAIL | 256 | 6.818 | 2.457 [2.433, 2.461] | 1.16633 / 0.387633 | 0.15 | 302 |
| anisotropic-128-180 | M | FAIL | 256 | 6.230 | 0.370 [0.366, 0.374] | 0.346961 / 0.248778 | 0.15 | 304 |
| parallel-128-180 | S | FAIL | 256 | 10.892 | 4.516 [4.379, 4.571] | 0.307556 / 0.23908 | 0.15 | 414 |
| parallel-128-180 | M | PASS | 16 | 5.566 | 0.140 [0.139, 0.142] | 0.14873 / 0.14873 | 0.15 | 416 |
| lamino-256-180 | S | FAIL | 256 | 102.030 | 49.112 [48.924, 49.165] | 7.00535 / 3.70982 | 0.30 | 1182 |
| lamino-256-180 | M | FAIL | 256 | 27.193 | 6.966 [6.937, 6.992] | 0.668868 / 0.668868 | 0.30 | 1184 |
| anisotropic-256-180 | S | FAIL | 256 | 42.975 | 19.823 [19.755, 19.874] | 1.19134 / 1.14248 | 0.15 | 926 |
| anisotropic-256-180 | M | FAIL | 256 | 13.651 | 2.728 [2.688, 2.745] | 0.464548 / 0.464548 | 0.15 | 928 |
| parallel-256-180 | S | FAIL | 256 | 83.089 | 39.949 [39.709, 40.059] | 0.998012 / 0.998012 | 0.15 | 1182 |
| parallel-256-180 | M | FAIL | 256 | 21.842 | 5.459 [5.415, 5.468] | 0.216242 / 0.216242 | 0.15 | 1184 |

## structured-noisy-v1

S = ordinary spectral CGLS; M = existing multiresolution policy with spectral CGLS at each level. `FAIL` times are unsuccessful attempts, never accepted-result times. Quality is final cold-search relative L2; the best value across the entire search is also shown. Every failed row still includes seven warm attempts at the final budget.

| Cell | Method | Status | Budget | Cold search/attempt s | Warm median [min, max] s | Final / best search L2 | Gate | GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| lamino-64-180 | S | FAIL | 256 | 3.359 | 0.747 [0.731, 0.756] | 0.688354 / 0.688354 | 0.32 | 230 |
| lamino-64-180 | M | FAIL | 256 | 5.276 | 0.206 [0.201, 0.207] | 0.417028 / 0.367431 | 0.32 | 296 |
| anisotropic-64-180 | S | FAIL | 256 | 2.585 | 0.315 [0.304, 0.324] | 0.62912 / 0.396245 | 0.18 | 228 |
| anisotropic-64-180 | M | FAIL | 256 | 5.164 | 0.108 [0.106, 0.109] | 0.390294 / 0.314199 | 0.18 | 294 |
| parallel-64-180 | S | FAIL | 256 | 2.902 | 0.530 [0.514, 0.536] | 0.287254 / 0.24801 | 0.18 | 230 |
| parallel-64-180 | M | PASS | 16 | 4.992 | 0.032 [0.031, 0.034] | 0.169947 / 0.169947 | 0.18 | 296 |
| lamino-128-180 | S | FAIL | 256 | 13.185 | 5.770 [5.627, 5.822] | 3.91257 / 3.26249 | 0.32 | 414 |
| lamino-128-180 | M | FAIL | 256 | 7.739 | 0.898 [0.885, 0.913] | 0.487932 / 0.427326 | 0.32 | 416 |
| anisotropic-128-180 | S | FAIL | 256 | 6.950 | 2.471 [2.442, 2.487] | 2.36771 / 0.475711 | 0.18 | 302 |
| anisotropic-128-180 | M | FAIL | 256 | 6.167 | 0.377 [0.365, 0.401] | 0.54112 / 0.277227 | 0.18 | 304 |
| parallel-128-180 | S | FAIL | 256 | 10.944 | 4.542 [4.407, 4.589] | 1.02592 / 0.459476 | 0.18 | 414 |
| parallel-128-180 | M | PASS | 16 | 5.865 | 0.138 [0.137, 0.141] | 0.166736 / 0.166736 | 0.18 | 416 |
| lamino-256-180 | S | FAIL | 256 | 102.096 | 49.178 [48.831, 49.294] | 7.19277 / 3.7099 | 0.32 | 1182 |
| lamino-256-180 | M | FAIL | 256 | 27.266 | 6.970 [6.967, 7.013] | 0.758696 / 0.719917 | 0.32 | 1184 |
| anisotropic-256-180 | S | FAIL | 256 | 42.957 | 19.822 [19.791, 19.855] | 3.14668 / 1.7995 | 0.18 | 926 |
| anisotropic-256-180 | M | FAIL | 256 | 13.724 | 2.723 [2.698, 2.732] | 0.62699 / 0.52224 | 0.18 | 928 |
| parallel-256-180 | S | FAIL | 256 | 82.569 | 39.826 [39.627, 39.910] | 2.17355 / 1.50773 | 0.18 | 1182 |
| parallel-256-180 | M | FAIL | 256 | 22.002 | 5.461 [5.418, 5.495] | 0.456056 / 0.274268 | 0.18 | 1184 |

## External denominators and comparisons

For each cell, the external method below is the fastest **accepted cold** ASTRA/TIGRE workflow from the [frozen baseline](system-matrix-2026-10-04.md). Its warm time and memory are the denominators for that same cell. These are retained baseline measurements, not contemporaneous reruns. A missing accepted external workflow leaves the comparison undefined. The raw comparison JSON retains candidate time ratios only for accepted results; failed times are not speedups.

| Suite | Cell | External workflow | Cold s | Warm median [min, max] s | Final L2 | GPU MiB |
|---|---|---|---:|---:|---:|---:|
| gaussian-v1 | lamino-64-180 | astra_cgls | 1.655 | 0.674 [0.674, 0.678] | 0.0933664 | 158 |
| gaussian-v1 | anisotropic-64-180 | astra_cgls | 0.334 | 0.030 [0.030, 0.030] | 0.0204787 | 150 |
| gaussian-v1 | parallel-64-180 | astra_fbp2d | 0.291 | 0.024 [0.024, 0.024] | 0.00410799 | 140 |
| gaussian-v1 | lamino-128-180 | astra_cgls | 6.953 | 3.191 [3.189, 3.193] | 0.0944359 | 202 |
| gaussian-v1 | anisotropic-128-180 | astra_fbp3d_cupy | 0.362 | 0.010 [0.010, 0.012] | 0.00964915 | 242 |
| gaussian-v1 | parallel-128-180 | astra_fbp2d | 0.318 | 0.059 [0.058, 0.060] | 0.00104495 | 140 |
| gaussian-v1 | lamino-256-180 | astra_cgls | 39.277 | 17.920 [17.896, 17.939] | 0.0971072 | 524 |
| gaussian-v1 | anisotropic-256-180 | astra_fbp3d_cupy | 0.463 | 0.077 [0.073, 0.080] | 0.0136194 | 544 |
| gaussian-v1 | parallel-256-180 | astra_fbp2d | 0.596 | 0.298 [0.296, 0.299] | 0.000285228 | 140 |
| structured-v1 | lamino-64-180 | astra_cgls | 0.338 | 0.029 [0.028, 0.029] | 0.268893 | 158 |
| structured-v1 | anisotropic-64-180 | No accepted method | — | — | — | — |
| structured-v1 | parallel-64-180 | astra_fbp2d | 0.310 | 0.027 [0.027, 0.028] | 0.131569 | 140 |
| structured-v1 | lamino-128-180 | astra_fbp_cgls | 0.633 | 0.107 [0.105, 0.109] | 0.275349 | 374 |
| structured-v1 | anisotropic-128-180 | astra_fbp3d_cupy | 0.372 | 0.010 [0.010, 0.013] | 0.123795 | 242 |
| structured-v1 | parallel-128-180 | astra_fbp2d | 0.341 | 0.061 [0.061, 0.063] | 0.0925228 | 140 |
| structured-v1 | lamino-256-180 | astra_cgls | 3.068 | 0.988 [0.975, 1.000] | 0.254414 | 524 |
| structured-v1 | anisotropic-256-180 | astra_fbp3d_cupy | 0.465 | 0.076 [0.074, 0.077] | 0.0934158 | 544 |
| structured-v1 | parallel-256-180 | astra_fbp2d | 0.607 | 0.301 [0.299, 0.303] | 0.0724881 | 140 |
| structured-noisy-v1 | lamino-64-180 | astra_cgls | 0.328 | 0.029 [0.027, 0.030] | 0.268988 | 158 |
| structured-noisy-v1 | anisotropic-64-180 | astra_fbp_cgls | 0.355 | 0.008 [0.008, 0.008] | 0.175221 | 182 |
| structured-noisy-v1 | parallel-64-180 | astra_fbp2d | 0.296 | 0.027 [0.027, 0.027] | 0.132838 | 140 |
| structured-noisy-v1 | lamino-128-180 | astra_cgls | 0.492 | 0.099 [0.089, 0.101] | 0.319828 | 202 |
| structured-noisy-v1 | anisotropic-128-180 | astra_fbp3d_cupy | 0.363 | 0.010 [0.010, 0.012] | 0.128229 | 242 |
| structured-noisy-v1 | parallel-128-180 | astra_fbp2d | 0.326 | 0.060 [0.059, 0.061] | 0.0993521 | 140 |
| structured-noisy-v1 | lamino-256-180 | astra_cgls | 2.110 | 0.713 [0.704, 0.720] | 0.317849 | 524 |
| structured-noisy-v1 | anisotropic-256-180 | astra_fbp3d_cupy | 0.455 | 0.074 [0.072, 0.077] | 0.115666 | 544 |
| structured-noisy-v1 | parallel-256-180 | astra_fbp2d | 0.587 | 0.297 [0.296, 0.298] | 0.102397 | 140 |

| Suite | Cell | S cold / warm speedup | M cold / warm speedup | S / M memory fraction of same external |
|---|---|---:|---:|---:|
| gaussian-v1 | lamino-64-180 | undefined | undefined | 1.456 / 1.873 |
| gaussian-v1 | anisotropic-64-180 | 0.142× / 0.100× | 0.075× / 0.478× | 1.520 / 1.960 |
| gaussian-v1 | parallel-64-180 | undefined | 0.069× / 0.570× | 1.643 / 2.114 |
| gaussian-v1 | lamino-128-180 | undefined | undefined | 2.050 / 2.059 |
| gaussian-v1 | anisotropic-128-180 | undefined | 0.067× / 0.049× | 1.248 / 1.256 |
| gaussian-v1 | parallel-128-180 | undefined | 0.055× / 0.242× | 2.957 / 2.971 |
| gaussian-v1 | lamino-256-180 | undefined | undefined | 2.256 / 2.260 |
| gaussian-v1 | anisotropic-256-180 | undefined | undefined | 1.702 / 1.706 |
| gaussian-v1 | parallel-256-180 | undefined | 0.027× / 0.055× | 8.443 / 8.457 |
| structured-v1 | lamino-64-180 | undefined | undefined | 1.456 / 1.873 |
| structured-v1 | anisotropic-64-180 | undefined | undefined | undefined / undefined |
| structured-v1 | parallel-64-180 | undefined | undefined | 1.643 / 2.114 |
| structured-v1 | lamino-128-180 | undefined | undefined | 1.107 / 1.112 |
| structured-v1 | anisotropic-128-180 | undefined | undefined | 1.248 / 1.256 |
| structured-v1 | parallel-128-180 | undefined | 0.061× / 0.437× | 2.957 / 2.971 |
| structured-v1 | lamino-256-180 | undefined | undefined | 2.256 / 2.260 |
| structured-v1 | anisotropic-256-180 | undefined | undefined | 1.702 / 1.706 |
| structured-v1 | parallel-256-180 | undefined | undefined | 8.443 / 8.457 |
| structured-noisy-v1 | lamino-64-180 | undefined | undefined | 1.456 / 1.873 |
| structured-noisy-v1 | anisotropic-64-180 | undefined | undefined | 1.253 / 1.615 |
| structured-noisy-v1 | parallel-64-180 | undefined | 0.059× / 0.834× | 1.643 / 2.114 |
| structured-noisy-v1 | lamino-128-180 | undefined | undefined | 2.050 / 2.059 |
| structured-noisy-v1 | anisotropic-128-180 | undefined | undefined | 1.248 / 1.256 |
| structured-noisy-v1 | parallel-128-180 | undefined | 0.056× / 0.432× | 2.957 / 2.971 |
| structured-noisy-v1 | lamino-256-180 | undefined | undefined | 2.256 / 2.260 |
| structured-noisy-v1 | anisotropic-256-180 | undefined | undefined | 1.702 / 1.706 |
| structured-noisy-v1 | parallel-256-180 | undefined | undefined | 8.443 / 8.457 |

## Decision and provenance

The positive preconditioner passes independent dense-system and matched-operator tests but does not preserve acceptable finite-budget reconstruction quality across this matrix. Improved tiny-system condition numbers did not predict workflow success. A possible explanation is changed implicit regularization and null-space selection in ill-conditioned/rank-deficient problems; that mechanism has not been established by this experiment. No spectral floor, probe-count or per-case tuning follows this rejection.

In the separate fixed-state alignment diagnostics, even high-budget proposals leave noisy anisotropic rotation above the gate. No end-to-end spectral alignment comparison was promoted after the reconstruction rejection. Those diagnostics do not establish a new successful-recovery baseline; the 20× and 99% goals remain open.

Frozen runtime/driver SHA-256: `308aafea5a2fe11705f7c6a64cc311b02bd5c21f1e79f667309b793b82fd439d`. The entire sweep used that immutable source and it still hashes identically. [Launch metadata](../bench/reference/system-matrix-spectral-v1-source.json) and the [source archive](../bench/reference/system-matrix-spectral-v1-source.tar.gz) preserve code and tests. The candidate passed 614 CPU tests, 28 selected CUDA numerical tests, 36 benchmark tests, lint, type and import checks before the sweep. Fixture-generation source was verified byte-identical to the audited original baseline.

All raw results: [Gaussian](../bench/reference/system-matrix-spectral-v1-gaussian-v1.json.gz), [structured](../bench/reference/system-matrix-spectral-v1-structured-v1.json.gz), [structured noisy](../bench/reference/system-matrix-spectral-v1-structured-noisy-v1.json.gz), and [derived comparisons](../bench/reference/system-matrix-spectral-v1-summary.json). The experimental runtime and driver additions were removed by restoring their pre-experiment content, preserving earlier work; the withdrawal diff and hashes accompany the source metadata.
