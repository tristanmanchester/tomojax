# Reconstruction comparison after the memory and kernel changes, 2026-10-05

TomoJAX source: commit `6fa1585` (sinograms kept in the library's layout by the
Joseph CUDA kernels, 8-by-16 forward ray tiles, in-place gather accumulation,
fused batch-wise least-squares gradients, dual projected-gradient TV prox). Same
fixtures, physical geometry, budgets (1, 2, 4, …, 256), full-volume relative L2
gates, RTX 4070 Laptop GPU and worker protocol as the
[morning matrix](system-matrix-2026-10-05.md); ASTRA was re-measured in the
same session. TIGRE was not re-run: it was the fastest accepted external method
in no cell of the previous matrices. Cold time is a fresh process through the
first accepted budget; warm is the median of three repeats of that budget.
Raw records: `bench/reference/system-matrix-v4-*.json.gz`.

| | Morning (`ab86e3c`) | Now (`6fa1585`) |
|---|---:|---:|
| Cold speedup, geometric mean over 26 paired cells | 0.95 | **1.00** |
| Worst paired cold speedup | 0.40 | 0.44 |
| Warm speedup, geometric mean | 2.62 | **2.78** |
| Worst paired warm speedup | 0.75 | 0.59 |

Speedup is the ASTRA time divided by TomoJAX's, each side's fastest accepted
workflow chosen by cold time. The worst warm cell, structured laminography at
128³, pairs TomoJAX's cold-fastest method (plain Joseph CGLS, 178 ms warm)
with ASTRA's FBP-initialised CGLS (106 ms); TomoJAX's coarse-to-fine CGLS in
the same cell is accepted at 61 ms warm but starts 0.3 s slower from a fresh
process. Largest gains are in laminography, from the faster forward projection:
Gaussian laminography 256³ fell from 6212 to 5572 ms warm and structured 256³
from 825 to 647 ms.

The remaining cold gaps are small laminography scans (0.44–0.59), where JAX's
import and device setup (about 0.4 s) and Python tracing and lowering of the
solver (about 0.2 s) outweigh 30–180 ms solves. Persisting traced programs
would remove the tracing, but only matters for problems this small, so it was
not adopted.

| Suite | Cell | TomoJAX fastest accepted (cold / warm ms) | Previous TomoJAX (cold / warm ms) | ASTRA fastest accepted (cold / warm ms) | Cold speedup | Warm speedup |
|---|---|---|---|---|---:|---:|
| gaussian-v1 | parallel-64 | fourier_cupy: 368 / 3.0 | 384 / 3.0 | fbp2d: 334 / 24.6 | 0.91 | 8.18 |
| gaussian-v1 | anisotropic-64 | fourier_cupy: 372 / 1.5 | 375 / 1.5 | cgls: 383 / 30.1 | 1.03 | 20.41 |
| gaussian-v1 | lamino-64 | multires_joseph_cgls_pallas: 1777 / 239.0 | 2058 / 238.8 | cgls: 1728 / 676.4 | 0.97 | 2.83 |
| gaussian-v1 | parallel-128 | fourier_cupy: 417 / 11.2 | 412 / 11.6 | fbp2d: 391 / 59.8 | 0.94 | 5.32 |
| gaussian-v1 | anisotropic-128 | fourier_cupy: 394 / 8.1 | 398 / 7.7 | fbp3d_cupy: 454 / 10.4 | 1.15 | 1.29 |
| gaussian-v1 | lamino-128 | multires_joseph_cgls_pallas: 2942 / 763.5 | 3506 / 809.2 | cgls: 7066 / 3200.5 | 2.40 | 4.19 |
| gaussian-v1 | parallel-256 | fourier_cupy: 593 / 74.6 | 573 / 71.4 | fbp2d: 697 / 297.5 | 1.17 | 3.99 |
| gaussian-v1 | anisotropic-256 | fourier_cupy: 507 / 39.0 | 508 / 47.5 | fbp3d_cupy: 539 / 77.5 | 1.06 | 1.99 |
| gaussian-v1 | lamino-256 | multires_joseph_cgls_pallas: 14077 / 5572.4 | 15958 / 6212.3 | cgls: 39612 / 18048.9 | 2.81 | 3.24 |
| structured-v1 | parallel-64 | fourier_cupy: 422 / 3.6 | 423 / 3.4 | fbp2d: 367 / 28.4 | 0.87 | 7.85 |
| structured-v1 | anisotropic-64-180 | none accepted | none accepted | — | — | — |
| structured-v1 | lamino-64 | joseph_cgls_pallas: 912 / 28.7 | 1031 / 29.3 | cgls: 405 / 29.2 | 0.44 | 1.02 |
| structured-v1 | parallel-128 | fourier_cupy: 513 / 12.0 | 491 / 11.9 | fbp2d: 439 / 61.8 | 0.86 | 5.14 |
| structured-v1 | anisotropic-128 | fourier_cupy: 416 / 8.0 | 411 / 8.1 | fbp3d_cupy: 474 / 10.6 | 1.14 | 1.33 |
| structured-v1 | lamino-128 | joseph_cgls_pallas: 1197 / 178.4 | 1361 / 142.0 | fbp_cgls: 711 / 105.8 | 0.59 | 0.59 |
| structured-v1 | parallel-256 | fourier_cupy: 614 / 69.5 | 627 / 70.9 | fbp2d: 678 / 308.0 | 1.10 | 4.43 |
| structured-v1 | anisotropic-256 | fourier_cupy: 533 / 39.6 | 511 / 39.2 | fbp3d_cupy: 555 / 76.5 | 1.04 | 1.93 |
| structured-v1 | lamino-256 | fbp_joseph_cgls_pallas: 2108 / 647.4 | 2589 / 824.6 | cgls: 3225 / 982.5 | 1.53 | 1.52 |
| structured-noisy-v1 | parallel-64 | fourier_cupy: 404 / 3.3 | 383 / 3.3 | fbp2d: 394 / 26.2 | 0.98 | 7.99 |
| structured-noisy-v1 | anisotropic-64 | fourier_cupy: 391 / 1.6 | 390 / 1.6 | fbp_cgls: 448 / 8.5 | 1.15 | 5.29 |
| structured-noisy-v1 | lamino-64 | joseph_cgls_pallas: 861 / 29.2 | 969 / 28.9 | cgls: 407 / 29.3 | 0.47 | 1.00 |
| structured-noisy-v1 | parallel-128 | fourier_cupy: 504 / 11.4 | 454 / 11.4 | fbp2d: 396 / 62.2 | 0.79 | 5.46 |
| structured-noisy-v1 | anisotropic-128 | fourier_cupy: 406 / 8.1 | 429 / 8.0 | fbp3d_cupy: 450 / 10.0 | 1.11 | 1.24 |
| structured-noisy-v1 | lamino-128 | joseph_cgls_pallas: 1043 / 104.2 | 1131 / 109.9 | cgls: 588 / 98.6 | 0.56 | 0.95 |
| structured-noisy-v1 | parallel-256 | fourier_cupy: 593 / 70.4 | 571 / 75.7 | fbp2d: 678 / 298.7 | 1.14 | 4.24 |
| structured-noisy-v1 | anisotropic-256 | fourier_cupy: 545 / 41.7 | 551 / 39.2 | fbp3d_cupy: 532 / 77.9 | 0.98 | 1.87 |
| structured-noisy-v1 | lamino-256 | fbp_joseph_cgls_pallas: 2099 / 645.0 | 2528 / 821.3 | cgls: 2277 / 736.3 | 1.08 | 1.14 |
