# Reconstruction comparison after exact FBP and startup changes, 2026-10-05

TomoJAX source: commit `ab86e3c` (exact FBP weights for every circular scan,
JAX-free Fourier imports, persistent compile cache, fused CGLS input checks).
Same fixtures, physical geometry, budgets (1, 2, 4, …, 256) and full-volume
relative L2 gates as the [previous matrix](system-matrix-2026-10-04.md); same
RTX 4070 Laptop GPU and worker protocol. Each cell's cold time is a fresh
process through the first accepted budget, including imports, unsuccessful
lower budgets and quality checks. Warm is the median of **three** repeats of
the accepted budget, not the seven used previously.

New compared to the previous matrix: TomoJAX FBP and FBP-initialised CGLS now
run on laminography, using the exact tilted-scan weights. TIGRE FBP and CGLS
were measured on the parallel cells, where its adapter applies; neither is
the fastest accepted external method in any cell.

| | Before (2026-10-04) | After |
|---|---:|---:|
| Cold speedup, geometric mean over 26 paired cells | 0.57 | **0.95** |
| Worst paired cold speedup | 0.21 | **0.40** |
| Warm speedup, geometric mean | 2.62 | 2.62 |
| Worst paired warm speedup | 0.62 | **0.75** |

Speedup is the external method's time divided by TomoJAX's, each side's
fastest accepted workflow by cold time. The sharp anisotropic 64 cell still has
no accepted method on either side, so the 27-cell aggregate is undefined.

Laminography FBP alone misses every laminography gate: these gates need part of
the missing cone back, which an iterative solver recovers only through the
bounded cubic grid. FBP-initialised CGLS halves the iterations in structured
128-cubed laminography (4 against 8).

The remaining gaps are small laminography cells. There JAX's fixed start-up
(about 300 ms import, 200 ms device setup) and about 200 ms of tracing outweigh
solves of 30–140 ms, and the matched Joseph backprojection is 2–3 times slower
per iteration than ASTRA's texture-interpolated backprojection.

| Suite | Cell | TomoJAX fastest accepted (cold / warm ms) | ASTRA or TIGRE fastest accepted (cold / warm ms) | Cold speedup | Warm speedup |
|---|---|---|---|---:|---:|
| gaussian-v1 | anisotropic-64 | fourier_cupy: 375 / 1.5 | astra_cgls: 382 / 31.3 | 1.02 | 20.51 |
| gaussian-v1 | anisotropic-128 | fourier_cupy: 398 / 7.7 | astra_fbp3d_cupy: 437 / 10.7 | 1.10 | 1.38 |
| gaussian-v1 | anisotropic-256 | fourier_cupy: 508 / 47.5 | astra_fbp3d_cupy: 548 / 77.8 | 1.08 | 1.64 |
| gaussian-v1 | lamino-64 | multires_joseph_cgls_pallas: 2058 / 238.8 | astra_cgls: 1755 / 675.4 | 0.85 | 2.83 |
| gaussian-v1 | lamino-128 | multires_joseph_cgls_pallas: 3506 / 809.2 | astra_cgls: 7048 / 3200.9 | 2.01 | 3.96 |
| gaussian-v1 | lamino-256 | multires_joseph_cgls_pallas: 15958 / 6212.3 | astra_cgls: 39803 / 18296.7 | 2.49 | 2.95 |
| gaussian-v1 | parallel-64 | fourier_cupy: 384 / 3.0 | astra_fbp2d: 336 / 24.8 | 0.88 | 8.34 |
| gaussian-v1 | parallel-128 | fourier_cupy: 412 / 11.6 | astra_fbp2d: 400 / 59.6 | 0.97 | 5.14 |
| gaussian-v1 | parallel-256 | fourier_cupy: 573 / 71.4 | astra_fbp2d: 704 / 303.3 | 1.23 | 4.25 |
| structured-v1 | anisotropic-64 | none accepted | none accepted | — | — |
| structured-v1 | anisotropic-128 | fourier_cupy: 411 / 8.1 | astra_fbp3d_cupy: 460 / 10.6 | 1.12 | 1.31 |
| structured-v1 | anisotropic-256 | fourier_cupy: 511 / 39.2 | astra_fbp3d_cupy: 538 / 77.5 | 1.05 | 1.98 |
| structured-v1 | lamino-64 | joseph_cgls_pallas: 1031 / 29.3 | astra_cgls: 412 / 28.0 | 0.40 | 0.96 |
| structured-v1 | lamino-128 | fbp_joseph_cgls_pallas: 1361 / 142.0 | astra_fbp_cgls: 711 / 106.6 | 0.52 | 0.75 |
| structured-v1 | lamino-256 | fbp_joseph_cgls_pallas: 2589 / 824.6 | astra_cgls: 3221 / 995.1 | 1.24 | 1.21 |
| structured-v1 | parallel-64 | fourier_cupy: 423 / 3.4 | astra_fbp2d: 363 / 28.0 | 0.86 | 8.17 |
| structured-v1 | parallel-128 | fourier_cupy: 491 / 11.9 | astra_fbp2d: 409 / 62.1 | 0.83 | 5.20 |
| structured-v1 | parallel-256 | fourier_cupy: 627 / 70.9 | astra_fbp2d: 666 / 302.8 | 1.06 | 4.27 |
| structured-noisy-v1 | anisotropic-64 | fourier_cupy: 390 / 1.6 | astra_fbp3d_cupy: 425 / 3.3 | 1.09 | 2.07 |
| structured-noisy-v1 | anisotropic-128 | fourier_cupy: 429 / 8.0 | astra_fbp3d_cupy: 455 / 10.0 | 1.06 | 1.25 |
| structured-noisy-v1 | anisotropic-256 | fourier_cupy: 551 / 39.2 | astra_fbp3d_cupy: 547 / 77.3 | 0.99 | 1.97 |
| structured-noisy-v1 | lamino-64 | joseph_cgls_pallas: 969 / 28.9 | astra_cgls: 418 / 29.0 | 0.43 | 1.00 |
| structured-noisy-v1 | lamino-128 | joseph_cgls_pallas: 1131 / 109.9 | astra_cgls: 573 / 97.2 | 0.51 | 0.88 |
| structured-noisy-v1 | lamino-256 | fbp_joseph_cgls_pallas: 2528 / 821.3 | astra_cgls: 2215 / 725.9 | 0.88 | 0.88 |
| structured-noisy-v1 | parallel-64 | fourier_cupy: 383 / 3.3 | astra_fbp2d: 353 / 26.3 | 0.92 | 8.09 |
| structured-noisy-v1 | parallel-128 | fourier_cupy: 454 / 11.4 | astra_fbp2d: 394 / 60.7 | 0.87 | 5.33 |
| structured-noisy-v1 | parallel-256 | fourier_cupy: 571 / 75.7 | astra_fbp2d: 663 / 303.9 | 1.16 | 4.01 |

Paired cells: 26. Cold speedup geometric mean 0.95 (range 0.40–2.49); warm 2.62 (range 0.75–20.51).

Raw records: [ASTRA and TomoJAX](../../bench/reference/system-matrix-v3-gaussian-v1.json.gz)
([structured](../../bench/reference/system-matrix-v3-structured-v1.json.gz),
[structured noisy](../../bench/reference/system-matrix-v3-structured-noisy-v1.json.gz));
TIGRE ([smooth](../../bench/reference/system-matrix-v3-tigre-gaussian-v1.json.gz),
[structured](../../bench/reference/system-matrix-v3-tigre-structured-v1.json.gz),
[structured noisy](../../bench/reference/system-matrix-v3-tigre-structured-noisy-v1.json.gz)).
Each record keeps every method's search, repeats, failures and sampled peak
process GPU memory.
