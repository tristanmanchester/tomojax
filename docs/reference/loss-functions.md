# Loss functions

The alignment loss measures how well reprojected data from the current
volume and pose estimates matches the measured projections. TomoJAX
provides a range of losses with different trade-offs between
robustness, sharpness, and optimizer compatibility.

## Available losses

The table below lists the most commonly used losses for alignment. All
names can be passed to `--loss` or used in `--loss-schedule`.

| Name | Description | Best for |
|------|-------------|----------|
| `l2` | Standard L2 (least squares) | Clean data with GN optimizer |
| `l2_otsu` | L2 with Otsu-thresholded mask | Fine-level stabilization |
| `charbonnier` | Smooth L1 approximation | Robust 5-DOF with L-BFGS |
| `ssim` | Structural similarity | Robust image-similarity alignment |
| `phasecorr` | Phase correlation | Coarse translation estimation |
| `edge_l2` | L2 on image edges | Edge-sensitive alignment |
| `pwls` | Penalized weighted least squares | Weighted data fidelity |

Additional losses available for specialized workflows:

| Name | Description |
|------|-------------|
| `zncc` | Zero-normalized cross-correlation |
| `huber` | Huber loss (smooth L1 transition) |
| `cauchy` | Cauchy robust loss |
| `welsch` | Welsch robust loss |
| `student_t` | Student-t robust loss |
| `barron` | General Barron robust loss |
| `mi` / `nmi` | Mutual information / normalized MI |
| `renyi_mi` | Renyi mutual information |
| `tversky` | Tversky index loss |
| `swd` | Sliced Wasserstein distance |
| `grad_l1` / `ngf` | Gradient-based losses |
| `poisson` | Poisson likelihood |
| `mind` | MIND descriptor |

> [!NOTE]
> Loss names have aliases. For example, `charb` maps to
> `charbonnier`, and `ncc` maps to `zncc`. The CLI normalizes
> these automatically.

## Loss schedules

You can assign different losses to different pyramid levels using
`--loss-schedule`:

```bash
--loss-schedule 4:phasecorr,2:ssim,1:l2_otsu
```

The numeric keys refer to the `--levels` values (pyramid downsample
factors). Levels omitted from the schedule fall back to the value of
`--loss` (default: `l2`).

In a TOML config file, you can write the schedule as either a string
or a mapping:

```toml
# String form
loss_schedule = "4:phasecorr,2:ssim,1:l2_otsu"

# Mapping form
loss_schedule = { "4" = "phasecorr", "2" = "ssim", "1" = "l2_otsu" }
```

## Guidance

These recommendations come from tested smoke-test runs on small
datasets across multiple seeds.

### Translation-only coarse-to-fine (2-DOF)

For `--optimise-dofs dx,dz`, start with:

```bash
--optimise-dofs dx,dz \
--loss-schedule 4:phasecorr,2:ssim,1:l2_otsu
```

This schedule reduced translation RMSE most consistently in 40^3
smoke tests over three seeds. The next-best tested schedule was
`4:ssim,2:l2,1:l2_otsu`.

### Full 5-DOF alignment

Prefer a conservative image-similarity loss first:

```bash
--loss ssim
```

or `--loss charbonnier`. In the same smoke tests, coarse L2-style
losses (`l2`, `l2_otsu`, `edge_l2`) with GN improved translations
but sometimes over-rotated poses.

### General tips

- Treat `phasecorr` as a coarse translation helper, not a full-level
  loss. It works well at the coarsest pyramid level but performs
  poorly when used everywhere.
- Use `l2_otsu` as a fine-level stabilizer or fallback. It's
  conservative and can reject unsafe GN steps, but it may not make
  progress by itself on coarse levels.
- L-BFGS is useful for losses where GN isn't available (SSIM,
  Charbonnier, mutual information). GN is generally faster when
  compatible with the loss.

## Loss parameters

Some losses accept hyperparameters via `--loss-param`:

```bash
--loss charbonnier --loss-param eps=0.001,delta=1.0
```

Common parameters:

| Loss | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `charbonnier` | `eps` | 0.001 | Smoothing constant |
| `charbonnier` | `delta` | 1.0 | Scale parameter |
| `l2_otsu` | `temp` | 0.5 | Softmax temperature |
| `ssim` | `K1`, `K2` | 0.01, 0.03 | Stability constants |
| `ssim` | `window` | 7 | Window size |
| `tversky` | `alpha`, `beta` | 0.7, 0.3 | Asymmetry parameters |

## Next steps

- [align CLI reference](../cli/align.md) — using losses on the
  command line
- [Alignment concepts](../concepts/alignment.md) — how losses fit
  into the optimization
- [loss-bench CLI](../cli/loss-bench.md) — benchmarking losses
  systematically
