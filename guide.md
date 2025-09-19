This guide is now covered by the new v2 CLIs and tutorial. Use the commands below or open `docs/tutorial_end_to_end.md` for a copy‑paste workflow.

Quick reference (inside `pixi shell`):

```
# Simulate
pixi run simulate \
  --out data/sim_aligned.nxs \
  --nx 256 --ny 256 --nz 256 --nu 256 --nv 256 --n-views 200 \
  --phantom random_shapes --n-cubes 40 --n-spheres 40 --min-size 4 --max-size 64 \
  --min-value 0.01 --max-value 0.1 --seed 42

# Misalign (and optionally add Poisson noise)
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 --seed 0
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned_poisson5k.nxs \
  --rot-deg 1.0 --trans-px 10 --poisson 100 --seed 0

# Reconstructions
pixi run recon --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp --views-per-batch auto --gather-dtype bf16 \
  --checkpoint-projector --out out/fbp_misaligned.nxs

# Alignment + reconstruction (multires)
pixi run align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch auto --gather-dtype bf16 --checkpoint-projector --projector-unroll 4 \
  --log-summary --out out/align_misaligned.nxs
```

Further details:
- Tutorial: `docs/tutorial_end_to_end.md`
- Data schema: `docs/schema_nxtomo.md`
- FAQ/Troubleshooting: `docs/faq_troubleshooting.md`

Tip: expose `recon_rel_tol` / `recon_patience` via `AlignConfig`
(or pass them directly to `fista_tv`) to short-circuit reconstructions once the objective plateaus.
