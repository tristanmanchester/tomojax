To generate noisy misaligned projections:

```
pixi run python examples/run_parallel_projector_misaligned.py --nx 256 --ny 256 --nz 256 --n-cubes 40 --n-spheres 40 --max-size 64 --min-size 4 --max-value 0.01 --min-value 0.001 --n-proj 512 --step-size 1 --checkpoint --max-trans-pixels 10 --max-rot-degrees 1 --add-projection-noise --incident-photons 100 --output-dir noisy-misaligned
```

This will create 3D tiffs with:
- a phantom
- clean projections
- misaligned projections
- misaligned noisy projections


For naive reconstructions of the misaligned data:
```
pixi run python examples/run_parallel_reconstruction.py --input-dir noisy-misaligned --output-dir misaligned_recon --projections-file projections_misaligned.tiff
```

For naive reconstructions of the noisy misaligned data:
```
pixi run python examples/run_parallel_reconstruction.py --input-dir noisy-misaligned --output-dir noisy_misaligned_recon --projections-file projections_noisy.tiff
```

For iterative alignment and reconstruction of the misaligned data:
```
pixi run python alignment-testing/run_alignment.py --optimizer lbfgs --input-dir noisy-misaligned --projections-file projections_misaligned.tiff --output-dir misaligned-alignment --optimize-phi --bin-factors 3 2 1 --recon-iters 10 20 30 --align-iters 10 15 20 --outer-iters 15 --lambda-tv 0.001 --optimizer adabelief
```

For iterative alignment and reconstruction of the noisy misaligned data:
```
pixi run python alignment-testing/run_alignment.py --optimizer lbfgs --input-dir noisy-misaligned --projections-file projections_noisy.tiff --output-dir noisy-alignment --optimize-phi --bin-factors 8 4 2 1 --recon-iters 10 10 20 30 --align-iters 5 5 10 15 --outer-iters 20 --lambda-tv 0.01
```
