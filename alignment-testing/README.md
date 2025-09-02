# Alignment Testing

Implementation of joint iterative reconstruction and 3D rigid alignment following Pande et al. (2022). This directory contains the complete multi-resolution alternating optimization algorithm for tomographic alignment.

## Overview

The alignment algorithm alternates between:
1. **FISTA-TV reconstruction** (fix alignment, update volume)
2. **Per-view alignment optimization** (fix volume, update rigid-body parameters)

Using multi-resolution strategy: 4x → 2x → 1x binning for robust convergence.

## Files

### Core Implementation
- `run_alignment.py` - Main alignment script with multi-resolution optimization
- `optimization_steps.py` - FISTA-TV reconstruction and alignment optimization functions
- `alignment_utils.py` - Multi-resolution utilities (binning, upsampling, metrics)

### Testing Scripts
- `test_synthetic_alignment.py` - Comprehensive validation with ground truth comparison
- `run_quick_test.py` - Fast debugging test (32³ volume, 64 projections)

## Usage

### Quick Debug Test
First, run the quick test to verify the algorithm works:

```bash
cd alignment-testing
pixi run python run_quick_test.py
```

This creates a 32³ volume test case that runs in ~1-2 minutes. Should see:
```
✓ Quick test passed - algorithm appears to be working
```

### Full Alignment Test
Run alignment on the full synthetic dataset:

```bash
# Standard test (recommended)
pixi run python run_alignment.py --input-dir ../misaligned_test --outer-iters 15 --lambda-tv 0.005

# Fast test (fewer iterations)  
pixi run python run_alignment.py --bin-factors 4 2 --outer-iters 8 --recon-iters 8 12 --align-iters 5 8

# Full test (slow but thorough)
pixi run python run_alignment.py --outer-iters 25 --recon-iters 15 25 35 --align-iters 8 12 18
```

### Validation with Ground Truth
Compare estimated parameters with true misalignments:

```bash
pixi run python test_synthetic_alignment.py
```

Creates comparison plots and detailed error metrics.

## Algorithm Parameters

### Key Settings
- `--bin-factors`: Multi-resolution levels (default: `[4, 2, 1]`)
- `--outer-iters`: Outer alternating iterations per level (default: 15)
- `--lambda-tv`: TV regularization weight (default: 0.005)
- `--recon-iters`: FISTA-TV iterations per level (default: `[10, 20, 30]`)
- `--align-iters`: Alignment iterations per level (default: `[5, 10, 15]`)

### Recommended Settings
- **Fast**: `--bin-factors 4 2 --outer-iters 8`
- **Standard**: Default parameters (good balance)
- **High quality**: `--outer-iters 25 --recon-iters 15 25 35`

## Expected Performance

### Timing (MacBook Pro CPU)
- Quick test: ~1-2 minutes
- Fast settings: ~15-30 minutes  
- Standard settings: ~45-90 minutes
- High quality: ~2-4 hours

### Accuracy Goals
Good alignment should achieve:
- Rotation RMSE: < 0.1° (true misalignment: ±1°)
- Translation RMSE: < 0.5 pixels (true misalignment: ±2 pixels)

## Output Files

### Alignment Results
- `final_reconstruction.tiff` - Final reconstructed volume
- `final_alignment_params.npy` - Estimated alignment parameters (n_proj, 5)
- `results.json` - Complete results with convergence history
- `resolution_*/` - Intermediate results per resolution level

### Validation Results  
- `alignment_comparison_*.png` - Parameter comparison plots
- `convergence_history_*.png` - Objective function convergence
- `metrics_*.json` - Quantitative error metrics

## Algorithm Details

### Multi-Resolution Strategy
1. **Level 1 (4x binning)**: 64³ volume, fast convergence, avoid local minima
2. **Level 2 (2x binning)**: 128³ volume, refine alignment
3. **Level 3 (1x binning)**: 256³ volume, final precision

### Parameter Space
5 DOF per projection: (α, β, φ, Δx, Δz)
- α, β: In-plane rotations (radians)
- φ: Projection angle (fixed)  
- Δx, Δz: Detector translations (world units)

### Optimization
- **Reconstruction**: FISTA with isotropic TV regularization
- **Alignment**: Gradient descent using JAX autodiff
- **Convergence**: Objective function relative change < 1e-5

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're in `alignment-testing/` directory
2. **Memory issues**: Reduce `--outer-iters` or use fewer angles
3. **Slow convergence**: Increase `--lambda-tv` to 0.01-0.02
4. **Poor alignment**: Check if misalignment is within expected range

### Debug Steps
1. Run `run_quick_test.py` first
2. Check that input data exists in `../misaligned_test/`
3. Monitor objective function convergence in output logs
4. Verify parameter scales are reasonable (degrees vs radians, etc.)

## Implementation Notes

- Uses JAX for automatic differentiation and GPU acceleration
- Memory-efficient: processes each resolution level sequentially
- Modular design: easy to modify parameters or add new optimization methods
- Follows paper algorithm closely but with practical optimizations for JAX