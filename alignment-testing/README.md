# Alignment Testing

Implementation of joint iterative reconstruction and 3D rigid alignment following Pande et al. (2022). This directory contains the complete multi-resolution alternating optimization algorithm for tomographic alignment.

## Overview

The alignment algorithm alternates between:
1. **FISTA-TV reconstruction** (fix alignment, update volume)
2. **Per-view alignment optimization** (fix volume, update rigid-body parameters)

Using multi-resolution strategy: 4x → 2x → 1x binning for robust convergence.

## Files

### Core Implementation

**`run_alignment.py`** - Main alternating optimization script
- Implements multi-resolution alternating minimization algorithm
- Coordinates between FISTA-TV reconstruction and per-view alignment optimization
- Manages resolution pyramid (4x → 2x → 1x binning) with parameter transfer
- Handles Lipschitz constant caching across outer iterations for efficiency

**`optimization_steps.py`** - Core optimization algorithms
- `fista_tv_reconstruction()`: FISTA with TV regularization using scan-based projector compilation
- `optimize_alignment_params()`: Gradient descent on 5-DOF rigid parameters per view
- `build_aligned_loss_and_grad_scan()`: Compiled JAX scan for efficient multi-view loss computation
- `estimate_L_power_scan()`: Power method for Lipschitz constant estimation using compiled A^T A operator

**`alignment_utils.py`** - Multi-resolution and geometric utilities
- `create_resolution_pyramid()`: Generates binned projections with consistent physical coordinate scaling
- `transfer_alignment_params()`: Transfers parameters between resolution levels (world coordinates preserved)
- `bin_projections()` / `bin_volume()`: Spatial binning with proper averaging
- `compute_alignment_metrics()`: RMSE computation for rotations (degrees) and translations (pixels)

### Testing and Debug Scripts

**`test_synthetic_alignment.py`** - Validation framework
- Runs multiple test configurations (fast/standard) with ground truth comparison
- Generates alignment parameter comparison plots and convergence history visualization
- Computes quantitative error metrics and timing benchmarks

**`run_quick_test.py`** - Fast debugging workflow
- Creates 32³ volume with 64 projections (8x spatial + angular downsampling)
- Minimal iterations for rapid algorithm validation (~1-2 min runtime)
- Tests core functionality without full computational cost

**`debug_geometry.py`** - Geometric parameter debugging
- Compares grid and detector parameters between test configurations
- Validates coordinate system consistency across resolution levels
- Helps debug scaling issues in multi-resolution optimization

**`setup_jax_cache.sh`** - Performance optimization
- Configures JAX compilation cache to avoid recompilation across runs
- Reduces startup time for repeated algorithm execution

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

## Mathematical Framework

### Objective Function
Joint optimization of reconstruction x and alignment parameters θ:
```
min_{x,θ} Σᵢ ½||Aᵢ(θᵢ)x - yᵢ||² + λ_TV · TV(x)
```
where Aᵢ(θᵢ) is the projection operator for view i with 5-DOF rigid parameters θᵢ = [α, β, φ, Δx, Δz].

### FISTA-TV Reconstruction
Proximal gradient method with isotropic total variation regularization:
- **Data fidelity**: f(x) = ½Σᵢ||Aᵢ(θᵢ)x - yᵢ||² with gradient ∇f(x) = Σᵢ Aᵢᵀ(Aᵢx - yᵢ)
- **TV regularization**: Isotropic total variation TV(x) = Σ||∇x|| with Chambolle-Pock proximal operator
- **FISTA acceleration**: Momentum step zₖ = xₖ + βₖ(xₖ - xₖ₋₁) with adaptive βₖ
- **Lipschitz estimation**: Power method on A^T A operator, cached across outer iterations for efficiency
- **JAX scan implementation**: Compiled scan over views for memory-efficient gradient computation

### Alignment Parameter Optimization  
Gradient descent on 5-DOF rigid-body parameters per view:
- **Parameter space**: θᵢ = [αᵢ, βᵢ, φᵢ, Δxᵢ, Δzᵢ] per projection i
- **Loss function**: Lᵢ(θᵢ) = ½||Aᵢ(θᵢ)x - yᵢ||² (reconstruction x fixed)
- **Gradient computation**: ∇_θᵢ Lᵢ via JAX autodiff through projector
- **Learning rates**: Separate rates for rotations (0.001 rad) and translations (0.1 world units)
- **Gradient clipping**: L2 norm clipping at 1.0 to prevent parameter explosion

## Algorithm Implementation

### Alternating Optimization Strategy
1. **Reconstruction step**: Fix alignment parameters θ, optimize volume x using FISTA-TV
2. **Alignment step**: Fix reconstruction x, optimize each θᵢ independently via gradient descent
3. **Multi-resolution**: Coarse-to-fine optimization avoiding local minima

### Multi-Resolution Pyramid
- **Level 0 (4x binning)**: 64³ → 64² projections, robust initialization
- **Level 1 (2x binning)**: 128³ → 128² projections, parameter refinement  
- **Level 2 (1x binning)**: 256³ → 256² projections, final precision
- **Parameter transfer**: World coordinates preserved, detector scaling automatic

### Computational Efficiency
- **JAX scan compilation**: Single compiled function processes all views sequentially
- **Lipschitz caching**: Recompute L every 3 outer iterations, reuse cached value otherwise
- **Memory optimization**: Streaming integration via scan avoids materializing ray tensors
- **Gradient checkpointing**: Optional checkpointing during backpropagation for memory reduction

### Convergence and Stability
- **Early stopping**: Relative objective change < 1e-5 over 3 iterations
- **Parameter initialization**: Small random perturbations to avoid zero gradients
- **Learning rate adaptation**: Separate rates for rotation (0.001 rad ≈ 0.057°) and translation (0.1 world units)
- **Robust metrics**: RMSE computation in physically meaningful units (degrees, pixels)

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