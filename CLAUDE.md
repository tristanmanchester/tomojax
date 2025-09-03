# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomoJAX is a differentiable parallel-beam CT projector implemented in JAX. It provides memory-efficient, physically accurate CT projection and reconstruction with exact gradients for rigid-body alignment optimization on both CPU and GPU systems.

## Core Architecture

### Main Components

1. **Core Projector** (`projector_parallel_jax.py`):
   - Differentiable parallel-beam forward projector with 5-DOF rigid-body parameterization
   - Uses trilinear interpolation and streaming integration via `jax.lax.scan`
   - Memory-efficient design avoids materializing ray-step tensors
   - Rigid-body transforms: T x = R_y(β) R_x(α) R_z(φ) x + t, where t = [Δx, 0, Δz]

2. **Example Scripts** (`examples/`):
   - `run_parallel_projector.py`: Demo projector with 3D phantom generation and forward projection
   - `run_parallel_reconstruction.py`: FBP reconstruction using slice-by-slice Ram-Lak filtering

3. **Test Data Directories**:
   - `parallel_proj_test/`: Generated phantom and projection data
   - `parallel_recon_test/`: Reconstruction results

### Key Design Patterns

- **Streaming Integration**: Uses `jax.lax.scan` to iterate over projection steps without storing intermediate arrays
- **Memory Efficiency**: Optional `jax.checkpoint` for reduced memory during backpropagation
- **Geometric Parameterization**: 5-parameter rigid-body model (α, β, φ, Δx, Δz) for per-view misalignment
- **Physical Accuracy**: Includes step-size scaling and proper coordinate transforms
- **Differentiability**: Full JAX autodiff support for gradient-based optimization

## Development Commands

### Environment Setup
```bash
# Activate pixi environment (automatically configured)
pixi shell
```

### Running Examples
```bash
# Generate phantom and compute projections
python examples/run_parallel_projector.py --nx 128 --ny 128 --nz 128 --n-proj 128

# Run FBP reconstruction
python examples/run_parallel_reconstruction.py --input-dir parallel_proj_test

# Custom phantom with more objects
python examples/run_parallel_projector.py --n-cubes 20 --n-spheres 20 --max-size 64
```

### Core API Usage

**Forward Projection (Single View)**:
```python
proj = forward_project_view(
    params=jnp.array([alpha, beta, phi, dx, dz]),  # 5-DOF rigid params
    recon_flat=volume.ravel(),  # Flattened volume (nx*ny*nz,)
    nx=nx, ny=ny, nz=nz,        # Grid dimensions
    vx=vx, vy=vy, vz=vz,        # Voxel sizes
    nu=nu, nv=nv,               # Detector size
    du=du, dv=dv,               # Detector pixel size
    vol_origin=vol_origin,       # Volume origin in world coords
    det_center=det_center,       # Detector center
    step_size=step_size,         # Integration step size
    n_steps=n_steps             # Number of integration steps
)
```

**Loss and Gradient (for Alignment)**:
```python
loss, grad = view_loss_value_and_grad(
    params, measured_proj, recon_flat, grid, det
)
```

## File Organization

- All files are kept under 300 LOC following modularity requirements
- Volume data stored as flattened 1D arrays in C-order: `index = ix * (ny*nz) + iy * nz + iz`
- Grid and detector parameters passed as dictionaries for flexibility
- TIFF files used for 3D data storage (z-stack format)

## Testing and Validation

- **Adjoint Test**: `adjoint_test_once()` validates forward/backward consistency via VJP
- **Memory Profiling**: All examples include RSS memory tracking
- **Finite Difference**: Manual gradient validation recommended for new geometric parameters
- **Visual Validation**: Phantom generation includes FOV constraint checking

## Performance Notes

- **CPU (Mac)**: Sequential view processing for memory efficiency
- **GPU**: Optional batched processing via `batch_forward_project()`
- **Memory**: Typical 256³ volume requires ~hundreds of MB peak memory
- **Speed**: ~0.06s per view on CPU for 256³→256² projection

## Dependencies

Managed via `pixi.toml`:
- JAX 0.5.3 (core computation with CUDA support)
- optax 0.2.5 (gradient-based optimization)
- NumPy ≥2.3.2 (data handling) 
- SciPy ≥1.16.1 (reconstruction filters)
- tifffile ≥2025.8.28 (I/O)
- matplotlib ≥3.10.5 (visualization)
- scikit-image ≥0.25.2 (phantom generation)
- psutil ≥7.0.0 (memory monitoring)

### Important Note on JAX/Optax Compatibility
JAX is constrained to `>=0.5.3,<0.6` to maintain compatibility with both:
- System gRPC libraries (conda-installed version 2301.0.0)  
- Optax requirements (needs jax≥0.5.3)

See `JAX_OPTAX_COMPATIBILITY_FIX.md` for details on version compatibility issues and solutions.