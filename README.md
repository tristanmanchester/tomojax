# TomoJAX: Differentiable CT Projector with Alignment

**TomoJAX** is a fully differentiable, memory-efficient parallel-beam CT projector implemented in JAX. It provides exact gradients for **5-DOF rigid-body alignment optimization**, making it ideal for CT reconstruction, view alignment, and deep learning applications requiring data consistency.

#### Alignment demonstration
<img src="images/montage_scroll.gif" width="1000">

**Left to Right**: Ground truth phantom → Naive reconstructions (misaligned & added noise) → Aligned reconstructions


## Key Features

### **Core Projector**
- **Differentiable forward projection** with exact gradients via JAX autodiff
- **5-DOF rigid-body parameterization**: `T x = R_y(β) R_x(α) R_z(φ) x + t` where `t = (Δx, 0, Δz)`
- **Memory efficient**: Streaming integration via `lax.scan` — O(detector pixels) memory usage
- **Trilinear interpolation** with analytical derivatives
- **CPU/GPU support** with JIT compilation

### **Joint Reconstruction & Alignment**
- **Multi-resolution approach**: Hierarchical optimization (4x → 2x → 1x binning)
- **Alternating optimization**: FISTA-TV reconstruction + per-view alignment
- **Multiple optimisers**: AdaBelief (recommended), Adam, L-BFGS, hybrid approaches
- **4-DOF or 5-DOF**: Option to optimise projection angle φ or keep fixed

### **Alignment Capabilities**
- **Per-view misalignment correction** for all 5 rigid-body parameters:
  - **α** (pitch): rotation around x-axis  
  - **β** (roll): rotation around y-axis
  - **φ** (yaw): rotation around z-axis (projection angle)
  - **Δx**: horizontal translation
  - **Δz**: vertical translation
- **Automatic parameter scaling** for different parameter sensitivities
- **Alignment accuracy metrics** (RMSE in degrees/pixels)

### 🏗**Reconstruction**
- **FISTA-TV regularization** with Chambolle-Pock proximal operator
- **Automatic Lipschitz estimation** for optimal step sizes
- **Alignment-aware reconstruction** with exact gradients
- **FBP initialization** and iterative refinement

## Quick Start

### Basic Forward Projection
```python
import jax.numpy as jnp
from projector_parallel_jax import forward_project_view

# Project single view with alignment parameters
proj = forward_project_view(
    params=jnp.array([alpha, beta, phi, dx, dz]),  # 5-DOF rigid params
    recon_flat=volume.ravel(),                     # Flattened volume
    nx=nx, ny=ny, nz=nz,                          # Grid dimensions
    vx=vx, vy=vy, vz=vz,                          # Voxel sizes
    nu=nu, nv=nv,                                 # Detector size
    du=du, dv=dv,                                 # Detector pixel size
    vol_origin=vol_origin,                        # Volume origin
    det_center=det_center,                        # Detector center
    step_size=step_size,                          # Integration step
    n_steps=n_steps                               # Steps count
)
```

### Joint Reconstruction & Alignment
```bash
# Generate misaligned test data
python examples/run_parallel_projector_misaligned.py \
    --nx 128 --ny 128 --nz 128 --n-proj 128 \
    --max-trans-pixels 3.0 --max-rot-degrees 2.0 \
    --output-dir misaligned_test

# Run joint optimisation
python alignment-testing/run_alignment.py \
    --input-dir misaligned_test \
    --bin-factors 4 2 1 \
    --outer-iters 15 \
    --optimizer adabelief \
    --lambda-tv 0.005
```

## Alignment Algorithm

The implementation uses multi-resolution alternating optimisation:

1. **Multi-resolution pyramid**: Start at coarse resolution for global alignment
2. **Alternating steps**: 
   - Fix alignment → FISTA-TV reconstruction
   - Fix reconstruction → optimise alignment parameters
3. **Parameter transfer**: Scale alignment between resolution levels
4. **Convergence**: Monitor objective function and parameter changes


## Examples & Usage

| Script | Description |
|--------|-------------|
| `examples/run_parallel_projector.py` | Basic projection demo |
| `examples/run_parallel_projector_misaligned.py` | Generate misaligned test data |
| `examples/run_parallel_reconstruction.py` | FBP reconstruction |
| `examples/run_fista_tv.py` | TV-regularized iterative reconstruction |
| `alignment-testing/run_alignment.py` | **Joint reconstruction & alignment** |
| `alignment-testing/optimizer_comparison.py` | Benchmark alignment optimizers |

## Visual Examples

### Basic Projector Workflow
| Phantom | Projections | Sinogram | Reconstruction |
|---------|-------------|----------|----------------|
| <img src="images/phantom_slice.png" width="200"><br><img src="images/phantom_volume.png" width="200"> | <img src="images/projections.gif" width="200"> | <img src="images/sinogram.png" width="200"> | <img src="images/recon_slice.png" width="200"><br><img src="images/recon_volume.png" width="200"> |
| Top: slice<br>Bottom: volume projection | Animated over 360° | Angle vs detector | Top: slice<br>Bottom: volume projection |

### Alignment & Reconstruction Workflow

#### Misaligned Input Data
| Clean Misaligned Projections | Noisy Misaligned Projections |
|------------------------------|------------------------------|
| <img src="images/spin_projections_misaligned.gif" width="300"> | <img src="images/spin_projections_noisy.gif" width="300"> |
| Random rigid-body misalignments | Same misalignments + Poisson noise |

#### Sinogram Analysis
| Clean Misaligned Sinogram | Noisy Misaligned Sinogram |
|---------------------------|---------------------------|
| <img src="images/misaligned_sinogram.png" width="300"> | <img src="images/noisy_sinogram.png" width="300"> |
| Clear view-to-view inconsistencies | Inconsistencies + noise artifacts |

#### Multi-Resolution Alignment Process
| Clean Data Alignment | Noisy Data Alignment |
|---------------------|---------------------|
| <img src="images/alignment_process_misaligned.gif" width="300"> | <img src="images/alignment_process_noisy.gif" width="300"> |
| 4x → 2x → 1x resolution refinement | Robust alignment despite noise |
