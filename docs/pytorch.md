# Differentiable projection in PyTorch

`tomojax.torch.Projector` wraps TomoJAX's CUDA forward and adjoint projectors as a PyTorch autograd operator using DLPack, with zero host copies. Install `tomojax[cuda12]` alongside a CUDA build of PyTorch matching your system.

```python
import numpy as np
import torch
import tomojax as tj
from tomojax.torch import Projector

geometry = tj.ParallelGeometry(
    tj.Grid(64, 64, 64, 1.0, 1.0, 1.0),
    tj.Detector(64, 64, 1.0, 1.0),
    np.linspace(0, 180, 90, endpoint=False),
)
operator = Projector(geometry)
volume = torch.ones((64, 64, 64), device="cuda", requires_grad=True)
projections = operator(volume)            # (view, v, u)
loss = projections.square().mean()
loss.backward()                           # exact transpose, not FBP
backprojection = operator.adjoint(projections)
```

`Projector` accepts parallel, laminography, and cone-beam geometries, including calibrated poses. Volumes use `(z, y, x)` order and all geometry lengths share one physical unit. Both `operator(volume)` and `operator.adjoint(projections)` support autograd with respect to tensor inputs (geometry and poses are fixed at construction).

Inputs must be contiguous CUDA `float32` tensors with 16-byte-aligned storage, which standard PyTorch allocations satisfy; `.clone()` an unaligned slice view before passing it. Strided or broadcast upstream gradients from PyTorch are copied contiguous automatically for the backward pass. Do not modify an input tensor while a projection is running. Each direction compiles on its first call and reuses the compiled kernel thereafter.

Because JAX and PyTorch share GPU memory, importing `tomojax` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (unless already set) so JAX allocates on demand instead of reserving 75% of the GPU up front. Import `tomojax` before JAX runs its first operation; see [GPU allocator](large-scans.md#gpu-allocator) for details.
