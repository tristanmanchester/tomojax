# Differentiable projection in PyTorch

Install TomoJAX's CUDA dependencies and a CUDA build of PyTorch matching your
system. PyTorch is optional: ordinary TomoJAX workflows do not import it.

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
projections = operator(volume)             # (view, v, u)
loss = projections.square().mean()
loss.backward()                           # exact transpose, not FBP
backprojection = operator.adjoint(projections)
```

The same interface accepts cone-beam and laminography geometries, including
calibrated poses. Volumes use TomoJAX's `(z, y, x)` order and geometry lengths
must use consistent units. The geometry is fixed at construction; gradients
are with respect to tensors, not poses. Both directions support autograd.

Pass contiguous CUDA float32 tensors with 16-byte-aligned storage (standard
PyTorch allocations); clone an unaligned offset view first. DLPack shares storage with JAX,
including the output storage returned to PyTorch, and handles CUDA stream
dependencies. No host transfers are involved. A strided or broadcast gradient
created by PyTorch is made contiguous for the backward projector. Do not modify
an input concurrently with projection. Each direction compiles on its first
call and reuses that executable thereafter.

JAX and PyTorch share GPU memory. When a training process needs most of the
GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX or TomoJAX
so JAX does not reserve memory needed by the network.
