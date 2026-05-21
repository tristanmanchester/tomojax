from __future__ import annotations

import ast
import inspect

# check-public-imports: allow-private
from tomojax.core import _pallas_adjoint_kernels, _pallas_single_view


def _unroll_none_branch(function: object) -> ast.If:
    source = inspect.getsource(function)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "unroll is None":
            return node
    raise AssertionError(f"{function.__name__} has no unroll is None branch")


def _fori_loop_upper_bounds(function: object) -> list[str]:
    branch = _unroll_none_branch(function)
    bounds: list[str] = []
    for stmt in branch.body:
        bounds.extend(
            ast.unparse(node.args[1])
            for node in ast.walk(stmt)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fori_loop"
                and len(node.args) >= 2
            )
        )
    return bounds


def test_pallas_cached_kernels_bound_dynamic_loops_to_active_tile_steps() -> None:
    kernels = (
        _pallas_single_view._projector_kernel_cached,
        _pallas_adjoint_kernels._backproject_kernel,
        _pallas_adjoint_kernels._projector_residual_sse_kernel,
    )

    for kernel in kernels:
        bounds = _fori_loop_upper_bounds(kernel)
        assert bounds
        assert set(bounds) == {"tile_steps"}
