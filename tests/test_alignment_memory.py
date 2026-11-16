"""A level's working memory beside its stacks must not grow with the number of views.

The plan reserves the level's other arrays as a count of volumes (``_stacks.level_reserve``).
That is only right if nothing a level compiles holds a temporary that scales with the views:
K11 67208 (4000 views) failed in its Schur solve because conjugate gradients copied the
whole weighted-projection stack ``r`` out of a conditional on every iteration, a temporary
of 0.47 GiB at 1441 views and 1.29 GiB at 4000 that the plan had counted as volumes.
"""

import jax
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint import _fit
from tomojax.geometry import stack_view_poses


def _scan(views, nu=96, nv=64, n=24, nz=16):
    grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(nu, nv, 1, 1)
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=-30)
    rng = np.random.default_rng(3)
    volume = rng.uniform(0, 1, (nz, n, n)).astype(np.float32)
    projections = np.array(tj.project(geometry, volume))
    projections += rng.normal(0, 0.01, projections.shape).astype(np.float32)
    return projections, np.asarray(stack_view_poses(geometry, views)), grid, detector


ARGS = {}


def _abstract(tree):
    def shape(leaf):
        if isinstance(leaf, jax.Array):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, weak_type=leaf.weak_type)
        return leaf

    return jax.tree.map(shape, tree)


def level_temporaries(views, monkeypatch, nu=96, nv=64, n=24, nz=16):
    """XLA's temporary bytes of every function a one-level fit compiles, by name."""
    jit, seen = jax.jit, {}

    def recording(fun=None, **options):
        if fun is None:
            return lambda f: recording(f, **options)
        compiled = jit(fun, **options)
        name = getattr(fun, "__qualname__", repr(fun))
        if "_joint" not in getattr(fun, "__module__", "") or name in seen:
            return compiled

        def call(*args, **kwargs):
            leaves = jax.tree.leaves((args, kwargs))
            if name not in seen and not any(isinstance(a, jax.core.Tracer) for a in leaves):
                seen[name] = (compiled, _abstract(args), _abstract(kwargs))
            return compiled(*args, **kwargs)

        return call

    monkeypatch.setattr(jax, "jit", recording)
    monkeypatch.setattr(_fit, "_LEVELS", ((1, 2),))
    projections, nominal, grid, detector = _scan(views, nu, nv, n, nz)
    _fit.fit_poses(projections, nominal, grid, detector, keep_grid=True)
    monkeypatch.setattr(jax, "jit", jit)
    out = {}
    for name, (compiled, args, kwargs) in seen.items():
        ARGS[name] = (compiled, args, kwargs)
        analysis = compiled.lower(*args, **kwargs).compile().memory_analysis()
        out[name] = analysis.temp_size_in_bytes
    return out


@pytest.mark.gpu
def test_no_compiled_level_function_keeps_a_temporary_that_grows_with_the_views(monkeypatch):
    # The same grid and detector at 48 and 144 views, compiled for the GPU, whose buffer
    # assignment made the copy (the CPU's did not). A temporary that holds a stack grows by at
    # least two bytes a pixel for each added view (FP16); the motion prior's per-view matrices
    # grow by about 2 KiB a view whatever the detector. At 96 x 64 pixels a stack is 12 KiB a
    # view at FP16, so one byte a pixel separates the two.
    few = level_temporaries(48, monkeypatch)
    many = level_temporaries(144, monkeypatch)
    limit = (144 - 48) * 96 * 64
    grown = {name: many[name] - few[name] for name in few if many.get(name, 0) - few[name] > limit}
    assert not grown, f"temporaries growing with the views (bytes, limit {limit}): {grown}"


def _conditionals(jaxpr):
    """Every cond equation in a jaxpr and the jaxprs nested in it."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "cond":
            yield eqn
        for value in eqn.params.values():
            for item in value if isinstance(value, list | tuple) else [value]:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from _conditionals(inner)


def test_the_schur_solves_conditionals_carry_no_stack(monkeypatch):
    # Conjugate gradients chose between advancing and verifying in a conditional that also made
    # the product and returned the weighted-projection stack r it had written. XLA keeps a
    # conditional's results in buffers of their own, so at K11's size the compiled solve copied
    # r out of it on every iteration: a whole stack of temporaries (0.47 GiB at 1441 views,
    # 1.29 GiB at 4000) that the plan did not count. The product is made outside the
    # conditional; nothing the size of the views may pass through one.
    level_temporaries(48, monkeypatch)
    compiled, args, kwargs = next(v for k, v in ARGS.items() if k.endswith("schur"))
    views = args[1].shape[0]  # the weights sw: (padded views, rows, pixels)
    jaxpr = jax.make_jaxpr(compiled)(*args, **kwargs).jaxpr
    carried = [
        var.aval.shape
        for eqn in _conditionals(jaxpr)
        for var in eqn.outvars
        if len(var.aval.shape) >= 3 and var.aval.shape[0] == views
    ]
    assert not carried, f"stack-sized results of a conditional in the Schur solve: {carried}"
