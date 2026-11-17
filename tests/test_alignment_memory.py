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


def level_memory(views, monkeypatch, nu=96, nv=64, n=24, nz=16):
    """Temporaries, outputs and captured constants (bytes) of each compiled level function.

    By name, with the fit's motion prior.
    """
    jit, seen, priors = jax.jit, {}, []

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

    built = _fit.MotionPrior

    def prior(*args, **kwargs):
        priors.append(built(*args, **kwargs))
        return priors[-1]

    monkeypatch.setattr(jax, "jit", recording)
    monkeypatch.setattr(_fit, "MotionPrior", prior)
    monkeypatch.setattr(_fit, "_LEVELS", ((1, 2),))
    projections, nominal, grid, detector = _scan(views, nu, nv, n, nz)
    _fit.fit_poses(projections, nominal, grid, detector, keep_grid=True)
    monkeypatch.setattr(jax, "jit", jit)
    monkeypatch.setattr(_fit, "MotionPrior", built)
    out = {}
    for name, (compiled, args, kwargs) in seen.items():
        ARGS[name] = (compiled, args, kwargs)
        with jax.enable_x64(True):  # as fit_poses runs them: its FP64 stays FP64
            analysis = compiled.lower(*args, **kwargs).compile().memory_analysis()
            consts = jax.make_jaxpr(compiled)(*args, **kwargs).consts
        out[name] = {
            "temp": analysis.temp_size_in_bytes,
            "output": analysis.output_size_in_bytes,
            "consts": sum(getattr(c, "nbytes", 0) for c in consts),
        }
    return out, priors[-1]


def level_temporaries(views, monkeypatch, nu=96, nv=64, n=24, nz=16):
    """XLA's temporary bytes of every function a one-level fit compiles, by name."""
    memory, _ = level_memory(views, monkeypatch, nu, nv, n, nz)
    return {name: m["temp"] for name, m in memory.items()}


# Known to grow faster than the views, with the growth order measured from 300 to 900 views:
# the motion prior's factor, built from its dense spline design (local rows by about
# views / 10 knots). Its temporaries and the innovation solve's grow the same way at more
# views (5 to 6 copies of the design at 1000 to 4000), but a fixed part hides that at these
# counts. Item 64 makes the prior linear; remove the entry then.
SUPERLINEAR = {
    "MotionPrior._factor": 1.9,  # its output, measured in FP64 as the fit runs
}


def _order(small, large, ratio):
    """The growth exponent between two view counts, or 0 for what is too small to grow."""
    if large <= 256 * 1024 or small <= 0:
        return 0.0
    return float(np.log(large / small) / np.log(ratio))


@pytest.mark.gpu
def test_no_level_function_grows_faster_than_the_views(monkeypatch):
    # A reserve counted in volumes and a plan's prior term are only right if what each compiled
    # function holds grows no faster than the views. At 100, 300 and 900 views on one grid and
    # detector, each function's temporaries, outputs and captured constants are measured; the
    # growth order from 300 to 900 must be at most 1.3 (linear, with fixed parts), except the
    # motion prior's known quadratic functions, which must still need their exception.
    counts = (100, 300, 900)
    profiles = [level_memory(views, monkeypatch) for views in counts]
    ratio = counts[2] / counts[1]
    superlinear = {}
    for name, memory in profiles[2][0].items():
        for kind in ("temp", "output", "consts"):
            small = profiles[1][0].get(name, {}).get(kind, 0)
            order = _order(small, memory[kind], ratio)
            if order > 1.3:
                superlinear[name] = max(order, superlinear.get(name, 0.0))
    unexpected = {name: order for name, order in superlinear.items() if name not in SUPERLINEAR}
    assert not unexpected, f"growing faster than the views (order from 300 to 900): {unexpected}"
    gone = set(SUPERLINEAR) - set(superlinear)
    assert not gone, f"no longer superlinear, remove from SUPERLINEAR (item 64): {gone}"
    # No function captures a large array as a compiled constant.
    captured = {name: m["consts"] for name, m in profiles[2][0].items() if m["consts"] > 256 * 1024}
    assert not captured, f"large captured constants (bytes): {captured}"
    # The plan's prior term covers what the prior holds and needs, within half again.
    for memory, prior in profiles[1:]:
        planned = prior.device_bytes()
        held = sum(int(a.size) * a.dtype.itemsize for a in prior.matrices)
        held += memory["MotionPrior._factor"]["output"]
        transient = max(memory[n]["temp"] for n in memory if n.startswith("MotionPrior."))
        assert held <= planned["held"] <= 1.5 * held, (planned, held)
        assert transient <= planned["transient"] <= 2 * transient, (planned, transient)


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
    level_memory(48, monkeypatch)
    compiled, args, kwargs = next(v for k, v in ARGS.items() if k.endswith("schur"))
    views = args[1].shape[0]  # the weights sw: (padded views, rows, pixels)
    with jax.enable_x64(True):
        jaxpr = jax.make_jaxpr(compiled)(*args, **kwargs).jaxpr
    carried = [
        var.aval.shape
        for eqn in _conditionals(jaxpr)
        for var in eqn.outvars
        if len(var.aval.shape) >= 3 and var.aval.shape[0] == views
    ]
    assert not carried, f"stack-sized results of a conditional in the Schur solve: {carried}"
