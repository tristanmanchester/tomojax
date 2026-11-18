"""The joint operators with arrays in and arrays out, for tests that check their algebra.

A level holds its stack-sized arrays as ``_stacks.Stack`` handles and the
operators read and write them a batch at a time; these helpers wrap padded
arrays as resident stacks and unwrap the results.
"""

import jax.numpy as jnp
import numpy as np

# check-public-imports: allow-private
from tomojax.alignment._joint._solver import _Level
from tomojax.alignment._joint._stacks import LevelStacks, Stack


def stack(ops, array):
    """A padded per-view array as a resident stack; None stays None (uniform precision)."""
    return None if array is None else ops.stack(array)


def arrays(result):
    """A result tuple with each stack replaced by its device array."""
    return tuple(item.data if isinstance(item, Stack) else item for item in result)


def level_stacks(ops, y, precision=None, placement=None):
    """A level's stacks holding already padded ``y`` and ``precision``, resident unless placed."""
    data = {"y": np.asarray(y, np.float32)}
    if precision is not None:
        data["precision"] = np.asarray(precision, np.float32)
    specs = ops.stack_specs(precision)
    return LevelStacks(specs, ops.batch, data=data, room=None, reserve=0, placement=placement)


def placed_level(ops, prior, y, precision=None, **options):
    """A solver level placed on already padded ``y`` and ``precision`` (``_Level.place``)."""
    level = _Level(ops, prior, **options)
    level.place(*(None if a is None else np.asarray(a)[: ops.n] for a in (y, precision)))
    return level


def linearize(ops, q, x, y, precision, delta):
    """``Operators.linearize`` on arrays, into fresh resident stacks."""
    specs = ops.stack_specs(None)
    volumes = (jnp.zeros(ops.shape, jnp.float32), jnp.zeros(ops.shape, jnp.float32))
    out = volumes + tuple(ops.stack(jnp.zeros(*specs[n])) for n in ("sw", "j", "residual"))
    return arrays(ops.linearize(q, x, stack(ops, y), stack(ops, precision), delta, out))


def weighted_forward(ops, q, x, sw, j, precision=None):
    """``Operators.weighted_forward`` on arrays: the weighted projections and their scores."""
    r = ops.stack(jnp.zeros(ops.image_shape, jnp.float32))
    r, score = ops.weighted_forward(q, x, stack(ops, sw), stack(ops, j), stack(ops, precision), r)
    return r.data, score


def weighted_back(ops, q, r, sw, j, p, active=None, precision=None):
    stacks = (stack(ops, a) for a in (r, sw, j))
    return ops.weighted_back(q, *stacks, p, active, stack(ops, precision))


def pose_back(ops, q, sw, j, p, active=None, precision=None):
    return ops.pose_back(q, stack(ops, sw), stack(ops, j), p, active, stack(ops, precision))
