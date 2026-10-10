"""Weighted quadratic dual proximal, without physical-scale numerical floors."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _frexp(x):
    return jnp.frexp(x)


@_frexp.defjvp
def _frexp_jvp(primals, tangents):
    (x,), (tangent,) = primals, tangents
    mantissa, exponent = _frexp(x)
    # JAX's exp2(-exponent) tangent adds avoidable relative error at large
    # integer exponents. The value of this exact power needs no differentiation.
    slope = tangent * jnp.ldexp(jnp.ones_like(x), -exponent)
    zero_exponent = jax.lax.full_like(exponent, 0, dtype=jax.dtypes.float0)
    return (mantissa, exponent), (slope, zero_exponent)


@jax.custom_jvp
def _binary_sum(a, ea, b, eb):
    """Add two mantissa/exponent terms before restoring their physical scale."""
    ea = jnp.where(a == 0, -1024, ea)
    eb = jnp.where(b == 0, -1024, eb)
    exponent = jnp.maximum(ea, eb)
    mantissa = jnp.ldexp(a, ea - exponent) + jnp.ldexp(b, eb - exponent)
    return jnp.ldexp(mantissa, exponent)


@functools.partial(_binary_sum.defjvp, symbolic_zeros=True)
def _binary_sum_jvp(primals, tangents):
    a, ea, b, eb = primals
    da, _, db, _ = tangents
    value = _binary_sum(*primals)
    terms = [
        tangent * jnp.ldexp(jnp.ones_like(value), exponent)
        for tangent, exponent in [(da, ea), (db, eb)]
        if not isinstance(tangent, jax.custom_derivatives.SymbolicZero)
    ]
    return value, sum(terms, jnp.zeros_like(value))


def _binary_parts(u, sigma, y, w):
    mu, eu = _frexp(u)
    ms, es = _frexp(sigma)
    my, ey = _frexp(y)
    mw, ew = _frexp(w)
    ed = jnp.maximum(es, ew)
    denominator = jnp.ldexp(ms, es - ed) + jnp.ldexp(mw, ew - ed)
    return mu, eu, ms, es, my, ey, mw, ew, ed, denominator


def _balanced_l2(u, sigma, y, w):
    """Balance affine products whose intermediate values exceed the dtype range."""
    mu, eu, ms, es, my, ey, mw, ew, ed, d = _binary_parts(u, sigma, y, w)
    value = _binary_sum(mu * mw / d, eu + ew - ed, -ms * my * mw / d, es + ey + ew - ed)
    return jnp.where(w > 0, value, 0.0)


def _balanced_partials(args):
    u, sigma, y, w = args
    mu, eu, ms, es, my, ey, mw, ew, ed, d = _binary_parts(u, sigma, y, w)
    du = jnp.ldexp(mw / d, ew - ed)
    dy = jnp.ldexp(-ms * mw / d, es + ew - ed)
    ds = _binary_sum(
        -mw * mu / (d * d), ew + eu - 2 * ed, -mw * mw * my / (d * d), 2 * ew + ey - 2 * ed
    )
    dw = _binary_sum(
        ms * mu / (d * d), es + eu - 2 * ed, -ms * ms * my / (d * d), 2 * es + ey - 2 * ed
    )
    return du, ds, dy, dw


def _ordinary_l2(args):
    u, sigma, y, w = args
    return jnp.where(w > 0, (u - sigma * y) * (w / (sigma + w)), 0.0)


def _ordinary_range(u, sigma, y, w):
    limits = jnp.finfo(u.dtype)
    exponent = (min(limits.maxexp, -limits.minexp) - 4) // 2
    lower, upper = 2.0 ** (-exponent), 2.0**exponent
    return jnp.all(
        (sigma >= lower)
        & (sigma <= upper)
        & (jnp.abs(u) <= upper)
        & (jnp.abs(y) <= upper)
        & ((w == 0) | ((w >= lower) & (w <= upper)))
    )


@jax.custom_jvp
def weighted_l2_conjugate(u, sigma, y, w):
    """Return ``(u - sigma*y)*w/(sigma+w)``, or exactly zero for masked weights.

    A dtype-derived exponent guard keeps the ordinary affine numerator finite
    and its positive gain normal. Outside that range, exponent-balanced affine
    terms avoid product/denominator overflow and premature gain underflow.
    No absolute lower bound changes the requested quadratic or its minimizer.
    """
    return jax.lax.cond(
        _ordinary_range(u, sigma, y, w),
        _ordinary_l2,
        lambda args: _balanced_l2(*args),
        (u, sigma, y, w),
    )


def _ordinary_partials(args):
    u, sigma, y, w = args
    denominator = sigma + w
    gain, complement = w / denominator, sigma / denominator
    scaled_u = u / denominator
    return (
        gain,
        -gain * (scaled_u + gain * y),
        -sigma * gain,
        complement * (scaled_u - complement * y),
    )


@functools.partial(weighted_l2_conjugate.defjvp, symbolic_zeros=True)
def _weighted_l2_conjugate_jvp(primals, tangents):
    # ldexp's automatic derivative at zero is not its mathematical scaling.
    # Use analytic, primal-only partials so this rule remains linear in the
    # tangents and transposes correctly for reverse mode, including cancellation.
    # Factored ordinary partials also avoid inverse-cube Hessian intermediates.
    u, sigma, y, w = primals
    value = weighted_l2_conjugate(*primals)
    partials = jax.lax.cond(
        _ordinary_range(u, sigma, y, w), _ordinary_partials, _balanced_partials, primals
    )
    terms = [
        jnp.where(w > 0, partial, 0.0) * tangent
        for partial, tangent in zip(partials, tangents, strict=True)
        if not isinstance(tangent, jax.custom_derivatives.SymbolicZero)
    ]
    return value, sum(terms, jnp.zeros_like(value))
