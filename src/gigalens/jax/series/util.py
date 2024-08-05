from jax import numpy as jnp
from jax import lax
from jax._src.numpy.util import promote_args_inexact


def factorial(n):
    n, = promote_args_inexact("factorial", n)
    return jnp.where(n < 0, 0, lax.exp(lax.lgamma(n + 1)))
