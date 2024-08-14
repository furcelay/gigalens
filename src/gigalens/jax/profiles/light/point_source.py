import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class PointSource(gigalens.profile.LightProfile):
    _name = "POINT_SOURCE"
    _params = []

    def __init__(self, use_lstsq=False, is_source=True):
        super(PointSource, self).__init__(use_lstsq=use_lstsq, is_source=is_source)
        self.depth = 0
        if not self.use_lstsq:
            self.params.pop(self.params.index(self._amp))

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y):
        if self.use_lstsq:
            return jnp.zeros_like(x)
        else:
            jnp.zeros((0, x.shape))
