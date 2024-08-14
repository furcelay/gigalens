import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class PointSource(gigalens.profile.LightProfile):
    _name = "POINT_SOURCE"
    _params = []
    _amp = "amp"

    def __init__(self, use_lstsq=False, is_source=True):
        super(PointSource, self).__init__(use_lstsq=use_lstsq, is_source=is_source)
        if not self.use_lstsq:
            self.params.pop(self.params.index(self._amp))

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y):
        ret = jnp.zeros_like(x)
        return ret[jnp.newaxis, ...] if self.use_lstsq else ret
