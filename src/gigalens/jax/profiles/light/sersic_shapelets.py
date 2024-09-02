import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile
from gigalens.jax.profiles.light import sersic, shapelets


class SersicShapelets(gigalens.profile.LightProfile):
    _name = "SERSIC_SHAPELETS"
    _params = []
    _amp = "Ie"

    def __init__(self, n_max, use_lstsq=False, is_source=False, interpolate=True):
        super(SersicShapelets, self).__init__(use_lstsq=use_lstsq, is_source=is_source)
        self.sersic = sersic.SersicEllipse(use_lstsq=use_lstsq, is_source=is_source)
        self.shapelets = shapelets.Shapelets(n_max, use_lstsq=use_lstsq, is_source=is_source, interpolate=interpolate)
        self.shared_params = ["center_x", "center_y", "e1", "e2"]
        self.params = []
        self.depth = self.sersic.depth + self.shapelets.depth
        for param in self.sersic.params + self.shapelets.params:
            if param not in self.params:
                self.params.append(param)
            else:
                if param not in self.shared_params:
                    self.shared_params.append(param)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, **params):
        ret_sersic = self.sersic.light(x, y, **{param: params[param] for param in self.sersic.params})
        ret_shapelets = self.shapelets.light(x, y, **{param: params[param] for param in self.shapelets.params})
        if self.use_lstsq:
            return jnp.concatenate((ret_sersic, ret_shapelets), axis=0)
        else:
            return ret_sersic + ret_shapelets
