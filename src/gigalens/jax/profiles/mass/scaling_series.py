from typing import List

import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile
from gigalens.jax.series.series_profile import MassSeries
from gigalens.jax.profiles.mass.scaling_relation import ScalingRelation


class ScalingRelationSeries(MassSeries, ScalingRelation):
    _constants: List[str]

    def __init__(self, profile: MassSeries, **kwargs):
        self._series_param = profile.series_param
        self._amplitude_param = profile.amplitude_param
        super(ScalingRelationSeries, self).__init__(profile=profile, **kwargs)

        self.params = self._params = self.profile.params
        self.scaling_constants = [p for p in self.scaling_params if p in self.constants]

    @functools.partial(jit, static_argnums=(0, 1))  # TODO: check if jit is convenient
    def precompute_deriv(self, order, x, y, **scales):
        scales[self.amplitude_param] = 1.
        out_shape = (*x.shape, self.order + 1)
        f_x, f_y = jnp.zeros(out_shape), jnp.zeros(out_shape)
        scaled = self.scale_params(scales)
        n = jnp.arange(order + 1)
        for s_chunk, u_chunk, c_chunk in zip(scaled, self._unscaled_params, self._galaxy_constants):
            amplitude_factor = jnp.expand_dims(u_chunk[self.amplitude_param], 0)
            amplitude_factor = jnp.expand_dims(amplitude_factor, -1)  # 1, chunk, 1
            series_factor = jnp.expand_dims(u_chunk[self.series_param], 0)
            series_factor = jnp.expand_dims(series_factor, -1)
            series_factor = jnp.power(series_factor, n)  # 1, chunk, n + 1
            pre_factor = amplitude_factor * series_factor
            f_x_chunk, f_y_chunk = self.profile.precompute_deriv(order, x, y, **s_chunk, **c_chunk)
            f_x += jnp.sum(pre_factor * f_y_chunk, -2, keepdims=True)
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0, 1))  # TODO: check if jit is convenient
    def precompute_hessian(self, order, x, y, **scales):
        scales[self.amplitude_param] = 1.
        out_shape = (*x.shape, self.order + 1)
        f_xx, f_xy, f_yy = jnp.zeros(out_shape), jnp.zeros(out_shape), jnp.zeros(out_shape)
        scaled = self.scale_params(scales)
        n = jnp.arange(order + 1)
        for s_chunk, u_chunk, c_chunk in zip(scaled, self._unscaled_params, self._galaxy_constants):
            amplitude_factor = jnp.expand_dims(u_chunk[self.amplitude_param], 0)
            amplitude_factor = jnp.expand_dims(amplitude_factor, -1)  # 1, chunk, 1
            series_factor = jnp.expand_dims(u_chunk[self.series_param], 0)
            series_factor = jnp.expand_dims(series_factor, -1)
            series_factor = jnp.power(series_factor, n)  # 1, chunk, n + 1
            pre_factor = amplitude_factor * series_factor
            f_xx_chunk, f_xy_chunk, f_yy_chunk = self.profile.precompute_hessian(order, x, y, **s_chunk, **c_chunk)
            f_xx += jnp.sum(pre_factor * f_xx_chunk, -2, keepdims=True)
            f_xy += jnp.sum(pre_factor * f_xy_chunk, -2, keepdims=True)
            f_yy += jnp.sum(pre_factor * f_yy_chunk, -2, keepdims=True)
        return f_xx, f_xy, f_yy

    def convergence(self, x, y, **kwargs):
        return MassProfile.convergence(self, x, y, **kwargs)

    def shear(self, x, y, **kwargs):
        return MassProfile.shear(self, x, y, **kwargs)

