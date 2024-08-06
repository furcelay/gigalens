import functools

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import factorial

from gigalens.jax.profile import MassProfile
from abc import ABC, abstractmethod


__all__ = ['MassSeries']


class MassSeries(MassProfile, ABC):
    """
    Series expansion of the given potential on a single variable
    """

    _series_param: str
    _amplitude_param: str
    _name = "SeriesExpansion"
    _constants = []

    def __init__(self, grid=(None, None), params=None, order=3, **kwargs):
        self.series_param = self._series_param
        self.amplitude_param = self._amplitude_param
        self._series_var_0 = None
        self.constants = self._constants
        self._order = order
        self._constants_dict = {}
        if params is not None:
            self._constants_dict = params.copy()
            self._series_var_0 = params[self.series_param]
        self._x, self._y = grid
        self._f_x, self._f_y = None, None
        self._f_xx, self._f_xy, self._f_yy = None, None, None

        super(MassSeries, self).__init__(**kwargs)

    @property
    def order(self):
        return self._order

    @property
    def series_var_0(self):
        return self._series_var_0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def constants_dict(self):
        return self._constants_dict

    def set_constants(self, params):
        self._series_var_0 = params[self.series_param]
        self._constants_dict = params.copy()

    def set_grid(self, x, y):
        self._x, self._y = x, y

    def set_deriv(self):
        self._f_x, self._f_y = self.precompute_deriv(self.x, self.y, **self.constants_dict)

    def set_hessian(self):
        self._f_xx, self._f_xy, self._f_yy = self.precompute_hessian(self.x, self.y, **self.constants_dict)

    @abstractmethod
    def precompute_deriv(self, x, y, **kwargs):
        pass

    @abstractmethod
    def precompute_hessian(self, x, y, **kwargs):
        pass

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, **kwargs):  # TODO: raise error if x, y are different to self.x or self.y
        scale = kwargs[self.amplitude_param]
        var = kwargs[self.series_param]
        f_x = self._evaluate_series(var, self._f_x)
        f_y = self._evaluate_series(var, self._f_y)
        return scale * f_x, scale * f_y

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, **kwargs):  # TODO: raise error if x, y are different to self.x or self.y
        scale = kwargs[self.amplitude_param]
        var = kwargs[self.series_param]
        f_xx = self._evaluate_series(var, self._f_xx)
        f_xy = self._evaluate_series(var, self._f_xy)
        f_yy = self._evaluate_series(var, self._f_yy)
        return scale * f_xx, scale * f_xy, scale * f_xy, scale * f_yy

    @functools.partial(jit, static_argnums=(0,))
    def _evaluate_series(self, var, coefs):
        n = jnp.arange(self.order + 1)  # (n+1)
        fact = factorial(n)
        powers = jnp.power((jnp.expand_dims(var, -1) - self.series_var_0), n)  # batch, (n+1)
        return jnp.sum(coefs * powers / fact, -1)  # x, y, batch | sum along (n+1)
