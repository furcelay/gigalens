from gigalens.tf.profile import MassProfile
from abc import ABC, abstractmethod
import tensorflow as tf


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
        self._f_x, self._f_y = self.precompute_deriv(self.order, self.x, self.y, **self.constants_dict)

    def set_hessian(self):
        self._f_xx, self._f_xy, self._f_yy = self.precompute_hessian(self.order, self.x, self.y, **self.constants_dict)

    @abstractmethod
    def precompute_deriv(self, order, x, y, **kwargs):
        pass

    @abstractmethod
    def precompute_hessian(self, order, x, y, **kwargs):
        pass

    def deriv(self, x, y, **kwargs):
        scale = kwargs[self.amplitude_param]
        constants = {k: v for k, v in self.constants_dict.items() if k not in kwargs}
        cond = tf.math.logical_and(tf.math.reduce_all(x == self.x),
                                   tf.math.reduce_all(y == self.y))
        f_x, f_y = tf.cond(cond,
                           lambda: self._get_deriv(**kwargs),
                           lambda: self._compute_deriv(x, y, **kwargs))
        return scale * f_x, scale * f_y

    def hessian(self, x, y, **kwargs):
        scale = kwargs[self.amplitude_param]
        constants = {k: v for k, v in self.constants_dict.items() if k not in kwargs}
        cond = tf.math.logical_and(tf.math.reduce_all(x == self.x),
                                   tf.math.reduce_all(y == self.y))
        f_xx, f_xy, f_yy = tf.cond(cond,
                                   lambda: self._get_hessian(**kwargs),
                                   lambda: self._compute_hessian(x, y, **kwargs))
        return scale * f_xx, scale * f_xy, scale * f_xy, scale * f_yy

    @tf.function
    def _evaluate_series(self, var, coefs):
        n = tf.range(self.order + 1, dtype=tf.float32)  # (n+1)
        fact = tf.exp(tf.math.lgamma(n + 1))
        powers = tf.math.pow((tf.expand_dims(var, -1) - self.series_var_0), n)  # batch, (n+1)
        return tf.reduce_sum(coefs * powers / fact, -1)  # x, y, batch | sum along (n+1)

    def _get_deriv(self, **kwargs):
        var = kwargs[self.series_param]
        f_x = self._evaluate_series(var, self._f_x)
        f_y = self._evaluate_series(var, self._f_y)
        return f_x, f_y

    def _get_hessian(self, **kwargs):
        var = kwargs[self.series_param]
        f_x = self._evaluate_series(var, self._f_x)
        f_y = self._evaluate_series(var, self._f_y)
        return f_x, f_y

    def _compute_deriv(self, x, y, **kwargs):
        # precomputing_warn()
        constants = {k: v for k, v in self.constants_dict.items() if k not in kwargs}
        f_x, f_y = self.precompute_deriv(0, x, y, **kwargs, **constants)
        return f_x[..., 0], f_y[..., 0]

    def _compute_hessian(self, x, y, **kwargs):
        # precomputing_warn()
        constants = {k: v for k, v in self.constants_dict.items() if k not in kwargs}
        f_xx, f_xy, f_yy = self.precompute_hessian(0, x, y, **kwargs, **constants)
        return f_xx[..., 0], f_xy[..., 0], f_yy[..., 0]
