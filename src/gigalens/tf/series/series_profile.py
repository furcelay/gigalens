from gigalens.tf.profile import MassProfile
from abc import ABC, abstractmethod
import tensorflow as tf


__all__ = ['MassSeries']


class MassSeries(MassProfile, ABC):
    """
    Series expansion of the given potential on a single variable
    """

    _series_param: str
    _scale_param: str
    _name = "SeriesExpansion"
    _constants = []

    def __init__(self, order=3, **kwargs):
        self.series_param = self._series_param
        self.scale_param = self._scale_param
        self.series_var_0 = tf.constant(0, dtype=tf.float32)
        self.constants = self._constants
        self.order = order
        self.constants_dict = {}
        self.deriv_cache = {}
        self.hessian_cache = {}
        super(MassSeries, self).__init__(**kwargs)

    def set_constants(self, params):
        self.series_var_0 = params[self.series_param]
        self.constants_dict = params.copy()
        self.clear_cache()

    @abstractmethod
    def precompute_deriv(self, x, y, **kwargs):
        pass

    @abstractmethod
    def precompute_hessian(self, x, y, **kwargs):
        pass

    def deriv(self, x, y, **kwargs):
        var = kwargs[self.series_param]
        scale = kwargs[self.scale_param]
        f_x_series, f_y_series = self.get_cached_deriv(x, y)
        f_x = self._evaluate_series(var, f_x_series)
        f_y = self._evaluate_series(var, f_y_series)
        return scale * f_x, scale * f_y

    def hessian(self, x, y, **kwargs):
        var = kwargs[self.series_param]
        scale = kwargs[self.scale_param]
        f_xx_series, f_xy_series, f_yy_series = self.get_cached_hessian(x, y)
        f_xx = self._evaluate_series(var, f_xx_series)
        f_xy = self._evaluate_series(var, f_xy_series)
        f_yy = self._evaluate_series(var, f_yy_series)
        return scale * f_xx, scale * f_xy, scale * f_xy, scale * f_yy

    @tf.function
    def _evaluate_series(self, var, coefs):
        n = tf.range(self.order + 1, dtype=tf.float32)  # (n+1)
        fact = tf.exp(tf.math.lgamma(n + 1))
        powers = tf.math.pow((tf.expand_dims(var, -1) - self.series_var_0), n)  # batch, (n+1)
        return tf.reduce_sum(coefs * powers / fact, -1)  # x, y, batch | sum along (n+1)

    def get_cached_deriv(self, x, y):
        ref = hash((x.ref(), y.ref()))
        if ref in self.deriv_cache:
            return self.deriv_cache[ref]
        else:
            value = self.precompute_deriv(x, y, **self.constants_dict)
            self.deriv_cache[ref] = value
            return value

    def get_cached_hessian(self, x, y):
        ref = hash((x.ref(), y.ref()))
        if ref in self.hessian_cache:
            # use precomputed value
            # must be always the same tensors (not only the same values)
            return self.hessian_cache[ref]
        else:
            # compute and save in chache
            value = self.precompute_hessian(x, y, **self.constants_dict)
            self.hessian_cache[ref] = value
            return value

    def clear_cache(self):
        self.deriv_cache = {}
        self.hessian_cache = {}
