from gigalens.tf.profile import MassProfile
from abc import ABC, abstractmethod
import tensorflow as tf


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
        self._constants_dict = {}
        self._f_x, self._f_y = None, None
        self._f_xx, self._f_xy, self._f_yy = None, None, None
        super(MassSeries, self).__init__(**kwargs)

    def update_models(self, x, y, params):
        self.series_var_0 = params[self.series_param]
        self._constants_dict = params.copy()
        for p in self._params:
            del self._constants_dict[p]
        self._f_x, self._f_y = self.precompute_deriv(x, y, **params)
        # self._f_xx, self._f_xy, _, self._f_yy = self.precompute_hessian(x, y, **params)

    @abstractmethod
    def precompute_deriv(self, x, y, **kwargs):
        pass

    @abstractmethod
    def precompute_hessian(self, x, y, **kwargs):
        pass

    def deriv(self, x, y, **kwargs):
        var = kwargs[self.series_param]
        scale = kwargs[self.scale_param]
        f_x = self._evaluate_series(var, self._f_x)
        f_y = self._evaluate_series(var, self._f_y)
        return scale * f_x, scale * f_y

    def hessian(self, x, y, **kwargs):
        var = kwargs[self.series_param]
        scale = kwargs[self.scale_param] / 2
        f_xx = self._evaluate_series(var, self._f_xx)
        f_xy = self._evaluate_series(var, self._f_xy)
        f_yy = self._evaluate_series(var, self._f_yy)
        return scale * f_xx, scale * f_xy, scale * f_xy, scale * f_yy

    @tf.function
    def _evaluate_series(self, var, coefs):
        n = tf.range(self.order + 1, dtype=tf.float32)  # (n+1)
        fact = tf.exp(tf.math.lgamma(n + 1))
        powers = tf.math.pow((tf.expand_dims(var, -1) - self.series_var_0), n)  # batch, (n+1)
        return tf.reduce_sum(coefs * powers / fact, -1)  # x, y, batch | sum along (n+1)


__all__ = [MassSeries]