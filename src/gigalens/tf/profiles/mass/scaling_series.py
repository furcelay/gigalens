from typing import List
from gigalens.tf.profile import MassProfile
from gigalens.tf.series.series_profile import MassSeries
from gigalens.tf.profiles.mass.scaling_relation import ScalingRelation
import tensorflow as tf


class ScalingRelationSeries(MassSeries, ScalingRelation):
    _constants: List[str]

    def __init__(self, profile: MassSeries, **kwargs):
        self._series_param = profile.series_param
        self._scale_param = profile.scale_param
        super(ScalingRelationSeries, self).__init__(profile=profile, **kwargs)

        self.params = self._params = self.profile.params
        self.scaling_constants = [p for p in self.scaling_params if p in self.constants]

    def precompute_deriv(self, x, y, **scales):
        out_shape = tf.concat([x.shape, [self.order + 1]], 0)
        f_x, f_y = tf.zeros(out_shape), tf.zeros(out_shape)
        scaled = self.scale_params(scales)
        n = tf.range(self.order + 1, dtype=tf.float32)
        for s_chunk, u_chunk, c_chunk in zip(scaled, self._unscaled_params, self._galaxy_constants):
            scale_factor = tf.expand_dims(u_chunk[self.scale_param], 0)
            scale_factor = tf.expand_dims(scale_factor, -1)  # 1, chunk, 1
            series_factor = tf.expand_dims(u_chunk[self.series_param], 0)
            series_factor = tf.expand_dims(series_factor, -1)
            series_factor = tf.pow(series_factor, n)  # 1, chunk, n + 1
            pre_factor = scale_factor * series_factor
            f_x_chunk, f_y_chunk = self.profile.precompute_deriv(x, y, **s_chunk, **c_chunk)
            f_x += tf.reduce_sum(pre_factor * f_x_chunk, -2, keepdims=True)
            f_y += tf.reduce_sum(pre_factor * f_y_chunk, -2, keepdims=True)
        return f_x, f_y

    def precompute_hessian(self, x, y, **scales):
        out_shape = tf.concat([x.shape, [self.order + 1]], 0)
        f_xx, f_xy, f_yy = tf.zeros(out_shape), tf.zeros(out_shape), tf.zeros(out_shape)
        scaled = self.scale_params(scales)
        n = tf.range(self.order + 1, dtype=tf.float32)
        for s_chunk, u_chunk, c_chunk in zip(scaled, self._unscaled_params, self._galaxy_constants):
            scale_factor = tf.expand_dims(u_chunk[self.scale_param], 0)
            scale_factor = tf.expand_dims(scale_factor, -1)  # 1, chunk, 1
            series_factor = tf.expand_dims(u_chunk[self.series_param], 0)
            series_factor = tf.expand_dims(series_factor, -1)
            series_factor = tf.pow(series_factor, n)  # 1, chunk, n + 1
            pre_factor = scale_factor * series_factor
            f_xx_chunk, f_xy_chunk, f_yy_chunk = self.profile.precompute_hessian(x, y, **s_chunk, **c_chunk)
            f_xx += tf.reduce_sum(pre_factor * f_xx_chunk, -2, keepdims=True)
            f_xy += tf.reduce_sum(pre_factor * f_xy_chunk, -2, keepdims=True)
            f_yy += tf.reduce_sum(pre_factor * f_yy_chunk, -2, keepdims=True)
        return f_xx, f_xy, f_xy, f_yy

    def convergence(self, x, y, **kwargs):
        return MassProfile.convergence(self, x, y, **kwargs)

    def shear(self, x, y, **kwargs):
        return MassProfile.shear(self, x, y, **kwargs)

