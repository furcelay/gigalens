from typing import List, Dict
from gigalens.tf.profile import MassProfile
import tensorflow as tf


class ScalingRelation(MassProfile):

    def __init__(
            self,
            profile: MassProfile,
            scaling_params: List,
            mag_star: float,
            scaling_params_power: Dict[str, float],
            galaxy_catalogue,
            mag_key='mag',
            chunk_size=None,
            **kwargs,
    ):
        self.profile = profile
        self._name = f"Scaled-{profile.name}"
        try:
            self._params = self.__getattribute__('_params')
        except AttributeError:
            self._params = scaling_params
        self.scaling_params = scaling_params
        super(ScalingRelation, self).__init__(**kwargs)

        self.mag_star = tf.constant(mag_star, dtype=tf.float32)
        self.power = {k: tf.constant(v, dtype=tf.float32) for k, v in scaling_params_power.items()}
        self.galaxy_cat = galaxy_catalogue
        lum = 10**((mag_star - self.galaxy_cat[mag_key]) / 2.5)
        self._luminosities = tf.constant(lum, dtype=tf.float32)
        self.n_galaxy = len(self._luminosities)

        if chunk_size is None:
            self.chunk_size = self.n_galaxy
        else:
            self.chunk_size = chunk_size

        try:
            constants = self.profile.__getattribute__('constants')
        except AttributeError:
            constants = []
        self.not_scaling_params = [p for p in self.profile.params + constants if p not in self.scaling_params]

        self._galaxy_constants = []
        self._unscaled_params = []
        for pos in range(0, self.n_galaxy, self.chunk_size):
            chunk = slice(pos, pos + self.chunk_size)
            self._galaxy_constants.append(
                {k: tf.constant(self.galaxy_cat[k][chunk], dtype=tf.float32)
                 for k in self.not_scaling_params}
            )
            self._unscaled_params.append(
                {k: (self._luminosities[chunk]) ** self.power[k]
                 for k in self.scaling_params}
            )

    @tf.function
    def scale_params(self, scales: Dict):
        return [{k: up[k] * tf.expand_dims(scales[k], -1) for k in self.scaling_params} for up in self._unscaled_params]

    @tf.function
    def deriv(self, x, y, **scales):
        alpha_x, alpha_y = tf.zeros_like(x), tf.zeros_like(x)
        x, y = tf.expand_dims(x, -1), tf.expand_dims(y, -1)  # (x, y, b) -> (x, y, b, g)
        scaled = self.scale_params(scales)
        for p_chunk, c_chunk in zip(scaled, self._galaxy_constants):
            alpha_x_chunk, alpha_y_chunk = self.profile.deriv(x, y, **p_chunk, **c_chunk)
            alpha_x += tf.reduce_sum(alpha_x_chunk, -1)
            alpha_y += tf.reduce_sum(alpha_y_chunk, -1)
        return alpha_x, alpha_y

    @tf.function
    def hessian(self, x, y, **scales):
        f_xx, f_xy, f_yx, f_yy = tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x)
        x, y = tf.expand_dims(x, -1), tf.expand_dims(y, -1)  # (x, y, b) -> (x, y, b, g)
        scaled = self.scale_params(scales)
        for p_chunk, c_chunk in zip(scaled, self._galaxy_constants):
            f_xx_c, f_xy_c, f_yx_c, f_yy_c = self.profile.hessian(x, y, **p_chunk, **c_chunk)
            f_xx += tf.reduce_sum(f_xx_c, -1)
            f_xy += tf.reduce_sum(f_xy_c, -1)
            f_yx += tf.reduce_sum(f_yx_c, -1)
            f_yy += tf.reduce_sum(f_yy_c, -1)
        return f_xx, f_xy, f_yx, f_yy

    @tf.function
    def convergence(self, x, y, **scales):
        kappa = tf.zeros_like(x)
        x, y = tf.expand_dims(x, -1), tf.expand_dims(y, -1)  # (x, y, b) -> (x, y, b, g)
        scaled = self.scale_params(scales)
        for p_chunk, c_chunk in zip(scaled, self._galaxy_constants):
            k = self.profile.convergence(x, y, **p_chunk, **c_chunk)
            kappa += tf.reduce_sum(k, -1)
        return kappa

    @tf.function
    def shear(self, x, y, **scales):
        gamma1, gamma2 = tf.zeros_like(x), tf.zeros_like(x)
        x, y = tf.expand_dims(x, -1), tf.expand_dims(y, -1)  # (x, y, b) -> (x, y, b, g)
        scaled = self.scale_params(scales)
        for p_chunk, c_chunk in zip(scaled, self._galaxy_constants):
            g1, g2 = self.profile.shear(x, y, **p_chunk, **c_chunk)
            gamma1 += tf.reduce_sum(g1, -1)
            gamma2 += tf.reduce_sum(g2, -1)
        return gamma1, gamma2
