import tensorflow as tf
from gigalens.tf.profile import MassProfile


class NFW(MassProfile):

    _name = 'NFW'
    _params = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    _r_min = 0.0000001
    _c = 0.000001

    @tf.function
    def deriv(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - tf.math.log(2.)))
        x, y = x - center_x, y - center_y
        R = tf.math.sqrt(x ** 2 + y ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0, x, y)
        return f_x, f_y

    @tf.function
    def nfwAlpha(self, R, Rs, rho0, ax_x, ax_y):

        R = tf.math.maximum(self._r_min, R)
        Rs = tf.math.maximum(self._r_min, Rs)
        x = R / Rs
        gx = self.g_(x)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    @tf.function
    def g_(self, x):
        x_shape = tf.shape(x)
        x = tf.reshape(x, (-1,))
        x = tf.math.maximum(self._c, x)
        a = tf.ones_like(x, dtype=tf.float32)
        inds1 = tf.where(x < 1)
        inds2 = tf.where(x > 1)
        x1, x2 = tf.reshape(tf.gather(x, inds1), (-1,)), tf.reshape(tf.gather(x, inds2), (-1,))
        a = tf.tensor_scatter_nd_update(
            a,
            inds1,
            tf.math.log(x1 / 2.) + 1 / tf.math.sqrt(1 - x1 ** 2) * tf.math.acosh(1. / x1)
        )
        a = tf.tensor_scatter_nd_update(
            a,
            inds2,
            tf.math.log(x2 / 2.) + 1 / tf.math.sqrt(x2 ** 2 - 1) * tf.math.acos(1. / x2)
        )
        return tf.reshape(a, x_shape)

    @tf.function
    def F_(self, x):
        # x is r/Rs
        x_shape = tf.shape(x)
        x = tf.reshape(x, (-1,))
        a = tf.ones_like(x, dtype=tf.float32) / 3  # a = 1/3 if x == 1
        inds1 = tf.where(x < 1)
        inds2 = tf.where(x > 1)
        x1, x2 = tf.reshape(tf.gather(x, inds1), (-1,)), tf.reshape(tf.gather(x, inds2), (-1,))
        a = tf.tensor_scatter_nd_update(
            a,
            inds1,
            1 / (x1 ** 2 - 1) * (
                    1 - 2 / tf.math.sqrt(1 - x1 ** 2) * tf.math.atanh(tf.math.sqrt((1 - x1) / (1 + x1)))),
        )
        a = tf.tensor_scatter_nd_update(
            a,
            inds2,
            1 / (x2 ** 2 - 1) * (
                    1 - 2 / tf.math.sqrt(x2 ** 2 - 1) * tf.math.atan(tf.math.sqrt((x2 - 1) / (1 + x2)))
            ),
        )
        return tf.reshape(a, x_shape)

    @tf.function
    def hessian(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - tf.math.log(2.)))
        Rs = tf.math.maximum(self._r_min, Rs)
        x, y = x - center_x, y - center_y
        R = tf.math.sqrt(x ** 2 + y ** 2)
        R = tf.math.maximum(self._c, R)
        X = R / Rs
        gx = self.g_(X)
        Fx = self.F_(X)
        kappa = 2 * rho0 * Rs * Fx
        a = 2 * rho0 * Rs * (2 * gx / X ** 2 - Fx)
        gamma1, gamma2 = a * (y ** 2 - x ** 2) / R ** 2, -a * 2 * (x * y) / R ** 2
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy


class NFW_ELLIPSE(MassProfile):

    _name = 'NFW_ELLIPSE'
    _params = ['Rs', 'alpha_Rs', 'e1', 'e2', 'center_x', 'center_y']

    def __init__(self):
        self.nfw = NFW()
        super(NFW_ELLIPSE, self).__init__()

    @tf.function
    def deriv(self, x, y, Rs, alpha_Rs, e1, e2, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - tf.math.log(2.)))
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * tf.math.sqrt(1 - e), y * tf.math.sqrt(1 + e)
        R = tf.math.sqrt(x ** 2 + y ** 2)

        fx, fy = self.nfw.nfwAlpha(R, Rs, rho0, x, y)
        fx *= tf.math.sqrt(1 - e)
        fy *= tf.math.sqrt(1 + e)
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @tf.function
    def _rotate(self, x, y, phi):
        cos_phi = tf.cos(phi, name=self.name + "rotate-cos")
        sin_phi = tf.sin(phi, name=self.name + "rotate-sin")
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @tf.function
    def _param_conv(self, e1, e2):
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.minimum(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        e = tf.math.abs(1 - q ** 2) / (1 + q ** 2)
        return e, phi
