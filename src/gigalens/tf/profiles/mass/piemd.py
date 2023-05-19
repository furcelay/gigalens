import tensorflow as tf
from gigalens.tf.profile import MassProfile


class DPIS(MassProfile):
    """
    Dual Pseudo Isothermal Sphere
    """

    _name = "dPIS"
    _params = ['sigma0', 'Ra', 'Rs', 'center_x', 'center_y']
    _r_min = 0.0001

    @tf.function
    def deriv(self, x, y, sigma0, Ra, Rs, center_x, center_y):
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x, y = x - center_x, y - center_y
        r = tf.math.sqrt(x**2 + y**2)
        r = tf.math.maximum(self._r_min, r)
        alpha_r = 2*sigma0 * Ra * Rs / (Rs - Ra) * self._f_A20(r / Ra, r / Rs)
        f_x = alpha_r * x/r
        f_y = alpha_r * y/r
        return f_x, f_y

    @tf.function
    def _f_A20(self, r_a, r_s):
        """
        equation A20 in Eliasdottir (2007)
        """
        return r_a/(1 + tf.math.sqrt(1 + r_a**2)) - r_s/(1 + tf.math.sqrt(1 + r_s**2))

    @tf.function
    def _sort_ra_rs(self, Ra, Rs):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        Ra = tf.where(Ra < Rs, Ra, Rs)
        Rs = tf.where(Ra > Rs, Ra, Rs)
        Ra = tf.math.maximum(self._r_min, Ra)
        Rs = tf.where(Rs > Ra + self._r_min, Rs, Rs + self._r_min)
        return Ra, Rs

    @tf.function
    def hessian(self, x, y, sigma0, Ra, Rs, center_x, center_y):
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x, y = x - center_x, y - center_y
        r = tf.math.sqrt(x ** 2 + y ** 2)
        r = tf.math.maximum(self._r_min, r)

        gamma = sigma0 * Ra * Rs / (Rs - Ra) * (
                    2 * (1. / (Ra + tf.math.sqrt(Ra ** 2 + r ** 2)) - 1. / (Rs + tf.math.sqrt(Rs ** 2 + r ** 2))) -
                    (1 / tf.math.sqrt(Ra ** 2 + r ** 2) - 1 / tf.math.sqrt(Rs ** 2 + r ** 2)))
        kappa = sigma0 * Ra * Rs / (Rs - Ra) * (1 / tf.math.sqrt(Ra ** 2 + r ** 2) - 1 / tf.math.sqrt(Rs ** 2 + r ** 2))
        sin_2phi = -2 * x * y / r ** 2
        cos_2phi = (y ** 2 - x ** 2) / r ** 2
        gamma1 = cos_2phi * gamma
        gamma2 = sin_2phi * gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy


class DPIE(MassProfile):
    _name = "dPIE"
    _params = ['sigma0', 'Ra', 'Rs', 'center_x', 'center_y', 'e1', 'e2']

    def __init__(self):
        self.spherical = DPIS()
        self._r_min = 0.0001
        super(DPIE, self).__init__()

    @tf.function
    def deriv(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * tf.math.sqrt(1 - e), y * tf.math.sqrt(1 + e)

        fx, fy = self.spherical.deriv(x, y, sigma0, Ra, Rs, center_x=0, center_y=0)
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
