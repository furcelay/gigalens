import tensorflow as tf
from gigalens.tf.profile import MassProfile


class Shear(MassProfile):
    """External shear model, parameterized by shear components ``gamma1`` and ``gamma2``."""

    _name = "SHEAR"
    _params = ["gamma1", "gamma2"]

    @tf.function
    def deriv(self, x, y, gamma1, gamma2):
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y

    def hessian(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0.
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy
