import tensorflow as tf
import gigalens.profile
from abc import ABC


class MassProfile(gigalens.profile.MassProfile, ABC):
    """Tensorflow interface for a mass profile."""

    @tf.function
    def hessian(self, x, y, **kwargs):
        """Calculates hessian with autograd.

                Args:
                    x: :math:`x` coordinate at which to evaluate the deflection
                    y: :math:`y` coordinate at which to evaluate the deflection
                    **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

                Returns:
                    A tuple :math:`(\\alpha_x, \\alpha_y)` containing the deflection angle in the :math:`x` and :math:`y` directions

        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            fx, fy = self.deriv(x, y, **kwargs)

        f_xx, f_xy = tape.gradient(fx, [x, y])
        f_yx, f_yy = tape.gradient(fy, [x, y])

        return f_xx, f_xy, f_yx, f_yy

    @tf.function
    def convergence(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        kappa = (f_xx + f_yy) / 2
        return kappa

    @tf.function
    def shear(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        gamma1 = (f_xx - f_yy) / 2
        gamma2 = f_xy
        return gamma1, gamma2
