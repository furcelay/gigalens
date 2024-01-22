import functools

from jax import jit

from gigalens.jax.profile import MassProfile


class Shear(MassProfile):
    _name = "SHEAR"
    _params = ["gamma1", "gamma2"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, gamma1, gamma2):
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0.
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy