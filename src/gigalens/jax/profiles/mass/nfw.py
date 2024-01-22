import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile


class NFW(MassProfile):

    _name = 'NFW'
    _params = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    _r_min = 0.0000001
    _c = 0.000001

    def __init__(self):
        super(NFW, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        x, y = x - center_x, y - center_y
        R = jnp.sqrt(x ** 2 + y ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0, x, y)
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0,))
    def nfwAlpha(self, R, Rs, rho0, ax_x, ax_y):

        R = jnp.maximum(self._r_min, R)
        Rs = jnp.maximum(self._r_min, Rs)
        x = R / Rs
        gx = self.g_(x)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    @functools.partial(jit, static_argnums=(0,))
    def g_(self, x):
        x_shape = jnp.shape(x)
        x = jnp.reshape(x, (-1,))
        x = jnp.maximum(self._c, x)
        a = jnp.ones_like(x)
        inds1 = jnp.where(x < 1)
        inds2 = jnp.where(x > 1)
        x1, x2 = jnp.reshape(x[inds1], (-1,)), jnp.reshape(x[inds2], (-1,))
        a = a.at[inds1].set(jnp.log(x1 / 2.) + 1 / jnp.sqrt(1 - x1 ** 2) * jnp.acosh(1. / x1))
        a = a.at[inds2].set(jnp.log(x2 / 2.) + 1 / jnp.sqrt(x2 ** 2 - 1) * jnp.acos(1. / x2))
        return jnp.reshape(a, x_shape)

    @functools.partial(jit, static_argnums=(0,))
    def F_(self, x):
        # x is r/Rs
        x_shape = jnp.shape(x)
        x = jnp.reshape(x, (-1,))
        a = jnp.ones_like(x) / 3  # a = 1/3 if x == 1
        inds1 = jnp.where(x < 1)
        inds2 = jnp.where(x > 1)
        x1, x2 = jnp.reshape(x[inds1], (-1,)), jnp.reshape(x[inds2], (-1,))
        a = a.at[inds1].set(1 / (x1 ** 2 - 1) * (
                1 - 2 / jnp.sqrt(1 - x1 ** 2) * jnp.atanh(jnp.sqrt((1 - x1) / (1 + x1)))),
        )
        a = a.at[inds2].set(1 / (x2 ** 2 - 1) * (
                1 - 2 / jnp.sqrt(x2 ** 2 - 1) * jnp.atan(jnp.sqrt((x2 - 1) / (1 + x2)))
            ),
        )
        return jnp.reshape(a, x_shape)

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, Rs, alpha_Rs, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        Rs = jnp.maximum(self._r_min, Rs)
        x, y = x - center_x, y - center_y
        R = jnp.sqrt(x ** 2 + y ** 2)
        R = jnp.maximum(self._c, R)
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

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, e1, e2, center_x, center_y):
        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. - jnp.log(2.)))
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * jnp.sqrt(1 - e), y * jnp.sqrt(1 + e)
        R = jnp.sqrt(x ** 2 + y ** 2)

        fx, fy = self.nfw.nfwAlpha(R, Rs, rho0, x, y)
        fx *= jnp.sqrt(1 - e)
        fy *= jnp.sqrt(1 + e)
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.atan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        e = jnp.abs(1 - q ** 2) / (1 + q ** 2)
        return e, phi
