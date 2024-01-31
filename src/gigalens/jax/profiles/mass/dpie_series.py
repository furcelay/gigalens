import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.series.profiles.dpie import deriv_fns, hessian_fns
from gigalens.jax.series.series_profile import MassSeries


class DPIESeries(MassSeries):
    """
    Series expansion of the given potential on a single variable
    """
    _params = ['r_cut', 'theta_E']
    _constants = ['r_core', 'center_x', 'center_y', 'e1', 'e2']
    _series_param = 'r_cut'
    _amplitude_param = 'theta_E'
    _name = "SeriesExpansion-dPIE"

    def __init__(self, order=3):
        super(DPIESeries, self).__init__(order=order)

    @functools.partial(jit, static_argnums=(0, 1))
    def precompute_deriv(self, order, x, y, theta_E, r_core, r_cut, e1, e2, center_x, center_y):
        # theta_E is not used
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        f_x, f_y = [], []
        for i in range(order + 1):
            f_x_i, f_y_i = deriv_fns[i](x, y, e, r_core, r_cut)  # x, y, batch
            f_x_i, f_y_i = self._rotate(f_x_i, f_y_i, -phi)
            f_x.append(f_x_i)
            f_y.append(f_y_i)
        f_x, f_y = jnp.stack(f_x, axis=-1), jnp.stack(f_y, axis=-1)  # x, y, batch, (n+1)
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0, 1))
    def precompute_hessian(self, order, x, y, theta_E, r_core, r_cut, e1, e2, center_x, center_y):
        # theta_E is not used
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        f_xx, f_xy, f_yy = [], [], []
        for i in range(order + 1):
            f_xx_i, f_xy_i, _, f_yy_i = hessian_fns[i](x, y, e, r_core, r_cut)  # x, y, batch
            f_xx_i, f_xy_i, f_yy_i = self._hessian_rotate(f_xx_i, f_xy_i, f_yy_i, -phi)
            f_xx.append(f_xx_i)
            f_xy.append(f_xy_i)
            f_yy.append(f_yy_i)
        # stack: x, y, batch, (n+1)
        f_xx, f_xy, f_yy = jnp.stack(f_xx, axis=-1), jnp.stack(f_xy, axis=-1), jnp.stack(f_yy, axis=-1)
        return f_xx, f_xy, f_yy

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.arctan2(e2, e1) / 2
        e = jnp.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - e) / (1 + e)
        return e, q, phi

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _hessian_rotate(self, f_xx, f_xy, f_yy, phi):
        """
         rotation matrix
         R = | cos(t) -sin(t) |
             | sin(t)  cos(t) |

        Hessian matrix
        H = | fxx  fxy |
            | fxy  fyy |

        returns R H R^T

        """
        cos_2phi = jnp.cos(2 * phi)
        sin_2phi = jnp.sin(2 * phi)
        a = 1 / 2 * (f_xx + f_yy)
        b = 1 / 2 * (f_xx - f_yy) * cos_2phi
        c = f_xy * sin_2phi
        d = f_xy * cos_2phi
        e = 1 / 2 * (f_xx - f_yy) * sin_2phi
        f_xx_rot = a + b + c
        f_xy_rot = d - e
        f_yy_rot = a - b - c
        return f_xx_rot, f_xy_rot, f_yy_rot
