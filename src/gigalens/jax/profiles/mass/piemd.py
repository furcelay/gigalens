import functools

import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile

"""
Dual Pseudo Isothermal Elliptical Mass Profile (dPIEMS or dPIE)
Composed of two PIE from Kassiola & Kovner (1993) similar (but not equal) to Elíasdóttir (2007)

Complex deflection: J = alpha_x + i alpha_y = theta_E * r_cut / (r_cut - r_core) * (I_cut - I_core)
with I_w from Kassiola & Kovner (1993) eq. 4.1.2 replacing w by r_core or r_cut

Convergence: k = theta_E * r_cut / (r_cut - r_core) * (1 / sqrt(r^2 + r_core^2) - 1 / sqrt(r^2 + r_cut^2))
Central convergence: k0 = theta_E / (2 r_core)

Velocity dispersion equivalence:
Lenstool:           theta_E = 6 pi (D_LS / D_S) (simga_v(Ls) / c)^2
Limousin (2005):    theta_E = 4 pi (D_LS / D_S) (simga_v(Li) / c)^2
Elíasdóttir (2007): theta_E = 6 pi (D_LS / D_S) (simga_v(El) / c)^2 * r_cut^2 / (r_cut^2 - r_core^2)
"""


class DPIS(MassProfile):
    """
    Dual Pseudo Isothermal Sphere
    """

    _name = "dPIS"
    _params = ['theta_E', 'r_core', 'r_cut', 'center_x', 'center_y']
    _r_min = 0.0001

    def __init__(self,):
        super(DPIS, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, r_core, r_cut, center_x, center_y):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r2 = x ** 2 + y ** 2  # r2 instead of dividing by r twice
        scale = theta_E * r_cut / (r_cut - r_core)
        alpha_r = scale / r2 * self._f_A20(r2, r_core, r_cut)
        f_x = alpha_r * x
        f_y = alpha_r * y
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0,))
    def _f_A20(self, r2, r_core, r_cut):
        """
        Faster and equiv to equation A20 * r in Eliasdottir (2007), (see Golse PhD)
        """
        return jnp.sqrt(r2 + r_core ** 2) - r_core - jnp.sqrt(r2 + r_cut ** 2) + r_cut

    @functools.partial(jit, static_argnums=(0,))
    def _sort_ra_rs(self, r_core, r_cut):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        r_core = jnp.where(r_core < r_cut, r_core, r_cut)
        r_cut = jnp.where(r_core > r_cut, r_core, r_cut)
        r_core = jnp.maximum(self._r_min, r_core)
        r_cut = jnp.where(r_cut > r_core + self._r_min, r_cut, r_cut + self._r_min)
        return r_core, r_cut

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, theta_E, r_core, r_cut, center_x, center_y):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r = jnp.sqrt(x ** 2 + y ** 2)
        r = jnp.maximum(self._r_min, r)
        scale = theta_E * r_cut / (r_cut - r_core)
        gamma = scale / 2 * (
                2 * (1. / (r_core + jnp.sqrt(r_core ** 2 + r ** 2))
                     - 1. / (r_cut + jnp.sqrt(r_cut ** 2 + r ** 2))) -
                (1 / jnp.sqrt(r_core ** 2 + r ** 2) - 1 / jnp.sqrt(r_cut ** 2 + r ** 2)))
        kappa = scale / 2 * (r_core + r_cut) / r_cut * (
                1 / jnp.sqrt(r_core ** 2 + r ** 2) - 1 / jnp.sqrt(r_cut ** 2 + r ** 2))
        sin_imphi = -2 * x * y / r ** 2
        cos_imphi = (y ** 2 - x ** 2) / r ** 2
        gamma1 = cos_imphi * gamma
        gamma2 = sin_imphi * gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, theta_E, r_core, r_cut, center_x=0, center_y=0):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r = jnp.sqrt(x ** 2 + y ** 2)
        r = jnp.maximum(self._r_min, r)
        scale = theta_E * r_cut / (r_cut - r_core)
        kappa = scale / 2 * (r_core + r_cut) / r_cut * (
                1 / jnp.sqrt(r_core ** 2 + r ** 2) - 1 / jnp.sqrt(r_cut ** 2 + r ** 2))
        return kappa


class DPIE(MassProfile):
    _name = "dPIE"
    _params = ['theta_E', 'r_core', 'r_cut', 'center_x', 'center_y', 'e1', 'e2']
    _r_min = 0.0001

    def __init__(self):
        super(DPIE, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, r_core, r_cut, e1, e2, center_x=0, center_y=0):
        """
        Same as Lenstool implementation of Kassiola & Kovner, 1993 PIEMD, paragraph 4.1
        but with r_cut = s from Eliasdottir (2007)
        """

        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        scale = theta_E * r_cut / (r_cut - r_core)
        alpha_x, alpha_y = self._complex_deriv_dual(x, y, r_core, r_cut, e, q)
        alpha_x, alpha_y = self._rotate(alpha_x, alpha_y, -phi)
        return scale * alpha_x, scale * alpha_y

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, theta_E, r_core, r_cut, e1, e2, center_x=0, center_y=0):
        """
        Same as Lenstool implementation of Kassiola & Kovner, 1993 PIEMD, paragraph 4.1
        but with r_cut = s from Eliasdottir (2007)
        """
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        scale = theta_E * r_cut / (r_cut - r_core)
        f_xx_core, f_xy_core, f_yy_core = self._complex_hessian_single(x, y, r_core, e, q)
        f_xx_cut, f_xy_cut, f_yy_cut = self._complex_hessian_single(x, y, r_cut, e, q)
        f_xx = scale * (f_xx_core - f_xx_cut)
        f_xy = f_yx = scale * (f_xy_core - f_xy_cut)
        f_yy = scale * (f_yy_core - f_yy_cut)
        f_xx, f_xy, f_yx, f_yy = self._hessian_rotate(f_xx, f_xy, f_yx, f_yy, -phi)
        return f_xx, f_xy, f_yx, f_yy

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, theta_E, r_core, r_cut, e1, e2, center_x=0, center_y=0):
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        scale = theta_E * r_cut / (r_cut - r_core)
        rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2
        kappa = scale / 2 * (1 / jnp.sqrt(rem2 + r_core ** 2) - 1 / jnp.sqrt(rem2 + r_cut ** 2))
        return kappa

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _hessian_rotate(self, f_xx, f_xy, f_yx, f_yy, phi):
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
        f_xy_rot = f_yx_rot = d - e
        f_yy_rot = a - b - c
        return f_xx_rot, f_xy_rot, f_yx_rot, f_yy_rot

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.arctan2(e2, e1) / 2
        e = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - e) / (1 + e)
        return e, q, phi

    @functools.partial(jit, static_argnums=(0,))
    def _sort_ra_rs(self, r_core, r_cut):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        r_core = jnp.where(r_core < r_cut, r_core, r_cut)
        r_cut = jnp.where(r_core > r_cut, r_core, r_cut)
        r_core = jnp.maximum(self._r_min, r_core)
        r_cut = jnp.where(r_cut > r_core + self._r_min, r_cut, r_cut + self._r_min)
        return r_core, r_cut

    @functools.partial(jit, static_argnums=(0,))
    def _complex_deriv_dual(self, x, y, r_core, r_cut, e, q):
        sqe = jnp.sqrt(e)
        rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2

        z_frac_re, z_frac_im = self._optimal_complex_divide_double(
            q * x,
            2. * sqe * jnp.sqrt(r_core ** 2 + rem2) - y / q,
            x,
            2. * r_core * sqe - y,
            2. * sqe * jnp.sqrt(r_cut ** 2 + rem2) - y / q,
            2. * r_cut * sqe - y
        )
        z_r_re, z_r_im = self._complex_log(z_frac_re, z_frac_im)
        scale = -0.5 * (1. - e ** 2) / sqe
        # f_x = Re(scale * z_r * i)
        # f_y = Im(scale * z_r * i)
        return - scale * z_r_im, scale * z_r_re

    @functools.partial(jit, static_argnums=(0,))
    def _complex_hessian_single(self, x, y, r_w, e, q):
        """
        I = (f_x + f_y * i)
        with I in Eq 4.1.2 in Kassiola & Kovner
        I = A * ln(u / v)) --> I' = A * (u'/u - v'/v)

        A = scale * i
        u = cx * x + (-cy * y + 2 * sqe * wrem) * i
        v = x + (-y + 2 * r_w * sqe) *i

        du_dx = cx + 2 * sqe * x / (cx * wrem) * i
        du_dy = (-cy + 2 * sqe * y / (cy * wrem)) * i

        dv_dx = 1
        dv_dy = -i

        f_xx = Re(dI_dx)
        f_xy = f_yx = Re(dI_dy) = Im(dI_dx)
        f_yy = Im(dI_dy)

        simplifying we get to:
        """

        sqe = jnp.sqrt(e)
        qinv = 1. / q
        cx = (1. + e) * (1. + e)
        cy = (1. - e) * (1. - e)
        scale = 0.5 * (1. - e ** 2) / sqe
        rem2 = x ** 2 / cx + y ** 2 / cy
        wrem = jnp.sqrt(r_w ** 2 + rem2)

        u2 = q ** 2 * x ** 2 + (2. * sqe * wrem - y * qinv) ** 2  # |u|**2
        v_im = 2. * r_w * sqe - y
        v2 = x ** 2 + v_im ** 2  # |v|**2

        f_xx = scale * (q * (2. * sqe * x ** 2 / cx / wrem - 2. * sqe * wrem + y * qinv) / u2 + v_im / v2)
        f_xy = scale * ((2 * sqe * x * y * q / cy / wrem - x) / u2 + x / v2)
        f_yy = scale * ((2 * sqe * wrem * qinv - y * qinv ** 2 - 4 * e * y / cy +
                         2 * sqe * y ** 2 / cy / wrem * qinv) / u2 - v_im / v2)
        return f_xx, f_xy, f_yy


    @staticmethod
    @jit
    def _complex_divide(a, b, c, d):  # TODO: check if builtin complex functions are faster
        """
        z = (a + b * i) / (c + d * i)
        """
        z_den_norm2 = c ** 2 + d ** 2
        return (a * c + b * d) / z_den_norm2, (b * c - a * d) / z_den_norm2

    @functools.partial(jit, static_argnums=(0,))
    def _optimal_complex_divide_double(self, a, b, c, d, e, f):
        """
        z = ((a + b * i) / (c + d * i)) / ((a + e * i) / (c + f * i))
          = ((a * c - b * f) + (a * f  + b * c) * i) / ((a * c - d * e) + (a * d + c * e) * i)
        """
        return self._complex_divide(a * c - b * f, a * f + b * c,
                                    a * c - d * e, a * d + c * e)

    @staticmethod
    @jit
    def _complex_log(a, b):
        """
        z = log(a + b * i)
        """
        norm2 = a ** 2 + b ** 2
        z_re = jnp.log(jnp.sqrt(norm2))
        z_im = jnp.arctan2(b, a)
        return z_re, z_im
