import tensorflow as tf
from gigalens.tf.profile import MassProfile

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

    @tf.function
    def deriv(self, x, y, theta_E, r_core, r_cut, center_x, center_y):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r2 = x ** 2 + y ** 2  # r2 instead of dividing by r twice
        scale = theta_E * r_cut / (r_cut - r_core)
        alpha_r = scale / r2 * self._f_A20(r2, r_core, r_cut)
        f_x = alpha_r * x
        f_y = alpha_r * y
        return f_x, f_y

    @tf.function
    def _f_A20(self, r2, r_core, r_cut):
        """
        Faster and equiv to equation A20 * r in Eliasdottir (2007), (see Golse PhD)
        """
        return tf.math.sqrt(r2 + r_core ** 2) - r_core - tf.math.sqrt(r2 + r_cut ** 2) + r_cut

    @tf.function
    def _sort_ra_rs(self, r_core, r_cut):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        r_core = tf.where(r_core < r_cut, r_core, r_cut)
        r_cut = tf.where(r_core > r_cut, r_core, r_cut)
        r_core = tf.math.maximum(self._r_min, r_core)
        r_cut = tf.where(r_cut > r_core + self._r_min, r_cut, r_cut + self._r_min)
        return r_core, r_cut

    @tf.function
    def hessian(self, x, y, theta_E, r_core, r_cut, center_x, center_y):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r = tf.math.sqrt(x ** 2 + y ** 2)
        r = tf.math.maximum(self._r_min, r)
        scale = theta_E * r_cut / (r_cut - r_core)
        gamma = scale / 2 * (
                2 * (1. / (r_core + tf.math.sqrt(r_core ** 2 + r ** 2))
                     - 1. / (r_cut + tf.math.sqrt(r_cut ** 2 + r ** 2))) -
                (1 / tf.math.sqrt(r_core ** 2 + r ** 2) - 1 / tf.math.sqrt(r_cut ** 2 + r ** 2)))
        kappa = scale / 2 * (r_core + r_cut) / r_cut * (
                1 / tf.math.sqrt(r_core ** 2 + r ** 2) - 1 / tf.math.sqrt(r_cut ** 2 + r ** 2))
        sin_imphi = -2 * x * y / r ** 2
        cos_imphi = (y ** 2 - x ** 2) / r ** 2
        gamma1 = cos_imphi * gamma
        gamma2 = sin_imphi * gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @tf.function
    def convergence(self, x, y, theta_E, r_core, r_cut, center_x=0, center_y=0):
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        x, y = x - center_x, y - center_y
        r = tf.math.sqrt(x ** 2 + y ** 2)
        r = tf.math.maximum(self._r_min, r)
        scale = theta_E * r_cut / (r_cut - r_core)
        kappa = scale / 2 * (r_core + r_cut) / r_cut * (
                1 / tf.math.sqrt(r_core ** 2 + r ** 2) - 1 / tf.math.sqrt(r_cut ** 2 + r ** 2))
        return kappa


class DPIE(MassProfile):
    _name = "dPIE"
    _params = ['theta_E', 'r_core', 'r_cut', 'center_x', 'center_y', 'e1', 'e2']
    _r_min = 0.0001

    def __init__(self):
        super(DPIE, self).__init__()

    @tf.function
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
        alpha_x, alpha_y = self.complex_deriv_dual(x, y, r_core, r_cut, e, q)
        alpha_x, alpha_y = self._rotate(alpha_x, alpha_y, -phi)
        return scale * alpha_x, scale * alpha_y

    @tf.function
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
        f_xx_core, f_xy_core, f_yy_core = self.complex_hessian_single(x, y, r_core, e, q)
        f_xx_cut, f_xy_cut, f_yy_cut = self.complex_hessian_single(x, y, r_cut, e, q)
        f_xx = scale * (f_xx_core - f_xx_cut)
        f_xy = f_yx = scale * (f_xy_core - f_xy_cut)
        f_yy = scale * (f_yy_core - f_yy_cut)
        f_xx, f_xy, f_yx, f_yy = self._hessian_rotate(f_xx, f_xy, f_yx, f_yy, -phi)
        return f_xx, f_xy, f_yx, f_yy

    @tf.function
    def convergence(self, x, y, E0, r_core, r_cut, e1, e2, center_x=0, center_y=0):
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        r_core, r_cut = self._sort_ra_rs(r_core, r_cut)
        scale = E0 * r_cut / (r_cut - r_core)
        rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2
        kappa = scale / 2 * (1 / tf.math.sqrt(rem2 + r_core ** 2) - 1 / tf.math.sqrt(rem2 + r_cut ** 2))
        return kappa

    @tf.function
    def _rotate(self, x, y, phi):
        cos_phi = tf.cos(phi, name=self.name + "rotate-cos")
        sin_phi = tf.sin(phi, name=self.name + "rotate-sin")
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @tf.function
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
        cos_2phi = tf.cos(2 * phi, name=self.name + "rotate-cos")
        sin_2phi = tf.sin(2 * phi, name=self.name + "rotate-sin")
        a = 1 / 2 * (f_xx + f_yy)
        b = 1 / 2 * (f_xx - f_yy) * cos_2phi
        c = f_xy * sin_2phi
        d = f_xy * cos_2phi
        e = 1 / 2 * (f_xx - f_yy) * sin_2phi
        f_xx_rot = a + b + c
        f_xy_rot = f_yx_rot = d - e
        f_yy_rot = a - b - c
        return f_xx_rot, f_xy_rot, f_yx_rot, f_yy_rot

    @tf.function
    def _param_conv(self, e1, e2):
        phi = tf.atan2(e2, e1) / 2
        e = tf.math.minimum(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - e) / (1 + e)
        return e, q, phi

    @tf.function
    def _sort_ra_rs(self, r_core, r_cut):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        r_core = tf.where(r_core < r_cut, r_core, r_cut)
        r_cut = tf.where(r_core > r_cut, r_core, r_cut)
        r_core = tf.math.maximum(self._r_min, r_core)
        r_cut = tf.where(r_cut > r_core + self._r_min, r_cut, r_cut + self._r_min)
        return r_core, r_cut

    @tf.function
    def complex_deriv_dual(self, x, y, r_core, r_cut, e, q):
        sqe = tf.math.sqrt(e)
        rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2

        zci_re = 0
        zci_im = -0.5 * (1. - e ** 2) / sqe

        # r_core: zsi_rc = (a + bi)/(c + di)
        znum_rc_re = q * x  # a
        znum_rc_im = 2. * sqe * tf.math.sqrt(r_core ** 2 + rem2) - y / q  # b
        zden_rc_re = x  # c
        zden_rc_im = 2. * r_core * sqe - y  # d

        # r_cut: zsi_rcut = (a + ei)/(c + fi)
        # znum_rcut_re = znum_rc_re  # a
        znum_rcut_im = 2. * sqe * tf.math.sqrt(r_cut ** 2 + rem2) - y / q  # e
        # zden_rcut_re = zden_rc_re  # c
        zden_rcut_im = 2. * r_cut * sqe - y  # f

        """
        compute the ratio zis_rc / zis_rcut:
        zis_rc / zis_rcut = (znum_rc / zden_rc) / (znum_rcut / zden_rcut)
        zis_rc / zis_rcut = ((a + b * I) / (c + d * I)) / ((a + e * I) / (c + f * I));
        zis_rc / zis_rcut = (a * c + a * f * I + b * c * I - b * f) / (a * c + a * d * I + c * e * I - d * e);
        zis_rc / zis_rcut = (aa + bb * I) / (cc + dd * I)
        znum_rc = a + b * I
        zden_rc = c + d * I
        znum_rcut = a + e * I
        zden_rcut = c + f * I
        aa = (a * c - b * f);
        bb = (a * f + b * c);
        cc = (a * c - d * e);
        dd = (a * d + c * e);
        """
        aa = (znum_rc_re * zden_rc_re - znum_rc_im * zden_rcut_im)
        bb = (znum_rc_re * zden_rcut_im + znum_rc_im * zden_rc_re)
        cc = (znum_rc_re * zden_rc_re - zden_rc_im * znum_rcut_im)
        dd = (znum_rc_re * zden_rc_im + zden_rc_re * znum_rcut_im)

        # zis_rc / zis_rcut = ((aa * cc + bb * dd) / norm) + ((bb * cc - aa * dd) / norm) * I
        # zis_rc / zis_rcut = aaa + bbb * I
        norm = (cc ** 2 + dd ** 2)
        aaa = (aa * cc + bb * dd) / norm
        bbb = (bb * cc - aa * dd) / norm

        # compute the zr = log(zis_rc / zis_rcut) = log(aaa + bbb * I)
        norm2 = aaa ** 2 + bbb ** 2
        zr_re = tf.math.log(tf.math.sqrt(norm2))
        zr_im = tf.math.atan2(bbb, aaa)

        # now compute final result: zres = zci * log(zr)
        zres_re = zci_re * zr_re - zci_im * zr_im
        zres_im = zci_im * zr_re + zci_re * zr_im
        return zres_re, zres_im

    @tf.function
    def complex_hessian_single(self, x, y, r_w, e, q):
        sqe = tf.math.sqrt(e)
        qinv = 1. / q
        cxro = (1. + e) * (1. + e)  # rem ^ 2 = x ^ 2 / (1 + e ^ 2) + y ^ 2 / (1 - e ^ 2) Eq 2.3.6
        cyro = (1. - e) * (1. - e)
        ci = 0.5 * (1. - e ** 2) / sqe
        wrem = tf.math.sqrt(
            r_w ** 2 + x ** 2 / cxro + y ** 2 / cyro)  # wrem ^ 2 = r_w ^ 2 + rem ^ 2 with r_w core radius
        """
        zden = cpx(x, (2. * r_w * sqe - y)) # denominator
        znum = cpx(q * x, (2. * sqe * wrem - y / q)) # numerator

        zdidx = acpx(dcpx(cpx(2. * ci * sqe * x / cxro / wrem, -q * ci), znum),
                     dcpx(cpx(0., ci), zden)) # dI / dx
        with I in Eq 4.1.2
        zdidy = acpx(dcpx(cpx(-ci / q + 2. * ci * sqe * y / cyro / wrem, 0.), znum),
                     dcpx(cpx(ci, 0.), zden)) # dI / dy
        with I in Eq 4.1.2
        # in Eq
        4.1
        .2
        I = A * ln(u / v)) == > dI / dx = A * (u'/u-1/v) because v'=1
        res.a = b0 * zdidx.re
        res.b = res.d = b0 * (zdidy.re + zdidx.im) / 2.
        res.c = b0 * zdidy.im
        """
        den1 = 2. * sqe * wrem - y * qinv
        den1 = q ** 2 * x ** 2 + den1 ** 2
        num2 = 2. * r_w * sqe - y
        den2 = x ** 2 + num2 ** 2

        didxre = ci * (q * (2. * sqe * x ** 2 / cxro / wrem - 2. * sqe * wrem + y * qinv) / den1 + num2 / den2)

        # didxim = ci * ((2 * sqe * x * y * qinv / cxro / wrem - q * q * x - 4 * e * x / cxro) / den1 + x / den2)
        didyre = ci * ((2 * sqe * x * y * q / cyro / wrem - x) / den1 + x / den2)

        didyim = ci * ((2 * sqe * wrem * qinv - y * qinv ** 2 - 4 * e * y / cyro +
                        2 * sqe * y ** 2 / cyro / wrem * qinv) / den1 - num2 / den2)

        f_xx = didxre
        f_xy = didyre  # (didyre + didxim) / 2.
        f_yy = didyim
        return f_xx, f_xy, f_yy
