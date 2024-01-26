import sympy as sp
from gigalens.series_codegen.profile import SPMassProfile


class DPIE(SPMassProfile):

    # params must be at the same order of evaluation
    _name = "DPIE"
    _params = ['x', 'y', 'e', 'r_core', 'r_cut']
    _series_var = 'r_cut'

    def __int__(self):
        super(DPIE, self).__init__()

    def deriv(self, x, y, e, r_core, r_cut):
        scale = r_cut / (r_cut - r_core)
        q = (1 - e) / (1 + e)
        sqe = sp.sqrt(e)
        rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2

        zci_re = 0
        zci_im = -0.5 * (1. - e ** 2) / sqe

        # r_core: zsi_rc = (a + bi)/(c + di)
        znum_rc_re = q * x  # a
        znum_rc_im = 2. * sqe * sp.sqrt(r_core ** 2 + rem2) - y / q  # b
        zden_rc_re = x  # c
        zden_rc_im = 2. * r_core * sqe - y  # d

        # r_cut: zsi_rcut = (a + ei)/(c + fi)
        # znum_rcut_re = znum_rc_re  # a
        znum_rcut_im = 2. * sqe * sp.sqrt(r_cut ** 2 + rem2) - y / q  # e
        # zden_rcut_re = zden_rc_re  # c
        zden_rcut_im = 2. * r_cut * sqe - y  # f

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
        zr_re = sp.log(sp.sqrt(norm2))
        zr_im = sp.atan2(bbb, aaa)

        # now compute final result: zres = zci * log(zr)
        zres_re = zci_re * zr_re - zci_im * zr_im
        zres_im = zci_im * zr_re + zci_re * zr_im
        return scale * sp.Matrix([zres_re, zres_im])

    def hessian(self, x, y, e, r_core, r_cut):
        scale = r_cut / (r_cut - r_core)
        q = (1 - e) / (1 + e)
        f_xx_core, f_xy_core, f_yy_core = self.complex_hessian_single(x, y, r_core, e, q)
        f_xx_cut, f_xy_cut, f_yy_cut = self.complex_hessian_single(x, y, r_cut, e, q)
        f_xx = (f_xx_core - f_xx_cut)
        f_xy = f_yx = (f_xy_core - f_xy_cut)
        f_yy = (f_yy_core - f_yy_cut)
        return scale * sp.Matrix([f_xx, f_xy, f_yx, f_yy])

    @staticmethod
    def complex_hessian_single(x, y, r_w, e, q):
        sqe = sp.sqrt(e)
        qinv = 1. / q
        cxro = (1. + e) * (1. + e)  # rem ^ 2 = x ^ 2 / (1 + e ^ 2) + y ^ 2 / (1 - e ^ 2) Eq 2.3.6
        cyro = (1. - e) * (1. - e)
        ci = 0.5 * (1. - e ** 2) / sqe
        wrem = sp.sqrt(
            r_w ** 2 + x ** 2 / cxro + y ** 2 / cyro)  # wrem ^ 2 = r_w ^ 2 + rem ^ 2 with r_w core radius

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
