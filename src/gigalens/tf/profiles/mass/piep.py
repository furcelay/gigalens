import tensorflow as tf
from gigalens.tf.profile import MassProfile
from gigalens.tf.profiles.mass.piemd import DPIS

"""
Dual Pseudo Isothermal Elliptical Potential (dPIEP)
Composed of two PIEP from Kassiola & Kovner (1993) similar (but not equal) to Elíasdóttir (2007)
with ellipticity on the potential. Gives a pseudo elliptical mass

Velocity dispersion equivalence:
Lenstool:           theta_E = 6 pi (D_LS / D_S) (simga_v(Ls) / c)^2
Limousin (2005):    theta_E = 4 pi (D_LS / D_S) (simga_v(Li) / c)^2
Elíasdóttir (2007): theta_E = 6 pi (D_LS / D_S) (simga_v(El) / c)^2 * r_cut^2 / (r_cut^2 - r_core^2)
"""


class DPIEP(MassProfile):
    """
    Dual Pseudo Isothermal Potential (pseudo elliptical mass distribution)
    """

    _name = "dPIE"
    _params = ['theta_E', 'Ra', 'Rs', 'center_x', 'center_y', 'e1', 'e2']

    def __init__(self):
        self.spherical = DPIS()
        self._r_min = 0.0001
        super(DPIEP, self).__init__()

    @tf.function
    def deriv(self, x, y, theta_E, Ra, Rs, e1, e2, center_x=0, center_y=0):
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * tf.math.sqrt(1 - e), y * tf.math.sqrt(1 + e)

        fx, fy = self.spherical.deriv(x, y, theta_E, Ra, Rs, center_x=0, center_y=0)
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
