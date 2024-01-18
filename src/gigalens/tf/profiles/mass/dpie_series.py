import tensorflow as tf
from gigalens.tf.series.profiles.dpie import deriv_fns, hessian_fns
from gigalens.tf.series.series_profile import MassSeries


class DPIESeries(MassSeries):
    """
    Series expansion of the given potential on a single variable
    """
    _params = ['r_cut', 'theta_E']
    _constants = ['r_core', 'center_x', 'center_y', 'e1', 'e2']
    _series_param = 'r_cut'
    _scale_param = 'theta_E'
    _name = "SeriesExpansion-dPIE"

    def __init__(self, order=3):
        super(DPIESeries, self).__init__(order)

    @tf.function
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
        f_x, f_y = tf.stack(f_x, axis=-1), tf.stack(f_y, axis=-1)  # x, y, batch, (n+1)
        return f_x, f_y

    @tf.function
    def precompute_hessian(self, order, x, y, theta_E, r_core, r_cut, e1, e2, center_x, center_y):
        # theta_E is not used
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        f_xx, f_xy, f_yy = [], [], []
        for i in range(order + 1):
            f_xx_i, f_xy_i, f_yy_i = hessian_fns[i](x, y, e, r_core, r_cut)  # x, y, batch
            f_xx_i, f_xy_i, f_yy_i = self._hessian_rotate(f_xx_i, f_xy_i, f_yy_i, -phi)
            f_xx.append(f_xx_i)
            f_xy.append(f_xy_i)
            f_yy.append(f_yy_i)
        f_xx, f_xy, f_yy = tf.stack(f_xx, axis=-1), tf.stack(f_xy, axis=-1), tf.stack(f_yy, axis=-1)  # x, y, batch, (n+1)
        return f_xx, f_xy, f_yy

    @tf.function
    def _param_conv(self, e1, e2):
        phi = tf.atan2(e2, e1) / 2
        e = tf.math.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - e) / (1 + e)
        return e, q, phi

    @tf.function
    def _rotate(self, x, y, phi):
        cos_phi = tf.cos(phi, name=self.name + "rotate-cos")
        sin_phi = tf.sin(phi, name=self.name + "rotate-sin")
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @tf.function
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
        return f_xx_rot, f_xy_rot, f_yy_rot
