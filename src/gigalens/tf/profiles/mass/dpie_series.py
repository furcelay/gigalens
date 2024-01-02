import tensorflow as tf
from gigalens.tf.series.dpie_deflection_series import deflection_fns
from gigalens.tf.series.series_profile import MassSeries


class DPIESeries(MassSeries):
    """
    Series expansion of the given potential on a single variable
    """
    _params = ['r_cut', 'E0']
    _constants = ['r_core', 'center_x', 'center_y', 'e1', 'e2']
    _series_param = 'r_cut'
    _scale_param = 'E0'
    _name = "SeriesExpansion-dPIE"

    def __init__(self, order=3):
        super(DPIESeries, self).__init__(order)

    @tf.function
    def precompute_deriv(self, x, y, E0, r_core, r_cut, e1, e2, center_x, center_y):
        # E0 is not used
        e, q, phi = self._param_conv(e1, e2)
        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        f_x, f_y = [], []
        for i in range(self.order + 1):
            f_x_i, f_y_i = deflection_fns[i](x, y, e, r_core, r_cut)  # x, y, batch
            f_x_i, f_y_i = self._rotate(f_x_i, f_y_i, -phi)
            f_x.append(f_x_i)
            f_y.append(f_y_i)
        f_x, f_y = tf.stack(f_x, axis=-1), tf.stack(f_y, axis=-1)  # x, y, batch, (n+1)
        return f_x, f_y

    @tf.function
    def precompute_hessian(self, x, y, E0, r_core, r_cut, e1, e2, center_x, center_y):
        pass

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