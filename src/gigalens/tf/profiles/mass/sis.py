import tensorflow as tf
from gigalens.tf.profile import MassProfile


class SIS(MassProfile):
    _name = "SIS"
    _params = ["theta_E", "center_x", "center_y"]

    def __init__(self):
        super(SIS, self).__init__()

    @tf.function
    def deriv(self, x, y, theta_E, center_x, center_y):
        x, y = x - center_x, y - center_y
        R = tf.math.sqrt(x ** 2 + y ** 2)
        a = tf.where(R == 0, 0.0, theta_E / R)
        return a * x, a * y

    @tf.function
    def hessian(self, x, y, theta_E, center_x, center_y):

        x, y = x - center_x, y - center_y
        R = (x**2 + y**2)**(3./2)
        a = tf.where(R == 0, 0.0, theta_E / R)

        f_xx = y**2 * a
        f_yy = x**2 * a
        f_xy = -x * y * a
        return f_xx, f_xy, f_xy, f_yy
