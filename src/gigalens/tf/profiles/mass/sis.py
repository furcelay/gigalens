import tensorflow as tf
from gigalens.tf.profile import MassProfile


class SIS(MassProfile):
    _name = "SIS"
    _params = ["theta_E", "center_x", "center_y"]

    @tf.function
    def deriv(self, x, y, theta_E, center_x, center_y):
        x, y = x - center_x, y - center_y
        R = tf.math.sqrt(x ** 2 + y ** 2)
        a = tf.where(R == 0, 0.0, theta_E / R)
        return a * x, a * y

    @tf.function
    def hessian(self, x, y, theta_E, center_x, center_y):

        dx, dy = x - center_x, y - center_y
        R = (x**2 + y**2)**(3./2)
        a = tf.where(R == 0, 0.0, theta_E / R)

        f_xx = dy**2 * a
        f_yy = dx**2 * a
        f_xy = -dx * dy * a
        return f_xx, f_xy, f_xy, f_yy
