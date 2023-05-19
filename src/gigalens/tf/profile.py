import tensorflow as tf
import gigalens.profile


class MassProfile(gigalens.profile.MassProfile):
    """Interface for a mass profile."""

    @tf.function
    def hessian(self, x, y, *args, **kwargs):
        # use autograd to compute derivatives
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            fx, fy = self.deriv(x, y, *args, **kwargs)

        f_xx, f_xy = tape.gradient(fx, [x, y])
        f_yx, f_yy = tape.gradient(fy, [x, y])

        return f_xx, f_xy, f_yx, f_yy
