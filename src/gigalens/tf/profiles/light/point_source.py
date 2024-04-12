import tensorflow as tf

import gigalens.profile


class PointSource(gigalens.profile.LightProfile):
    _name = "POINT_SOURCE"
    _params = ["center_x", "center_y"]
    _amp = "amp"

    def __init__(self, use_lstsq=False, is_source=True):
        super(PointSource, self).__init__(use_lstsq=use_lstsq, is_source=is_source)

    @tf.function
    def light(self, x, y):
        ret = tf.zeros_like(x)
        return ret[tf.newaxis, ...] if self.use_lstsq else ret
