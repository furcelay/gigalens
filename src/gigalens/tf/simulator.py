from typing import List, Dict

import numpy as np
import tensorflow as tf
from lenstronomy.Util.kernel_util import subgrid_kernel

import gigalens.tf.model
import gigalens.simulator


# TODO: no need for batched grid

class LensSimulator(gigalens.simulator.LensSimulatorInterface):
    def __init__(
            self,
            phys_model: gigalens.tf.model.PhysicalModel,
            sim_config: gigalens.simulator.SimulatorConfig,
            bs: int,
    ):
        super(LensSimulator, self).__init__(phys_model, sim_config, bs)
        self.supersample = int(sim_config.supersample)
        self.transform_pix2angle = (
            tf.eye(2) * sim_config.delta_pix
            if sim_config.transform_pix2angle is None
            else sim_config.transform_pix2angle
        )
        self.conversion_factor = tf.constant(
            tf.linalg.det(self.transform_pix2angle), dtype=tf.float32
        )
        self.transform_pix2angle = tf.constant(
            self.transform_pix2angle, dtype=tf.float32
        ) / float(self.supersample)

        if sim_config.pix_region is None:
            region = np.ones((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample), dtype=bool)
            img_region = np.ones((self.wcs.n_x, self.wcs.n_y))
        else:
            img_region = sim_config.pix_region
            region = np.repeat(img_region, self.supersample, axis=0).reshape(self.wcs.n_x * self.supersample,
                                                                             self.wcs.n_y)
            region = np.repeat(region, self.supersample, axis=1).reshape(self.wcs.n_x * self.supersample,
                                                                         self.wcs.n_y * self.supersample)
        self.region = tf.constant(tf.where(region))
        self.img_region = tf.constant(img_region.astype(np.float32))
        img_X, img_Y = self.wcs.pix2angle(self.region[:, 1], self.region[:, 0])
        self.img_X = tf.constant(
            tf.repeat(img_X[..., tf.newaxis], [bs], axis=-1), dtype=tf.float32
        )
        self.img_Y = tf.constant(
            tf.repeat(img_Y[..., tf.newaxis], [bs], axis=-1), dtype=tf.float32
        )

        self.bs = tf.constant(bs)

        self.numPix = tf.constant(sim_config.num_pix)
        self.bs = tf.constant(bs)
        self.depth = tf.constant(
            len(self.phys_model.lens_light) + len(self.phys_model.source_light)
        )
        self.kernel = None
        self.flat_kernel = None
        if sim_config.kernel is not None:
            kernel = subgrid_kernel(
                sim_config.kernel, sim_config.supersample, odd=True
            )[::-1, ::-1, tf.newaxis, tf.newaxis]
            self.kernel = tf.constant(
                tf.cast(tf.repeat(kernel, self.depth, axis=2), tf.float32),
                dtype=tf.float32,
            )
            self.flat_kernel = tf.constant(kernel, dtype=tf.float32)

    @tf.function
    def beta(self, x, y, lens_params: List[Dict]):
        beta_x, beta_y = x, y
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            f_xi, f_yi = lens.deriv(x, y, **p, **c)
            beta_x, beta_y = beta_x - f_xi, beta_y - f_yi
        return beta_x, beta_y

    @tf.function
    def magnification(self, x, y, lens_params: List[Dict]):
        f_xx, f_xy, f_yx, f_yy = tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x), tf.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            f_xx_i, f_xy_i, f_yx_i, f_yy_i = lens.hessian(x, y, **p, **c)
            f_xx += f_xx_i
            f_xy += f_xy_i
            f_yx += f_yx_i
            f_yy += f_yy_i
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx

        return 1. / det_A  # attention, if dividing by zero

    @tf.function
    def convergence(self, x, y, lens_params: List[Dict]):
        kappa = tf.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            kappa += lens.convergence(x, y, **p, **c)
        return kappa

    @tf.function
    def shear(self, x, y, lens_params: List[Dict]):
        gamma1, gamma2 = tf.zeros_like(x), tf.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            g1, g2 = lens.shear(x, y, **p, **c)
            gamma1 += g1
            gamma2 += g2
        return gamma1, gamma2

    @tf.function
    def simulate(self, params, no_deflection=False):
        if 'lens_mass' in params:
            lens_params = params['lens_mass']
        else:
            lens_params = [{} for _ in self.phys_model.lenses]
        if 'lens_light' in params:
            lens_light_params = params['lens_light']
        else:
            lens_light_params = [{} for _ in self.phys_model.lens_light]
        if 'source_light' in params:
            source_light_params = params['source_light']
        else:
            source_light_params = [{} for _ in self.phys_model.source_light]

        beta_x, beta_y = self.beta(self.img_X, self.img_Y, lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y

        img = tf.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for lightModel, p, c in zip(self.phys_model.lens_light, lens_light_params,
                                    self.phys_model.lens_light_constants):
            img = tf.tensor_scatter_nd_add(img,
                                           self.region,
                                           lightModel.light(self.img_X, self.img_Y, **p, **c))
        for lightModel, p, c in zip(self.phys_model.source_light, source_light_params,
                                    self.phys_model.source_light_constants):
            img = tf.tensor_scatter_nd_add(img,
                                           self.region,
                                           lightModel.light(beta_x, beta_y, **p, **c))

        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(img, (2, 0, 1))  # batch size, height, width
        ret = (
            img[..., tf.newaxis]
            if self.kernel is None
            else tf.nn.conv2d(
                img[..., tf.newaxis], self.flat_kernel, padding="SAME", strides=1
            )
        )
        ret = (
            tf.nn.avg_pool2d(
                ret, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else ret
        )
        return tf.squeeze(ret) * self.conversion_factor

    @tf.function
    def lstsq_simulate(
            self,
            params,
            observed_image,
            err_map,
            return_stacked=False,
            return_coeffs=False,
            no_deflection=False,
    ):
        if 'lens_mass' in params:
            lens_params = params['lens_mass']
        else:
            lens_params = [{} for _ in self.phys_model.lenses]
        if 'lens_light' in params:
            lens_light_params = params['lens_light']
        else:
            lens_light_params = [{} for _ in self.phys_model.lens_light]
        if 'source_light' in params:
            source_light_params = params['source_light']
        else:
            source_light_params = [{} for _ in self.phys_model.source_light]

        beta_x, beta_y = self.beta(self.img_X, self.img_Y, lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y
        img = tf.zeros((0, self.wcs.n_x, self.wcs.n_y, self.bs))
        for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
            img = tf.concat(
                (img,
                 tf.scatter_nd(self.region,
                               lightModel.light(self.img_X, self.img_Y, **p)[tf.newaxis, ...],
                               img.shape)),
                axis=0,
            )
        for lightModel, p in zip(self.phys_model.source_light, source_light_params):
            img = tf.concat(
                (img,
                 tf.scatter_nd(self.region,
                               lightModel.light(beta_x, beta_y, **p)[tf.newaxis, ...],
                               img.shape)),
            )
        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(
            img, (3, 1, 2, 0)
        )  # batch size, height, width, number of light components
        img = tf.reshape(
            img,
            (
                self.bs,
                self.numPix * self.supersample,
                self.numPix * self.supersample,
                self.depth,
            ),
        )

        img = (
            tf.nn.depthwise_conv2d(
                img, self.kernel, padding="SAME", strides=[1, 1, 1, 1]
            )
            if self.kernel is not None
            else img
        )
        ret = (
            tf.nn.avg_pool2d(
                img, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else img
        )
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        if return_stacked:
            return ret
        W = (1 / err_map)[..., tf.newaxis]
        Y = tf.reshape(tf.cast(observed_image, tf.float32) * tf.squeeze(W), (1, -1, 1))
        X = tf.reshape((ret * W), (self.bs, -1, self.depth))
        Xt = tf.transpose(X, (0, 2, 1))
        coeffs = (tf.linalg.pinv(Xt @ X, rcond=1e-6) @ Xt @ Y)[..., 0]
        if return_coeffs:
            return coeffs
        ret = tf.reduce_sum(ret * coeffs[:, tf.newaxis, tf.newaxis, :], axis=-1)
        return tf.squeeze(ret)

    @tf.function
    def simulate_source(self, params):
        source_light_params = params['source_light']

        img = tf.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for lightModel, p, c in zip(self.phys_model.source_light, source_light_params,
                                    self.phys_model.source_light_constants):
            img = tf.tensor_scatter_nd_add(img,
                                           self.region,
                                           lightModel.light(self.img_X, self.img_Y, **p, **c))

        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(img, (2, 0, 1))  # batch size, height, width
        ret = (
            img[..., tf.newaxis]
            if self.kernel is None
            else tf.nn.conv2d(
                img[..., tf.newaxis], self.flat_kernel, padding="SAME", strides=1
            )
        )
        ret = (
            tf.nn.avg_pool2d(
                ret, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else ret
        )
        return tf.squeeze(ret) * self.conversion_factor

    def simulate_lens_light(self, params):
        lens_light_params = params['lens_light']

        img = tf.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for lightModel, p, c in zip(self.phys_model.lens_light, lens_light_params,
                                    self.phys_model.lens_light_constants):
            img = tf.tensor_scatter_nd_add(img,
                                           self.region,
                                           lightModel.light(self.img_X, self.img_Y, **p, **c))

        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(img, (2, 0, 1))  # batch size, height, width
        ret = (
            img[..., tf.newaxis]
            if self.kernel is None
            else tf.nn.conv2d(
                img[..., tf.newaxis], self.flat_kernel, padding="SAME", strides=1
            )
        )
        ret = (
            tf.nn.avg_pool2d(
                ret, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else ret
        )
        return tf.squeeze(ret) * self.conversion_factor

    def simulate_images(self, params):
        lens_params = params['lens_mass']
        source_light_params = params['source_light']

        beta_x, beta_y = self.beta(self.img_X, self.img_Y, lens_params)

        img = tf.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for lightModel, p, c in zip(self.phys_model.source_light, source_light_params,
                                    self.phys_model.source_light_constants):
            img = tf.tensor_scatter_nd_add(img,
                                           self.region,
                                           lightModel.light(beta_x, beta_y, **p, **c))

        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(img, (2, 0, 1))  # batch size, height, width
        ret = (
            img[..., tf.newaxis]
            if self.kernel is None
            else tf.nn.conv2d(
                img[..., tf.newaxis], self.flat_kernel, padding="SAME", strides=1
            )
        )
        ret = (
            tf.nn.avg_pool2d(
                ret, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else ret
        )
        return tf.squeeze(ret) * self.conversion_factor
