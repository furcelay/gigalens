import functools
from typing import List, Dict

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax
from lenstronomy.Util.kernel_util import subgrid_kernel
from objax.constants import ConvPadding
from objax.functional import average_pool_2d

import gigalens.model
import gigalens.simulator


# TODO: no need for batched grid

class LensSimulator(gigalens.simulator.LensSimulatorInterface):
    def __init__(
            self,
            phys_model: gigalens.model.PhysicalModelBase,
            sim_config: gigalens.simulator.SimulatorConfig,
            bs: int,
    ):
        super(LensSimulator, self).__init__(phys_model, sim_config, bs)
        self.supersample = int(sim_config.supersample)
        self.transform_pix2angle = (
            jnp.eye(2) * sim_config.delta_pix
            if sim_config.transform_pix2angle is None
            else sim_config.transform_pix2angle
        )
        self.conversion_factor = jnp.linalg.det(self.transform_pix2angle)
        self.transform_pix2angle = self.transform_pix2angle / float(self.supersample)

        if sim_config.pix_region is None:
            region = np.ones((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample), dtype=bool)
            img_region = np.ones((self.wcs.n_x, self.wcs.n_y))
        else:
            img_region = sim_config.pix_region
            region = np.repeat(img_region, self.supersample, axis=0).reshape(self.wcs.n_x * self.supersample,
                                                                             self.wcs.n_y)
            region = np.repeat(region, self.supersample, axis=1).reshape(self.wcs.n_x * self.supersample,
                                                                         self.wcs.n_y * self.supersample)

        self.region = jnp.array(jnp.where(region))
        self.img_region = jnp.array(img_region.astype(np.float32))
        img_X, img_Y = self.wcs.pix2angle(self.region[1], self.region[0])

        self.img_X = jnp.array(img_X)[..., jnp.newaxis]
        self.img_Y = jnp.array(img_Y)[..., jnp.newaxis]

        self.numPix = sim_config.num_pix
        self.bs = bs
        self.depth = (sum([x.depth for x in self.phys_model.lens_light]) +
                      sum([x.depth for x in self.phys_model.source_light_1]) +
                      sum([x.depth for x in self.phys_model.source_light_2]))
        self.kernel = None
        self.flat_kernel = None

        if sim_config.kernel is not None:
            kernel = subgrid_kernel(
                sim_config.kernel, sim_config.supersample, odd=True
            )[::-1, ::-1, jnp.newaxis, jnp.newaxis]
            self.kernel = jnp.repeat(jnp.array(kernel), self.depth, axis=2)
            self.flat_kernel = jnp.transpose(kernel, (2, 3, 0, 1))

    @functools.partial(jit, static_argnums=(0,))
    def alpha(self, x, y, lens_params: List[Dict]):
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(y)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            f_xi, f_yi = lens.deriv(x, y, **p, **c)
            f_x += f_xi
            f_y += f_yi
        return f_x, f_y

    @functools.partial(jit, static_argnums=(0,))
    def beta_1(self, x, y, lens_params: List[Dict]):
        f_x_1, f_y_1 = self.alpha(x, y, lens_params)
        beta_x_1, beta_y_1 = x - f_x_1, y - f_y_1
        return beta_x_1, beta_y_1

    @functools.partial(jit, static_argnums=(0,))
    def beta_2(self, x, y, lens_params: List[Dict], source_mass_1_params, deflection_ratio=1.):
        f_x_1, f_y_1 = self.alpha(x, y, lens_params)
        beta_x_1, beta_y_1 = x - deflection_ratio * f_x_1, y - deflection_ratio * f_y_1
        f_x_2, f_y_2 = self.alpha(beta_x_1, beta_y_1, source_mass_1_params)
        beta_x_2, beta_y_2 = x - f_x_1 - f_x_2, y - f_y_1 - f_y_2
        return beta_x_1, beta_y_2

    @functools.partial(jit, static_argnums=(0,))
    def points_beta_barycentre_1(self,
                               x,
                               y,
                               params):
        beta_points = []
        beta_barycentre = []
        for x_i, y_i, sp in zip(x, y):
            beta_points_i = jnp.stack(self.beta_1(x_i, y_i, params['lens_mass']), axis=0)
            beta_points_i = jnp.transpose(beta_points_i, (2, 0, 1))  # batch size, xy, images
            beta_barycentre_i = jnp.mean(beta_points_i, axis=2, keepdims=True)
            beta_points.append(beta_points_i)
            beta_barycentre.append(beta_barycentre_i)
        return beta_points, beta_barycentre

    @functools.partial(jit, static_argnums=(0,))
    def points_beta_barycentre_2(self,
                                 x,
                                 y,
                                 params):
        deflection_ratio = params.get('deflection_ratio', self.phys_model.deflection_ratio_constants)
        beta_points = []
        beta_barycentre = []
        for x_i, y_i, sp in zip(x, y):
            beta_points_i = jnp.stack(self.beta_2(x_i, y_i,
                                                  params['lens_mass'], params['source_mass_1'],
                                                  deflection_ratio), axis=0)
            beta_points_i = jnp.transpose(beta_points_i, (2, 0, 1))  # batch size, xy, images
            beta_barycentre_i = jnp.mean(beta_points_i, axis=2, keepdims=True)
            beta_points.append(beta_points_i)
            beta_barycentre.append(beta_barycentre_i)
        return beta_points, beta_barycentre

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, lens_params: List[Dict]):
        f_xx, f_xy, f_yx, f_yy = jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            f_xx_i, f_xy_i, f_yx_i, f_yy_i = lens.hessian(x, y, **p, **c)
            f_xx += f_xx_i
            f_xy += f_xy_i
            f_yx += f_yx_i
            f_yy += f_yy_i
        return f_xx, f_xy, f_yx, f_yy

    @functools.partial(jit, static_argnums=(0,))
    def magnification_1(self, x, y, lens_params: List[Dict]):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, lens_params)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
        return 1. / det_A  # attention, if dividing by zero

    @functools.partial(jit, static_argnums=(0,))
    def magnification_2(self, x, y, lens_params: List[Dict], source_mass_1_params, deflection_ratio=1.):
        f_xx_1, f_xy_1, f_yx_1, f_yy_1 = self.hessian(x, y, lens_params)
        beta_x_1, beta_y_1 = self.beta_1(x, y, lens_params)
        f_xx_2, f_xy_2, f_yx_2, f_yy_2 = self.hessian(beta_x_1, beta_y_1, source_mass_1_params)
        f_xx = f_xx_1 + deflection_ratio * f_xx_2
        f_xy = f_xy_1 + deflection_ratio * f_xy_2
        f_yx = f_yx_1 + deflection_ratio * f_yx_2
        f_yy = f_yy_1 + deflection_ratio * f_yy_2
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
        return 1. / det_A  # attention, if dividing by zero

    @functools.partial(jit, static_argnums=(0,))
    def points_magnification_1(self,
                               x,
                               y,
                               params):
        deflection_ratio = params.get('deflection_ratio', self.phys_model.deflection_ratio_constants)
        magnifications = []
        for x_i, y_i, sp in zip(x, y):
            magnifications.append(self.magnification_1(x_i, y_i, params['lens_mass'], deflection_ratio))
        return magnifications

    @functools.partial(jit, static_argnums=(0,))
    def points_magnification_2(self,
                               x,
                               y,
                               params):

        deflection_ratio = params.get('deflection_ratio', self.phys_model.deflection_ratio_constants)
        magnifications = []
        for x_i, y_i, sp in zip(x, y):
            magnifications.append(self.magnification_2(x_i, y_i,
                                                       params['lens_mass'], params['source_mass_1'],
                                                       deflection_ratio))
        return magnifications

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, lens_params: List[Dict]):
        kappa = jnp.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            kappa += lens.convergence(x, y, **p, **c)
        return kappa

    @functools.partial(jit, static_argnums=(0,))
    def shear(self, x, y, lens_params: List[Dict]):
        gamma1, gamma2 = jnp.zeros_like(x), jnp.zeros_like(x)
        for lens, p, c in zip(self.phys_model.lenses, lens_params, self.phys_model.lenses_constants):
            g1, g2 = lens.shear(x, y, **p, **c)
            gamma1 += g1
            gamma2 += g2
        return gamma1, gamma2

    @functools.partial(jit, static_argnums=(0,))
    def simulate(self, params):
        if 'lens_mass' in params:
            lens_params = params['lens_mass']
        else:
            lens_params = [{} for _ in self.phys_model.lenses]
        if 'lens_light' in params:
            lens_light_params = params['lens_light']
        else:
            lens_light_params = [{} for _ in self.phys_model.lens_light]
        if 'source_light_1' in params:
            source_light_1_params = params['source_light_1']
        else:
            source_light_1_params = [{} for _ in self.phys_model.source_light_1]
        if 'source_light_2' in params:
            source_light_2_params = params['source_light_2']
        else:
            source_light_2_params = [{} for _ in self.phys_model.source_light_2]
        if 'source_mass_1' in params:
            source_mass_1_params = params['source_mass_1']
        else:
            source_mass_1_params = [{} for _ in self.phys_model.source_mass_1]
        if 'deflection_ratio' in params:
            deflection_ratio = params['deflection_ratio']
        else:
            deflection_ratio = self.phys_model.deflection_ratio_constants

        img = jnp.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))

        for lightModel, p, c in zip(self.phys_model.lens_light, lens_light_params,
                                    self.phys_model.lens_light_constants):
            img = img.at[(self.region[0], self.region[1])].add(lightModel.light(self.img_X, self.img_Y, **p, **c,))

        # deflection 1
        f_x_1, f_y_1 = self.alpha(self.img_X, self.img_Y, lens_params)
        beta_x_1, beta_y_1 = self.img_X - deflection_ratio * f_x_1, self.img_Y - deflection_ratio * f_y_1

        # deflected source light 1, considering redshift
        for lightModel, p, c in zip(self.phys_model.source_light_1,
                                    source_light_1_params, self.phys_model.source_light_1_constants):
            img = img.at[(self.region[0], self.region[1])].add(lightModel.light(beta_x_1, beta_y_1, **p, **c))

        # deflection 2
        f_x_2, f_y_2 = self.alpha(beta_x_1, beta_y_1, source_mass_1_params)
        beta_x_2, beta_y_2 = self.img_X - f_x_1 - f_x_2, self.img_Y - f_y_1 - f_y_2

        # deflected source light 2, considering redshift
        for lightModel, p, c in zip(self.phys_model.source_light_2,
                                    source_light_2_params, self.phys_model.source_light_2_constants):
            img = img.at[(self.region[0], self.region[1])].add(lightModel.light(beta_x_2, beta_y_2, **p, **c))

        img = jnp.transpose(img, (2, 0, 1))
        img = jnp.nan_to_num(img)
        ret = (
            lax.conv(img[:, jnp.newaxis, ...], self.flat_kernel, (1, 1), "SAME")
            if self.flat_kernel is not None
            else img
        )
        ret = (
            average_pool_2d(ret, size=self.supersample, padding=ConvPadding.SAME)
            if self.supersample != 1
            else ret
        )
        return jnp.squeeze(ret) * self.conversion_factor

    @functools.partial(jit, static_argnums=(0,))
    def lstsq_simulate(
            self,
            params,
            observed_image,
            err_map,
            return_stacked=False,
            return_coeffs=False,
            no_deflection=False,
    ):
        pass
