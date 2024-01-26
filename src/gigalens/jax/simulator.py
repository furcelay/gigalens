import functools
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax
from lenstronomy.Util.kernel_util import subgrid_kernel
from objax.constants import ConvPadding
from objax.functional import average_pool_2d

import gigalens.jax.physical_model
import gigalens.simulator


# TODO: no need for batched grid

class LensSimulator(gigalens.simulator.LensSimulatorInterface):
    def __init__(
            self,
            phys_model: gigalens.jax.physical_model.PhysicalModel,
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
        self.depth = sum([x.depth for x in self.phys_model.lens_light]) + sum(
            [x.depth for x in self.phys_model.source_light])
        self.kernel = None
        self.flat_kernel = None

        if sim_config.kernel is not None:
            kernel = subgrid_kernel(
                sim_config.kernel, sim_config.supersample, odd=True
            )[::-1, ::-1, jnp.newaxis, jnp.newaxis]
            self.kernel = jnp.repeat(jnp.array(kernel), self.depth, axis=2)
            self.flat_kernel = jnp.transpose(kernel, (2, 3, 0, 1))

    @functools.partial(jit, static_argnums=(0,))
    def beta(self, x, y, lens_params: Dict[str, Dict]):
        lens_constants = self.phys_model.constants.get('lens_mass', {})
        beta_x, beta_y = x, y
        for i, lens in enumerate(self.phys_model.lenses):
            p = lens_params.get(str(i), {})
            c = lens_constants.get(str(i), {})
            f_xi, f_yi = lens.deriv(x, y, **p, **c)
            beta_x, beta_y = beta_x - f_xi, beta_y - f_yi
        return beta_x, beta_y

    @functools.partial(jit, static_argnums=(0,))
    def magnification(self, x, y, lens_params: Dict[str, Dict]):
        lens_constants = self.phys_model.constants.get('lens_mass', {})
        f_xx, f_xy, f_yx, f_yy = jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)
        for i, lens in enumerate(self.phys_model.lenses):
            p = lens_params.get(str(i), {})
            c = lens_constants.get(str(i), {})
            f_xx_i, f_xy_i, f_yx_i, f_yy_i = lens.hessian(x, y, **p, **c)
            f_xx += f_xx_i
            f_xy += f_xy_i
            f_yx += f_yx_i
            f_yy += f_yy_i
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx

        return 1. / det_A  # attention, if dividing by zero

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, lens_params: Dict[str, Dict]):
        lens_constants = self.phys_model.constants.get('lens_mass', {})
        kappa = jnp.zeros_like(x)
        for i, lens in enumerate(self.phys_model.lenses):
            p = lens_params.get(str(i), {})
            c = lens_constants.get(str(i), {})
            kappa += lens.convergence(x, y, **p, **c)
        return kappa

    @functools.partial(jit, static_argnums=(0,))
    def shear(self, x, y, lens_params: Dict[str, Dict]):
        lens_constants = self.phys_model.constants.get('lens_mass', {})
        gamma1, gamma2 = jnp.zeros_like(x), jnp.zeros_like(x)
        for i, lens in enumerate(self.phys_model.lenses):
            p = lens_params.get(str(i), {})
            c = lens_constants.get(str(i), {})
            g1, g2 = lens.shear(x, y, **p, **c)
            gamma1 += g1
            gamma2 += g2
        return gamma1, gamma2

    @functools.partial(jit, static_argnums=(0,))
    def simulate(self, params, no_deflection=False):
        lens_params = params.get('lens_mass', {})
        lens_light_params = params.get('lens_light', {})
        source_light_params = params.get('source_light', {})

        lens_light_constants = self.phys_model.constants.get('lens_light', {})
        source_light_constants = self.phys_model.constants.get('source_light', {})

        beta_x, beta_y = self.beta(self.img_X, self.img_Y, lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y

        img = jnp.zeros((self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for i, lightModel in enumerate(self.phys_model.lens_light):
            p = lens_light_params.get(str(i), {})
            c = lens_light_constants.get(str(i), {})
            img = img.at[(self.region[0], self.region[1])].add(lightModel.light(self.img_X, self.img_Y, **p, **c))
        for i, lightModel in enumerate(self.phys_model.source_light):
            p = source_light_params.get(str(i), {})
            c = source_light_constants.get(str(i), {})
            img = img.at[(self.region[0], self.region[1])].add(lightModel.light(beta_x, beta_y, **p, **c))
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

    @functools.partial(jit, static_argnums=(0,))  # TODO: update lstsq_simulate
    def lstsq_simulate(
            self,
            params,
            observed_image,
            err_map,
            return_stacked=False,
            return_coeffs=False,
            no_deflection=False,
    ):
        lens_params = params.get('lens_mass', {})
        lens_light_params = params.get('lens_light', {})
        source_light_params = params.get('source_light', {})

        lens_light_constants = self.phys_model.constants.get('lens_light', {})
        source_light_constants = self.phys_model.constants.get('source_light', {})

        beta_x, beta_y = self.beta(self.img_X, self.img_Y, lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y

        img = jnp.zeros((0, self.wcs.n_x * self.supersample, self.wcs.n_y * self.supersample, self.bs))
        for i, lightModel in enumerate(self.phys_model.lens_light):
            p = lens_light_params.get(str(i), {})
            c = lens_light_constants.get(str(i), {})
            img = jnp.concatenate((img, lightModel.light(self.img_X, self.img_Y, **p, **c)), axis=0)
        for i, lightModel in enumerate(self.phys_model.source_light):
            p = source_light_params.get(str(i), {})
            c = source_light_constants.get(str(i), {})
            img = jnp.concatenate((img, lightModel.light(beta_x, beta_y, **p, **c)), axis=0)

        img = jnp.nan_to_num(img)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n components, h, w
        ret = jax.lax.conv_general_dilated(img, self.kernel, (1, 1), padding='SAME', feature_group_count=self.depth,
                                           dimension_numbers=(
                                           'NCHW', 'HWOI', 'NCHW')) if self.flat_kernel is not None else img
        ret = average_pool_2d(ret, size=(self.supersample, self.supersample),
                              padding="SAME") if self.supersample != 1 else ret
        ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n components
        if return_stacked:
            return ret
        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret * W), (self.bs, -1, self.depth))
        Xt = jnp.transpose(X, (0, 2, 1))
        coeffs = (jnp.linalg.pinv(Xt @ X, rcond=1e-6) @ Xt @ Y)[..., 0]
        if return_coeffs:
            return coeffs
        ret = jnp.sum(ret * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)
        return jnp.squeeze(ret)

# TODO: new simulation features for JAX
