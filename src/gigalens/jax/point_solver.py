import functools
from typing import Dict

import jax.numpy as jnp
from jax import jit

from gigalens.point_solver import PointSolverBase


class PointSolver(PointSolverBase):

    def __init__(self, phys_model, positions_structure):
        super().__init__(phys_model, positions_structure)

    @functools.partial(jit, static_argnums=(0,))
    def points_beta_barycentre(self, x, y, params: Dict[str, Dict]):
        lens_params = params.get('lens_mass', {})
        source_light_params = params.get('source_light', {})
        source_light_constants = self.phys_model.constants.get('source_light', {})
        beta_points = []
        beta_barycentre = []
        for x_i, y_i, i in zip(x, y, range(len(self.phys_model.source_light))):
            sp = source_light_params.get(str(i), {})
            sc = source_light_constants.get(str(i), {})
            deflect_rat = (sp | sc)['deflection_ratio']
            beta_points_i = jnp.stack(self.beta(x_i, y_i, lens_params, deflect_rat), axis=0)
            beta_points_i = jnp.transpose(beta_points_i, (2, 0, 1))  # batch size, xy, images
            beta_barycentre_i = jnp.mean(beta_points_i, axis=2, keepdims=True)
            beta_points.append(beta_points_i)
            beta_barycentre.append(beta_barycentre_i)
        return beta_points, beta_barycentre

    @functools.partial(jit, static_argnums=(0,))
    def points_magnification(self, x, y, params: Dict[str, Dict]):
        lens_params = params.get('lens_mass', {})
        source_light_params = params.get('source_light', {})
        source_light_constants = self.phys_model.constants.get('source_light', {})
        magnifications = []
        for x_i, y_i, i in zip(x, y, range(len(self.phys_model.source_light))):
            sp = source_light_params.get(str(i), {})
            sc = source_light_constants.get(str(i), {})
            deflect_rat = (sp | sc)['deflection_ratio']
            magnifications.append(self.magnification(x_i, y_i, lens_params, deflect_rat))
        return magnifications