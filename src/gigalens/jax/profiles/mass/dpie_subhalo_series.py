from typing import List, Dict
from gigalens.jax.profiles.mass.scaling_series import ScalingRelationSeries
from gigalens.jax.profiles.mass.dpie_series import DPIESeries


class DPIESubhaloSeries(ScalingRelationSeries):
    _params = ['theta_E', 'r_cut']
    _constants = ['r_core', 'center_x', 'center_y', 'e1', 'e2']
    _name = 'Scaled-SeriesExpansion-dPIE'

    def __init__(self,
                 lum_star: float,
                 galaxy_catalogue: Dict[str, List],
                 scaling_params_power=None,
                 order=3,
                 chunk_size=None):

        if scaling_params_power is None:
            scaling_params_power = {'theta_E': 0.5, 'r_core': 0.5, 'r_cut': 0.5}
        profile = DPIESeries(order=order)

        super(DPIESubhaloSeries, self).__init__(profile=profile,
                                                order=order,
                                                lum_star=lum_star,
                                                scaling_params=['theta_E', 'r_core', 'r_cut'],
                                                scaling_params_power=scaling_params_power,
                                                galaxy_catalogue=galaxy_catalogue,
                                                chunk_size=chunk_size)
