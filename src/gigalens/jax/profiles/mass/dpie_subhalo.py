from typing import List, Dict
from gigalens.jax.profiles.mass.scaling_relation import ScalingRelation
from gigalens.jax.profiles.mass.piemd import DPIE


class DPIESubhalo(ScalingRelation):

    def __init__(self,
                 mag_star: float,
                 galaxy_catalogue: Dict[str, List],
                 scaling_params_power=None,
                 **kwargs):
        if scaling_params_power is None:
            scaling_params_power = {'theta_E': 0.5, 'r_core': 0.5, 'r_cut': 0.5}

        super(DPIESubhalo, self).__init__(profile=DPIE(),
                                          scaling_params=['theta_E', 'r_core', 'r_cut'],
                                          mag_star=mag_star,
                                          scaling_params_power=scaling_params_power,
                                          galaxy_catalogue=galaxy_catalogue,
                                          **kwargs)
