from typing import List, Dict

from jax import numpy as jnp

import gigalens.model
import gigalens.profile


class PhysicalModel(gigalens.model.PhysicalModelBase):
    """A physical model for the lensing system.

    Args:
        lenses (:obj:`list` of :obj:`~gigalens.profile.MassProfile`): A list of mass profiles used to model the deflection
        lens_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the lens light
        source_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the source light

    Attributes:
        lenses (:obj:`list` of :obj:`~gigalens.profile.MassProfile`): A list of mass profiles used to model the deflection
        lens_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the lens light
        source_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the source light
    """

    def __init__(
        self,
        lenses: List[gigalens.profile.MassProfile],
        lens_light: List[gigalens.profile.LightProfile],
        source_light: List[gigalens.profile.LightProfile],
        lenses_constants: List[Dict] = None,
        lens_light_constants: List[Dict] = None,
        source_light_constants: List[Dict] = None,
    ):
        super(PhysicalModel, self).__init__(lenses, lens_light, source_light,
                                            lenses_constants, lens_light_constants, source_light_constants)
        self.lenses_constants = [{k: jnp.array(v) for k, v in d.items()}
                                 for d in self.lenses_constants]
        self.lens_light_constants = [{k: jnp.array(v) for k, v in d.items()}
                                     for d in self.lens_light_constants]
        self.source_light_constants = [{k: jnp.array(v) for k, v in d.items()}
                                       for d in self.source_light_constants]
