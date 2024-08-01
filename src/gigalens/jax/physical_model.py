from typing import List, Dict

from jax import numpy as jnp

import gigalens.cosmo
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
        constants: (:obj:`dict` of :obj:`dict`): fixed parameters with the same structure as the prior
    """

    def __init__(
        self,
            lenses: List[gigalens.profile.MassProfile],
            lens_light: List[gigalens.profile.LightProfile],
            source_light: List[gigalens.profile.LightProfile],
            cosmo: gigalens.cosmo.CosmoBase,
            constants: Dict = None,
    ):

        super(PhysicalModel, self).__init__(lenses, lens_light, source_light, cosmo,
                                            constants)
        self._constants = _to_jnp_array(self._constants)


def _to_jnp_array(d):
    """
    Recursively apply tf.constant(v, dtype=tf.float32) to all leaf values in a structured dictionary.
    """
    if isinstance(d, dict):
        return {k: _to_jnp_array(v) for k, v in d.items()}
    else:
        # Apply tf.constant to the leaf value
        return jnp.array(d)
