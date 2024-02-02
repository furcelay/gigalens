from abc import ABC
from typing import List, Dict

import gigalens.profile


class PhysicalModelBase(ABC):
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
        constants: Dict = None
    ):
        self.lenses = lenses
        self.lens_light = lens_light
        self.source_light = source_light
        if constants is None:
            constants = {  # TODO: review if change names
                'lens_mass': {str(i): {} for i in range(len(self.lenses))},
                'source_light': {str(i): {} for i in range(len(self.source_light))},
                'lens_light': {str(i): {} for i in range(len(self.lens_light))}
            }
        self._constants = constants

    @property
    def constants(self):
        return self._constants


class LensModelBase(ABC):

    def __init__(self, lenses, lenses_constants):
        self.lenses = lenses
        self.lenses_constants = lenses_constants

    def alpha(self, x, y, lens_params: Dict[str, Dict]):
        pass

    def beta(self, x, y, lens_params: Dict[str, Dict], deflection_ratio=1.):
        pass

    def hessian(self, x, y, lens_params: Dict[str, Dict]):
        pass

    def magnification(self, x, y, lens_params: Dict[str, Dict], deflection_ratio=1.):
        pass

    def convergence(self, x, y, lens_params: Dict[str, Dict]):
        pass

    def shear(self, x, y, lens_params: Dict[str, Dict]):
        pass
