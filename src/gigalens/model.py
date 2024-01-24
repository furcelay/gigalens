from abc import ABC, abstractmethod
from typing import List, Dict

import gigalens.profile


class PhysicalModelBase:
    """A physical model for the lensing system.

    Args:
        lenses (:obj:`list` of :obj:`~gigalens.profile.MassProfile`): A list of mass profiles used to model the deflection
        lens_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the lens light
        source_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the source light

    Attributes:
        lenses (:obj:`list` of :obj:`~gigalens.profile.MassProfile`): A list of mass profiles used to model the deflection
        lens_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the lens light
        source_light (:obj:`list` of :obj:`~gigalens.profile.LightProfile`): A list of light profiles used to model the source light
        lenses_constants: (:obj:`list` of :obj:`dict`): fixed lenses parameters
        lens_light_constants: (:obj:`list` of :obj:`dict`): fixed lens light parameters
        source_light_constants: (:obj:`list` of :obj:`dict`): fixed source light parameters
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
        self.lenses = lenses
        self.lens_light = lens_light
        self.source_light = source_light
        if lenses_constants is None:
            lenses_constants = [dict() for _ in range(len(lenses))]
        if lens_light_constants is None:
            lens_light_constants = [dict() for _ in range(len(lens_light))]
        if source_light_constants is None:
            source_light_constants = [dict() for _ in range(len(source_light))]
        self.lenses_constants = lenses_constants
        self.lens_light_constants = lens_light_constants
        self.source_light_constants = source_light_constants

    @property
    def constants(self):
        return {  # TODO: review if change names
            'lens_mass': self.lenses_constants,
            'source_light': self.source_light_constants,
            'lens_light': self.lens_light_constants
        }


class ProbabilisticModel(ABC):
    """A probabilistic model for the lensing system.

    Args:
        prior: Prior distribution of lens parameters
        bij: A bijector that can unconstrain physical parameters (e.g., applying an exponential bijector to strictly positive variables)
        *args: Information about observed data (typically includes the observed image, estimated noise characteristics, etc.)

    Attributes:
        prior: Prior distribution of lens parameters
        bij: A bijector that can unconstrain physical parameters (e.g., applying an exponential bijector to strictly positive variables)
    """

    def __init__(self, prior, bij=None, *args):
        self.prior = prior
        self.bij = bij

    @abstractmethod
    def log_prob(self, simulator, z):
        """
        Returns the unconstrained log posterior density (i.e., includes the Jacobian factor due to the bijector)

        Args:
             simulator (:obj:`~gigalens.simulator.LensSimulatorInterface`): an object that can simulate a lens with (unconstrained parameters) z
             z: Unconstrained parameters
        """
        pass
