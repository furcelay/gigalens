from abc import ABC, abstractmethod
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
        cosmo: gigalens.profile.CosmoBase,
        constants: Dict = None
    ):
        self.lenses = lenses
        self.lens_light = lens_light
        self.source_light = source_light
        self.cosmo = cosmo
        if constants is None:
            constants = {  # TODO: review if change names
                'lens_mass': {str(i): {} for i in range(len(self.lenses))},
                'source_light': {str(i): {} for i in range(len(self.source_light))},
                'lens_light': {str(i): {} for i in range(len(self.lens_light))},
                'cosmo': {}
            }
        self._constants = constants

    @property
    def constants(self):
        return self._constants


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

    def __init__(self,
                 prior,
                 include_pixels=True,
                 include_positions=True
                 ):

        self.prior = prior

        self.include_pixels = include_pixels
        self.include_positions = include_positions

        self.observed_image = None
        self.error_map = None
        self.background_rms = None
        self.exp_time = None
        self.centroids_x = None
        self.centroids_y = None
        self.centroids_errors_x = None  # TODO: assume symmetric error, only one needed
        self.centroids_errors_y = None
        self.n_position = None
        self.bij = None
        self.pack_bij = None
        self.unconstraining_bij = None

    @abstractmethod
    def log_prob(self, simulator, z):
        """
        Returns the unconstrained log posterior density (i.e., includes the Jacobian factor due to the bijector)

        Args:
             simulator (:obj:`~gigalens.simulator.LensSimulatorInterface`): an object that can simulate a lens with (unconstrained parameters) z
             z: Unconstrained parameters
        """
        pass

    @abstractmethod
    def log_like(self, simulator, z):
        pass

    @abstractmethod
    def log_prior(self, z):
        pass

    @abstractmethod
    def stats_pixels(self, simulator, x):
        pass

    @abstractmethod
    def stats_positions(self, simulator, x):
        pass


