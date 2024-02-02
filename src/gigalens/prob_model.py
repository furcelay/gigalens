from abc import ABC, abstractmethod


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
        self.centroids_errors_x = None
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


