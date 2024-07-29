from abc import ABC, abstractmethod
from typing import List


class Parameterized(ABC):
    """Interface for a parameterized profile of any kind.

    Attributes:
        name (str): Name of the profile
        params (:obj:`list` of :obj:`str`): List of parameter names
    """

    _name: str  # Static class level default for name
    _params: List[str]  # Static class level default for parameter names

    def __init__(self, *args, **kwargs):
        # self.constants = constants  TODO: include constants in Parametrized plus a @with_constants decorator
        self.name = self._name
        self.params = self._params.copy()  # [param for param in self._params if param not in constants]

    def __str__(self):
        return self.name


class LightProfile(Parameterized, ABC):
    """Interface for a light profile.

    Keyword Args:
         use_lstsq (bool): Whether to use least squares to solve for linear parameters

    Attributes:
         _use_lstsq (bool): Whether to use least squares to solve for linear parameters
    """

    _amp = ""

    def __init__(self, use_lstsq=False, is_source=False, *args, **kwargs):
        super(LightProfile, self).__init__(*args, **kwargs)
        self._use_lstsq = use_lstsq
        self._is_source = is_source
        self.depth = 1
        if not self.use_lstsq:
            self.params.append(self._amp)
        if is_source:
            self.params.append("z_source")

    @property
    def use_lstsq(self):
        return self._use_lstsq

    @use_lstsq.setter
    def use_lstsq(self, use_lstsq: bool):
        """
        Arguments:
             use_lstsq (bool): Whether to use least squares to solve for linear parameters
        """
        if use_lstsq and not self.use_lstsq:  # Turn least squares on
            self.params.pop(self.params.index(self._amp))
        elif not use_lstsq and self.use_lstsq:  # Turn least squares off
            self.params.append(self._amp)
        self._use_lstsq = use_lstsq

    @property
    def is_source(self):
        return self._is_source

    @is_source.setter
    def is_source(self, is_source: bool):
        """
        Arguments:
             is_source (bool): Whether the light profile is a lensed source
        """
        if is_source and not self.is_source:  # Include the deflection ratio to set the source distance
            self.params.append("deflection_ratio")
        elif not is_source and self.is_source:
            self.params.pop(self.params.index("deflection_ratio"))
        self._is_source = is_source

    @abstractmethod
    def light(self, x, y, **kwargs):
        pass

    def __str__(self):
        return f"{self.name}(use_lstsq={self.use_lstsq}, is_source={self.is_source})"


class MassProfile(Parameterized, ABC):
    """Interface for a mass profile."""

    def __init__(self, *args, **kwargs):
        super(MassProfile, self).__init__(*args, **kwargs)

    @abstractmethod
    def deriv(self, x, y, **kwargs):
        """Calculates deflection angle.

        Args:
            x: :math:`x` coordinate at which to evaluate the deflection
            y: :math:`y` coordinate at which to evaluate the deflection
            **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

        Returns:
            A tuple :math:`(\\alpha_x, \\alpha_y)` containing the deflection angle in the :math:`x` and :math:`y` directions

        """
        pass


class CosmoBase(Parameterized, ABC):

    def __init__(self, z_lens, z_source_ref=10.0):
        super(CosmoBase, self).__init__()
        self.z_lens = z_lens
        self.z_source_ref = z_source_ref

    @abstractmethod
    def efunc(self, z, H0, Om0, k, w0):
        pass

    def omega_rad0(self, H0):
        h = H0 / 100
        return 2.469e-5 * h ** -2.0 * (1.0 + 0.2271 * self.Neff)

    def comoving_distance_z1z2(self, z1, z2, H0, Om0, k, w0):
        def integrand(z):
            return 1.0 / self.efunc(z, H0, Om0, k, w0)
        return (self.c / H0) * self._integrate(integrand, z1, z2)

    def comoving_distance(self, z, H0, Om0, k, w0):
        return self.comoving_distance_z1z2(0, z, H0, Om0, k, w0)

    def angular_distance(self, z, H0, Om0, k, w0):
        return (1 / (1 + z)) * self.comoving_distance(z, H0, Om0, k, w0)

    def angular_distance_z1z2(self, z1, z2, H0, Om0, k, w0):
        return (1 / (1 + z2)) * self.comoving_distance_z1z2(z1, z2, H0, Om0, k, w0)

    def luminosity_distance(self, z, H0, Om0, k, w0):
        return (1 + z) * self.comoving_distance(z, H0, Om0, k, w0)

    def lensing_distance(self, z_source, H0, Om0, k, w0):
        d_ls = self.angular_distance_z1z2(self.z_lens, z_source, H0, Om0, k, w0)
        d_s = self.angular_distance(z_source, H0, Om0, k, w0)
        return d_ls / d_s

    def deflection_ratio(self, z_source, H0, Om0, k, w0):
        d_lensing = self.lensing_distance(z_source, H0, Om0, k, w0)
        d_ref = self.lensing_distance(self.z_source_ref, H0, Om0, k, w0)
        return d_lensing / d_ref

    @staticmethod
    @abstractmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        pass
