from abc import ABC, abstractmethod

from gigalens.profile import Parameterized


class CosmoBase(Parameterized, ABC):

    Neff = 3.04  # number of relativistic species
    c = 299792.458  # km/s #speed of light

    def __init__(self, z_lens, z_source_ref=10.0):
        super(CosmoBase, self).__init__()
        self.z_lens = z_lens
        self.z_source_ref = z_source_ref

    @abstractmethod
    def efunc(self, z, H0, Om0, k, w0, wa):
        pass

    def omega_rad0(self, H0):
        h = H0 / 100
        return 2.469e-5 * h ** -2.0 * (1.0 + 0.2271 * self.Neff)

    def dark_energy_eos(self, z, w0, wa):
        return w0 + wa * (1 - (1 + z) ** -1)

    def comoving_distance_z1z2(self, z1, z2, H0, Om0, k, w0, wa):
        def integrand(z):
            return 1.0 / self.efunc(z, H0, Om0, k, w0, wa)
        return (self.c / H0) * self._integrate(integrand, z1, z2)

    def comoving_distance(self, z, H0, Om0, k, w0, wa):
        return self.comoving_distance_z1z2(0, z, H0, Om0, k, w0, wa)

    def angular_distance(self, z, H0, Om0, k, w0, wa):
        return (1 / (1 + z)) * self.comoving_distance(z, H0, Om0, k, w0, wa)

    def angular_distance_z1z2(self, z1, z2, H0, Om0, k, w0, wa):
        return (1 / (1 + z2)) * self.comoving_distance_z1z2(z1, z2, H0, Om0, k, w0, wa)

    def luminosity_distance(self, z, H0, Om0, k, w0, wa):
        return (1 + z) * self.comoving_distance(z, H0, Om0, k, w0, wa)

    def lensing_distance(self, z_source, H0, Om0, k, w0, wa):
        d_ls = self.angular_distance_z1z2(self.z_lens, z_source, H0, Om0, k, w0, wa)
        d_s = self.angular_distance(z_source, H0, Om0, k, w0, wa)
        return d_ls / d_s

    def deflection_ratio(self, z_source, H0, Om0, k, w0, wa):
        d_lensing = self.lensing_distance(z_source, H0, Om0, k, w0, wa)
        d_ref = self.lensing_distance(self.z_source_ref, H0, Om0, k, w0, wa)
        return d_lensing / d_ref

    @staticmethod
    @abstractmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        pass
