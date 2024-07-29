import jax.numpy as jnp
import jax
from gigalens.profile import CosmoBase


class Cosmo(CosmoBase):
    _name = "dPIS"
    _params = ['H0', 'Om0', 'k', 'w0']

    Neff = 3.04  # number of relativistic species
    c = 299792.458  # km/s #speed of light

    def __init__(self, z_lens, z_source_ref=10.0):
        super(Cosmo, self).__init__(z_lens, z_source_ref)

    def efunc(self, z, H0, Om0, k, w0):
        """
        dimensionless Friedmann equation
        """
        matter = Om0 * (1 + z) ** 3
        Or0 = self.omega_rad0(H0)
        relativistic = Or0 * (1 + z)**4
        Ok0 = - k / H0 ** 2
        curvature = Ok0 * (1 + z)**2
        Ode0 = (1.0 - Om0 - Or0 - Ok0)
        dark_energy = Ode0 * (1 + z) ** (3 * (1 + w0))

        E = jnp.sqrt(matter + relativistic + dark_energy + curvature)
        return E

    def omega_rad0(self, H0):
        h = H0 / 100
        return 2.469e-5 * h**-2.0 * (1.0 + 0.2271 * self.Neff)

    def comoving_distance_z1z2(self, z1, z2, H0, Om0, k, w0):
        def integrand(z):
            return 1.0 / self.efunc(z, H0, Om0, k, w0)
        # This function returns the transverse comoving distance for a flat Universe  HOGG EC.(16)
        return (self.c/ H0) * integrate(integrand, z1, z2)

    def comoving_distance(self, z, H0, Om0, k, w0):
        return self.comoving_distance_z1z2(0, z, H0, Om0, k, w0)

    def angular_distance(self, z, H0, Om0, k, w0):
        # This function returns the angular diameter distance to any source at redshift z  HOGG EC.(18)
        return (1 / (1 + z)) * self.comoving_distance(z, H0, Om0, k, w0)

    def angular_distance_z1z2(self, z1, z2, H0, Om0, k, w0):
        return (1 / (1 + z2)) * self.comoving_distance_z1z2(z1, z2, H0, Om0, k, w0)

    def luminosity_distance(self, z, H0, Om0, k, w0):
        # "This function returns the luminosity distance at redshift z"   HOGG EC.(21)
        return (1 + z) * self.comoving_distance(z, H0, Om0, k, w0)

    def lensing_distance(self, z_source, H0, Om0, k, w0):
        d_ls = self.angular_distance_z1z2(self.z_lens, z_source, H0, Om0, k, w0)
        d_s = self.angular_distance(z_source, H0, Om0, k, w0)
        return d_ls / d_s

    def deflection_ratio(self, z_source, H0, Om0, k, w0):
        d_lensing = self.lensing_distance(z_source, H0, Om0, k, w0)
        d_ref = self.lensing_distance(self.z_source_ref, H0, Om0, k, w0)
        return d_lensing / d_ref


def integrate(func, z_min, z_max, n_grid=1000):
    z = jnp.linspace(z_min, z_max, n_grid)
    f = func(z)
    integrated = jax.scipy.integrate.trapezoid(f, z, axis=0)
    return integrated
