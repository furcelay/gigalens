import jax.numpy as jnp
from gigalens.profile import CosmoBase


class Cosmo(CosmoBase):
    _name = "dPIS"
    _params = ['theta_E', 'r_core', 'r_cut', 'center_x', 'center_y']

    neff0 = 3.04  # number of relativistic species
    c = 299792  # km/s #speed of light
    kmtoMpc = 3.2408e-20

    def Ez_model(self, z, H0, omega_mat0, k, wde):
        """
        dimensionless Friedmann equation
        """
        matter = omega_mat0 * (1 + z)**3
        omega_rad0 = self.omega_rad0(H0/100)
        radiation = omega_rad0 * (1 + z)**4
        omega_k0 = - k / H0 ** 2
        curvature = omega_k0 * (1 + z)**2
        omega_de0 = (1.0 - omega_mat0 - omega_rad0 - omega_k0)
        dark_energy = self.dark_energy_eos(z, omega_de0, wde)

        E = jnp.sqrt(matter + radiation + dark_energy + curvature)
        return E

    def omega_rad0(self, h):
        return 2.469e-5 * h**-2.0 * (1.0 + 0.2271 * self.neff0)

    def dark_energy_eos(self, z, omega_de0, wde):
        exponent = 3 * (1 + wde)
        return omega_de0 * (1 + z) ** exponent

    def comoving_distance(self, z, H0, omega_mat0, k, wde):
        def integrand(z):
            return 1.0 / self.Ez_model(z, H0, omega_mat0, k, wde)

        c_over_H0 = self.c * self.kmtoMpc / H0
        # This function returns the transverse comoving distance for a flat Universe  HOGG EC.(16)
        return c_over_H0 * integrate(integrand, 0, z)

    def angular_distance(self, z, H0, omega_mat0, k, wde):
        # This function returns the angular diameter distance to any source at redshift z  HOGG EC.(18)
        return (1 / (1 + z)) * self.comoving_distance(z, H0, omega_mat0, k, wde)

    def angular_distance_z1z2(self, z1, z2, H0, omega_mat0, k, wde):
        def integrand(z):
            return 1.0 / self.Ez_model(z, H0, omega_mat0, k, wde)
        c_over_H0 = self.c * self.kmtoMpc / H0
        # "This function returns the angular diameter distance between two objects"  HOGG EC.(19)
        return (c_over_H0 / (1 + z2)) * integrate(integrand, z1, z2)

    def luminosity_distance(self, z=0):
        # "This function returns the luminosity distance at redshift z"   HOGG EC.(21)
        return (1 + z) * self.comoving_distance(z)

    def lensing_distance(self, z_source, H0, omega_mat0, k, wde):
        def integrand(z):
            return 1.0 / self.Ez_model(z, H0, omega_mat0, k, wde)
        d_ls = integrate(integrand, self.z_lens, z_source)
        d_s = integrate(integrand, 0, z_source)
        return d_ls / d_s

    def deflection_ratio(self, z_source, H0, omega_mat0, k, wde):
        d_lensing = self.lensing_distance(z_source, H0, omega_mat0, k, wde)
        d_ref = self.lensing_distance(self.z_source_ref, H0, omega_mat0, k, wde)
        return d_lensing / d_ref


def integrate(func, z_min, z_max, n_grid=1000):
    z = jnp.linspace(z_min, z_max, n_grid)
    f = func(z)
    integrated = jnp.trapz(f, z)
    return integrated


def log_integrate(func, z_min, z_max, n_grid=1000):
    min_logz = jnp.log(z_min)
    max_logz = jnp.log(z_max)
    dlogz = (max_logz - min_logz) / (n_grid - 1)
    z = jnp.logspace(
        min_logz + dlogz / 2.0,
        max_logz + dlogz / 2.0,
        n_grid,
        base=jnp.e,
    )
    y = func(z)
    return jnp.sum(y * dlogz * z)
