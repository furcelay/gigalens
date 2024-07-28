import jax.numpy as jnp


class Cosmo:
    neff0 = 3.04  # number of relativistic species
    c = 299792  # km/s #speed of light
    kmtoMpc = 3.2408e-20

    def __init__(self, H0=70, omega_mat0=0.3, wde=-1.0, k=0.0):
        self.h = H0 / 100
        self.H0 = H0
        self.H0s = self.H0 * self.kmtoMpc
        self.omega_mat0 = omega_mat0
        self.wde = wde  # Eq of state dark energy
        self.c_over_H0 = self.c * self.kmtoMpc / self.H0s  # c/H0
        self.omega_k0 = - k / self.H0 ** 2  # curvature density parameter
        self.omega_de0 = (1.0 - self.omega_mat0 - self.omega_rad0 - self.omega_k0)

    @property
    def omega_rad0(self):
        return 2.469e-5 * self.h**-2.0 * (1.0 + 0.2271 * self.neff0)

    def Ez_model(self, z):
        """
        dimensionless Friedmann equation
        """
        matter = self.omega_mat0 * (1 + z)**3
        radiation = self.omega_rad0 * (1 + z)**4
        dark_energy = self.dark_energy_eos(z)
        curvature = self.omega_k0 * (1 + z)**2

        E = jnp.sqrt(matter + radiation + dark_energy + curvature)
        return E

    def dark_energy_eos(self, z):
        exponent = 3 * (1 + self.wde)
        return self.omega_de0 * (1 + z) ** exponent

    # this function computes the theoretical Dth for lens analysis
    def lens_distance(self, zl, zs):
        def integrand(z):
            return 1.0 / self.Ez_model(z)
        d_ls = integrate(integrand, zl, zs)
        d_s = integrate(integrand, 0, zs)
        return d_ls / d_s

    def f_model(self, z=0):
        # This function returns the inverse of the dimensionless Friedmann equation
        return 1.0 / self.Ez_model(z)

    def comoving_distance(self, z):
        # This function returns the transverse comoving distance for a flat Universe  HOGG EC.(16)
        return self.c_over_H0 * integrate(self.f_model, 0, z)

    def angular_distance(self, z):
        # This function returns the angular diameter distance to any source at redshift z  HOGG EC.(18)
        return (1 / (1 + z)) * self.comoving_distance(z)

    def angular_distance_z1z2(self, z1, z2):
        # "This function returns the angular diameter distance between two objects"  HOGG EC.(19)
        return (self.c_over_H0 / (1 + z2)) * integrate(self.f_model, z1, z2)

    def luminosity_distance(self, z=0):
        # "This function returns the luminosity distance at redshift z"   HOGG EC.(21)
        return (1 + z) * self.comoving_distance(z)


class LambdaCDM(Cosmo):
    def __init__(self, H0=70.0, omega_mat0=0.3):
        super().__init__(H0=H0, omega_mat0=omega_mat0)


class wCDM(Cosmo):
    def __init__(self, H0=70.0, omega_mat0=0.3, wde=-1.):
        super().__init__(H0=H0, omega_mat0=omega_mat0, wde=wde)


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
