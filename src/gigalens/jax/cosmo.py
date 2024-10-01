import jax.numpy as jnp
import jax
from gigalens.cosmo import CosmoBase


class Cosmo(CosmoBase):
    _name = "dPIS"
    _params = ['H0', 'Om0', 'k', 'w0', 'wa']

    def __init__(self, z_lens, z_source_ref=10.0):
        super(Cosmo, self).__init__(z_lens, z_source_ref)
        self.z_lens = jnp.array([z_lens])
        self.z_source_ref = jnp.array([z_source_ref])

    def efunc(self, z, H0, Om0, k, w0, wa):
        """
        dimensionless Friedmann equation
        """
        matter = Om0 * (1 + z) ** 3
        Or0 = self.omega_rad0(H0)
        relativistic = Or0 * (1 + z)**4
        Ok0 = - k / H0 ** 2
        curvature = Ok0 * (1 + z)**2
        Ode0 = (1.0 - Om0 - Or0 - Ok0)
        w_de = self.dark_energy_eos(z, w0, wa)
        dark_energy = Ode0 * (1 + z) ** (3 * (1 + w_de))

        E = jnp.sqrt(matter + relativistic + dark_energy + curvature)
        return E

    @staticmethod
    def _integrate(func, z_min, z_max, n_grid=1000):
        z = jnp.linspace(z_min, z_max, n_grid)
        f = func(z)
        integrated = jax.scipy.integrate.trapezoid(f, z, axis=0)
        return integrated
