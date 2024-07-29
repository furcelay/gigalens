from astropy.cosmology import wCDM
from gigalens.jax.cosmo import Cosmo
import numpy as np
import numpy.testing as npt
import unittest


class TestCosmo(unittest.TestCase):
    def setUp(self):
        self.cosmo_params = {
            'H0': 70,
            'Om0': 0.3,
            'k': 0.0,
            'w0': -1
        }
        self.cosmo = Cosmo(z_lens=0.5, z_source_ref=2.0)
        Ok0 = - self.cosmo_params['k'] / self.cosmo_params['H0'] ** 2
        Or0 = self.cosmo.omega_rad0(self.cosmo_params['H0'])
        Ode0 = (1.0 - self.cosmo_params['Om0'] - Or0 - Ok0)
        self.cosmo_ref = wCDM(
            H0=self.cosmo_params['H0'],
            Om0=self.cosmo_params['Om0'],
            Ode0=Ode0,
            w0=self.cosmo_params['w0'],
            Tcmb0=2.725
        )

    def test_neff(self):
        self.assertEqual(self.cosmo.Neff, self.cosmo_ref.Neff)

    def test_Or(self):
        z = 0.0
        Or0_ref = self.cosmo_ref._Ogamma0 + self.cosmo_ref._Onu0
        Or0 = self.cosmo.omega_rad0(self.cosmo_params['H0'])
        npt.assert_almost_equal(Or0, Or0_ref, decimal=5)

    def test_efunc(self):
        z = np.linspace(0.001, 10, 100)
        e_ref = self.cosmo_ref.efunc(z)
        e = self.cosmo.efunc(z, **self.cosmo_params)
        npt.assert_allclose(e, e_ref, rtol=1e-5)

    def test_comoving_distance(self):
        z = np.linspace(0.001, 10, 100)
        d_ref = self.cosmo_ref.comoving_distance(z).value
        d = self.cosmo.comoving_distance(z, **self.cosmo_params)
        npt.assert_allclose(d, d_ref, rtol=1e-5)

    def test_luminosity_distance(self):
        z = np.linspace(0.001, 10, 100)
        d_ref = self.cosmo_ref.luminosity_distance(z).value
        d = self.cosmo.luminosity_distance(z, **self.cosmo_params)
        npt.assert_allclose(d, d_ref, rtol=1e-5)

    def test_angular_distance(self):
        z = np.linspace(0.001, 10, 100)
        d_ref = self.cosmo_ref.angular_diameter_distance(z).value
        d = self.cosmo.angular_distance(z, **self.cosmo_params)
        npt.assert_allclose(d, d_ref, rtol=1e-5)

    def test_angular_distance_z1z2(self):
        z1 = 0.1
        z2 = 0.5
        d_ref = self.cosmo_ref.angular_diameter_distance_z1z2(z1, z2).value
        d = self.cosmo.angular_distance_z1z2(z1, z2, **self.cosmo_params)
        npt.assert_allclose(d, d_ref, rtol=1e-5)

    def test_lensing_distance(self):
        z_source = np.linspace(0.6, 10, 100)
        d_ls_ref = self.cosmo_ref.angular_diameter_distance_z1z2(self.cosmo.z_lens, z_source).value
        d_s_ref = self.cosmo_ref.angular_diameter_distance(z_source).value
        d_ref = d_ls_ref / d_s_ref
        d = self.cosmo.lensing_distance(z_source, **self.cosmo_params)
        npt.assert_allclose(d, d_ref, rtol=1e-5)

    def test_multi_param(self):
        z_source = np.array([0.7])
        cosmo_params = {
            'H0': 70,
            'Om0': np.array([0.3, 0.4, 0.5]),
            'k': 0.0,
            'w0': -1
        }
        d = self.cosmo.lensing_distance(z_source, **cosmo_params)
        assert d.shape == (3,)


if __name__ == '__main__':
    unittest.main()
