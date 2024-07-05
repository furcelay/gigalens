from tensorflow_probability.substrates.jax import distributions as tfd
from gigalens.jax.profiles import mass, light
from gigalens.jax.prior import Prior, make_prior_and_model
from astropy.table import Table
import pickle
import json


close_subhalo_tbl = Table.read('model_data/close_subhalos.fits')

with open(f'models/best_model_foreground_new.pkl', 'rb') as f:
    best_x_foreground = pickle.load(f)

model_globals = json.load(open('model_data/model_globals.json', 'r'))
deflect_ratio_sources = model_globals['deflect_ratio_sources']

halo_c_err = 5

halo_model = Prior(mass.epl.EPL(),
                   dict(
                       center_x = tfd.TruncatedNormal(0, halo_c_err / 2, -halo_c_err, halo_c_err),
                       center_y = tfd.TruncatedNormal(0, halo_c_err / 2, -halo_c_err, halo_c_err),
                       e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                       theta_E = tfd.Uniform(11., 15.),
                       gamma = tfd.Uniform(1.1, 2.5),
                   ))

ld_tbl = close_subhalo_tbl[close_subhalo_tbl['name'] == 'Ld'][0]
ld_model = Prior(
                mass.epl.EPL(),  # mass.piemd.DPIE(),
                # mass.sie.SIE(),
                   dict(
                       center_x = ld_tbl['center_x'],
                       center_y = ld_tbl['center_y'],
                       e1 = ld_tbl['e1'],
                       e2 = ld_tbl['e2'],  
                       theta_E = tfd.Uniform(0.1, 3.),
                       gamma = tfd.TruncatedNormal(2.0, 0.05, 1.3, 2.8),
                   ))

shear_model = Prior(mass.shear.Shear(),
                    dict(
                        gamma1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                        gamma2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                    )
)


def num_amps_shapelets(n):
     return (n + 1) * (n + 2) // 2


source_model = [
    # source 1
    Prior(
        light.sersic_shapelets.SersicShapelets(8),
        dict(
            deflection_ratio = deflect_ratio_sources[0],
            center_x = tfd.Normal(0., halo_c_err),
            center_y = tfd.Normal(0., halo_c_err),
            e1 = tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
            e2 = tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
            R_sersic = tfd.TruncatedNormal(1.5, 0.5, 0.05, 2.5),
            n_sersic = tfd.Uniform(0.5, 5),
            Ie = tfd.TruncatedNormal(10.0, 2.0, 0.5, 25.),
            beta = tfd.Uniform(0.03, 0.1),
        ) |
        {f"amp{i:02d}": tfd.Normal(0., 10.) for i in range(num_amps_shapelets(8))}
    ),
    # source 3
    Prior(
        light.sersic_shapelets.SersicShapelets(12),
        dict(
            deflection_ratio = deflect_ratio_sources[2],
            center_x = tfd.Normal(0., halo_c_err),
            center_y = tfd.Normal(0., halo_c_err),
            e1 = tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
            e2 = tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
            R_sersic = tfd.TruncatedNormal(1.5, 0.5, 0.05, 2.5),
            n_sersic = tfd.Uniform(0.5, 5),
            Ie = tfd.TruncatedNormal(10.0, 2.0, 0.5, 25.),
            beta = 0.2,
        )|
        {f"amp{i:02d}": tfd.Normal(0., 10.) for i in range(num_amps_shapelets(12))}
    ),
    # source 4
    Prior(
        light.sersic_shapelets.SersicShapelets(10),
        dict(
            deflection_ratio = deflect_ratio_sources[3],
            center_x = tfd.Normal(0., halo_c_err),
            center_y = tfd.Normal(0., halo_c_err),
            e1 = tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
            e2 = tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
            R_sersic = tfd.TruncatedNormal(1.5, 0.5, 0.05, 2.5),
            n_sersic = tfd.Uniform(0.5, 5),
            Ie = tfd.TruncatedNormal(10.0, 2.0, 0.5, 25.),
            beta = 0.2,
        )|
        {f"amp{i:02d}": tfd.Normal(0., 10.) for i in range(num_amps_shapelets(10))}
    ),
    # source 5
    Prior(
        light.sersic.SersicEllipse(),
        dict(
            deflection_ratio = deflect_ratio_sources[4],
            center_x = tfd.Normal(0., halo_c_err),
            center_y = tfd.Normal(0., halo_c_err),
            e1 = tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
            e2 = tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
            R_sersic =  tfd.TruncatedNormal(0.1, 0.1, 0.05, 0.3),
            n_sersic = tfd.Uniform(0.5, 5),
            Ie = tfd.TruncatedNormal(5.0, 2.0, 0.5, 15.),
        )
    ),
]


foreground_gals = []

for i in range(len(best_x_foreground['lens_light'])):
    l = best_x_foreground['lens_light'][str(i)]
    foreground_gals.append(
        Prior(
            light.sersic.SersicEllipse(),
                dict(
                    center_x = l['center_x'][0],
                    center_y = l['center_y'][0],
                    e1 = l['e1'][0],
                    e2 = l['e2'][0],
                    R_sersic = l['R_sersic'][0],
                    n_sersic = l['n_sersic'][0],
                    Ie = l['Ie'][0],
                    )
                )
            )


prior, phys_model = make_prior_and_model(
    lenses=[
        halo_model,
        ld_model,
        shear_model,
    ],
    sources=[
        *source_model,
    ],
    foreground=[
        *foreground_gals
    ]
)