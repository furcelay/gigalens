from corner import corner
import matplotlib.pyplot as plt
import matplotlib as mpl
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
import numpy as np


def flatten_samples(samples, components=('lens_mass')):
    flat_samples = []
    labels = []
    for k0 in components:
        ds = samples.get(k0, {})
        for k1 in sorted(ds.keys()):
            for k2 in sorted(ds[k1].keys()):
                # if k0 == 'source_light' and k2 != 'deflection_ratio':
                #     continue
                # if k0 == 'lens_light':
                #     continue
                flat_samples.append(np.asarray(samples[k0][k1][k2].flatten()))
                labels.append(k2)
    return np.stack(flat_samples), labels


def corner_plot(samples, sigma_levels=np.array([1., 2., 3.])):
    corner_samples, labels = flatten_samples(samples)
    
    figure = corner(corner_samples.T, bins=20, range=np.ones(len(labels)) * 0.999,
                    plot_datapoints=False, plot_density=False,
                    fill_contours=True, show_titles=True,
                    quantiles=[0.16, 0.50, 0.84],
                    levels=(1.0 - np.exp(-0.5 * sigma_levels ** 2)),
                    labels=labels
                    )
    return figure


def model_plot(model_params, image, phys_model, sim_config, exp_time, bkg_rms):

    sim_config_full = SimulatorConfig(
        delta_pix=sim_config.delta_pix,
        num_pix=sim_config.num_pix,
        kernel=sim_config.kernel
    )

    lens_sim = LensSimulator(
                    phys_model,
                    sim_config_full,
                    bs=1,
            )

    afmhot_w = mpl.colormaps['afmhot']
    newcolors = afmhot_w(np.linspace(0, 1, 256))
    newcolors[:1, :] = [1, 1, 1, 1]
    afmhot_w = mpl.colors.ListedColormap(newcolors)

    coolwarm_w = mpl.colormaps['coolwarm']
    newcolors = coolwarm_w(np.linspace(0, 1, 256))
    newcolors[127:129, :] = [1, 1, 1, 1]
    coolwarm_w = mpl.colors.ListedColormap(newcolors)

    simulated = lens_sim.simulate(model_params)
    mask = sim_config.pix_region
    pix_scale = sim_config.delta_pix

    fig, ax = plt.subplots(3, 1, figsize=(4, 10), layout='compressed')

    ax[0].imshow(image, norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=1.5), cmap='afmhot', origin="lower", interpolation='none')
    ax[0].xaxis.set_ticklabels([])
    ax[0].yaxis.set_ticklabels([])

    ax[0].arrow(5, 9, 5 / pix_scale, 0,
            head_width=0, head_length=0,
            fc='w', ec='w', width=0.5)
    ax[0].text(10, 11, "5''",
            color='w', fontsize=14)

    ax[0].text(5, image.shape[0] - 50, "Data",
            color='w', fontsize=14)

    im1 = ax[1].imshow(simulated, norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=1.5), cmap='afmhot', origin="lower")
    ax[1].xaxis.set_ticklabels([])
    ax[1].yaxis.set_ticklabels([])

    ax[1].arrow(5, 9, 5 / pix_scale, 0,
            head_width=0, head_length=0,
            fc='w', ec='w', width=0.5)
    ax[1].text(10, 11, "5''",
            color='w', fontsize=14)

    ax[1].text(5, image.shape[0] - 50, "Model",
            color='w', fontsize=14)

    resid = image - simulated
    err_map = np.sqrt(simulated / exp_time + bkg_rms**2)
    im2 = ax[2].imshow(resid/err_map * mask, cmap=coolwarm_w, interpolation='none', vmin=-3, vmax=3, origin="lower")
    ax[2].xaxis.set_ticklabels([])
    ax[2].yaxis.set_ticklabels([])

    ax[2].arrow(5, 9, 5 / pix_scale, 0,
            head_width=0, head_length=0,
            fc='k', ec='k', width=0.5)
    ax[2].text(15, 11, "5''",
            color='k', fontsize=14)

    ax[2].text(5, image.shape[0] - 50, "Normalized Residual",
            color='k', fontsize=14)

    cbar1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal', pad=0.02, ticks=[0, 0.5, 1.0, 1.5])
    # cbar1.set_label(r'flux  $[nMgy]$')

    cbar2 = fig.colorbar(im2, ax=ax[-1], orientation='horizontal', pad=0.02)
    # cbar2.set_label(r'Normalized Residual')

    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.05, hspace=0,
                                wspace=0)

    return fig
