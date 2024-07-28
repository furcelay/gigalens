from gigalens.jax.inference import ModellingSequence
from gigalens.jax.prob_model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
import jax
from jax import numpy as jnp
import pickle
from model import prior, phys_model, model_globals
from data import (image, psf, mask, binning, bkg_rms, exp_time, centroids_x, centroids_y, centroids_err)
from sampling import add_model_scatter, sample_hmc, get_samples_stats
from plotting import corner_plot, model_plot


input_model = 'model_s1345_refined_shear'
std_frac = 0.1
num_particles = 10
num_ensembles = jax.device_count()
num_samples_smc = num_particles * num_ensembles
post_sampling_steps = 100

num_samples_hmc = 100
num_burnin = 100
num_adaptation_steps = int(num_burnin * 0.8)

pix_scale = model_globals['pix_scale'] * binning
num_pix = model_globals['n_pix'] // binning

print("Num devices:", num_ensembles)

sim_config = SimulatorConfig(
    delta_pix=pix_scale, 
    num_pix=num_pix, 
    kernel=psf, 
    pix_region=mask
)
prob_model = ForwardProbModel(
    prior,
    image,
    background_rms=bkg_rms,
    exp_time=exp_time,
    centroids_x=centroids_x,
    centroids_y=centroids_y,
    centroids_errors_x=centroids_err,
    centroids_errors_y=centroids_err,
    include_pixels=True,
    include_positions=True
)
fitter = ModellingSequence(
    phys_model,
    prob_model,
    sim_config
)

with open(f"models/best_{input_model}.pkl", 'rb') as f:
    init_model = pickle.load(f)

start = add_model_scatter(init_model, prior, num_samples_smc, scatter=0.1, seed=1)
start = jnp.stack(prob_model.bij.inverse(start), axis=-1)

print("sampling stage 1: SMC + HMC\n")
samples_z = fitter.SMC(
    start=start,
    num_particles=num_particles,
    num_ensembles=num_ensembles,
    num_leapfrog_steps=10,
    post_sampling_steps=post_sampling_steps,
    ess_threshold_ratio=0.5,
    max_sampling_per_stage=8,
    sampler='HMC',
    seed=1
)
print("stage 1 done!\nSummary stats:\n")

lens_sim = LensSimulator(
                phys_model,
                sim_config,
                bs=num_particles,
        )

map_estimate, median_estimate, R_conv = get_samples_stats(samples_z, prob_model, lens_sim)
samples = prob_model.bij.forward(list(samples_z.T))
print(f"R_conv mean {R_conv.mean():1.2e}, max {R_conv.max():1.2e}")

with open(f'samples/samples_SMC_{input_model}.pkl', 'wb') as f:
    pickle.dump(samples, f)

with open(f'samples/median_SMC_{input_model}.pkl', 'wb') as f:
    pickle.dump(median_estimate, f)

with open(f'samples/map_SCM_{input_model}.pkl', 'wb') as f:
    pickle.dump(map_estimate, f)

corner_fig = corner_plot(samples)
corner_fig.savefig(f'samples/corner_SMC_{input_model}.png')

model_fig = model_plot(map_estimate, image, phys_model, sim_config, exp_time, bkg_rms)
model_fig.savefig(f'samples/model_SMC_{input_model}.png')


if num_samples_hmc > 0:
    print("sampling stage 2: HMC preconditioned\n")
    new_samples_z = sample_hmc(samples_z[:,-1], lens_sim, prob_model, num_samples_hmc, num_burnin, num_adaptation_steps,
                               seed=0)
    print("stage 2 done!\nSummary stats:\n")

    map_estimate, median_estimate, R_conv = get_samples_stats(new_samples_z, prob_model, lens_sim)
    samples = prob_model.bij.forward(list(new_samples_z.T))
    print(f"R_conv mean {R_conv.mean():1.2e}, max {R_conv.max():1.2e}")

    with open(f'samples/samples_HMC_{input_model}.pkl', 'wb') as f:
        pickle.dump(samples, f)

    with open(f'samples/median_HMC_{input_model}.pkl', 'wb') as f:
        pickle.dump(median_estimate, f)

    with open(f'samples/map_HMC_{input_model}.pkl', 'wb') as f:
        pickle.dump(map_estimate, f)

    corner_fig = corner_plot(samples)
    corner_fig.savefig(f'samples/corner_HMC_{input_model}.png')

    model_fig = model_plot(map_estimate, image, phys_model, sim_config, exp_time, bkg_rms)
    model_fig.savefig(f'samples/model_HMC_{input_model}.png')

print("done!")
