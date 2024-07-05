from time import time
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
import jax
from jax import numpy as jnp
import numpy as np


def prior_limits(prior, k0, k1, k2):
    model = prior.model[k0].model[k1].model[k2]
    try:
        low, high = model.low, model.high
    except AttributeError:
        low, high = -jnp.inf, jnp.inf
    return low, high


def add_model_scatter(start_model, prior, num_samples=100, scatter=0.1, seed=0):
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    start = prior.sample(num_samples, subkey)

    for k0 in start:
        if k0 not in start_model:
            continue
        for k1 in start[k0]:
            if k1 not in start_model[k0]:
                continue
            for k2 in start[k0][k1]:
                if k2 not in start_model[k0][k1]:
                    continue
                key, subkey = jax.random.split(key)
                mu = start_model[k0][k1][k2].flatten()
                std = scatter * jnp.std(start[k0][k1][k2])
                s = jax.random.normal(subkey, shape=(num_samples,)) * std + mu
                low, high = prior_limits(prior, k0, k1, k2)
                start[k0][k1][k2] = jnp.clip(s, low, high)
    return start


def sample_hmc(
    start, 
    lens_sim, 
    prob_model, 
    num_samples,
    num_burnin,
    num_adaptation_steps,
    seed=0):

    cov_estimate = jnp.cov(jnp.reshape(start, (-1, start.shape[-1])).T)
    momentum_distribution = tfd.MultivariateNormalFullCovariance(
                                loc=jnp.zeros(start.shape[-1]),
                                covariance_matrix=jnp.linalg.inv(cov_estimate),
            )

    @jax.pmap
    def run_hmc(start, seed):
        precond_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=lambda x: prob_model.log_prob(lens_sim, x)[0],
            momentum_distribution=momentum_distribution,
            step_size=1.,
            num_leapfrog_steps=10,
        )

        # precond_kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        #     precond_kernel,
        #     num_adaptation_steps=num_adaptation_steps,
        #     max_leapfrog_steps=max_leapfrog_steps,
        # )
        precond_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=precond_kernel, num_adaptation_steps=num_adaptation_steps
        )
        samples_z = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin,
            current_state=start,
            kernel=precond_kernel,
            trace_fn=None,
            seed=seed,
        )
        return samples_z

    t = time()
    seeds = jax.random.split(jax.random.PRNGKey(seed), jax.device_count())
    samples_z = run_hmc(start, seeds)
    t_sample = time() - t
    print(f'time: {t_sample / 60:.1f} min')
    return samples_z


def get_samples_stats(samples_z, prob_model, lens_sim):
    n_dim = samples_z.shape[-1]
    samples_z = samples_z.reshape(-1, lens_sim.bs, n_dim)

    R_conv = tfp.mcmc.potential_scale_reduction(
        samples_z,
        independent_chain_ndims=1
    )

    log_prob = jnp.stack([prob_model.log_prob(lens_sim, samples_z[i])[0] for i in range(samples_z.shape[0])], axis=0)
    map_idx = np.unravel_index(np.argmax(log_prob), log_prob.shape)
    map_estimate = prob_model.bij.forward([samples_z[map_idx]])

    samples_z = jnp.reshape(samples_z, (-1, n_dim))
    median_estimate = [[np.percentile(samples_z[...,i], 50).astype('float32') for i in range(samples_z.shape[-1])]]
    median_estimate = prob_model.bij.forward(median_estimate)

    return map_estimate, median_estimate, R_conv
