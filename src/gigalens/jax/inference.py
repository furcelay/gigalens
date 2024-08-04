import functools

import jax.random
import optax
import tensorflow_probability.substrates.jax as tfp
import time
from jax import jit, pmap
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange

import gigalens.inference
import gigalens.jax.simulator as sim
import gigalens.model


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )

        event_size = jnp.array(0)
        if self.prob_model.include_pixels:
            event_size += jnp.count_nonzero(lens_sim.img_region)
        if self.prob_model.include_positions:
            event_size += self.prob_model.n_position

        seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=seed)
            if start is None
            else start
        )
        params = self.prob_model.bij.inverse(start)

        opt_state = optimizer.init(params)

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / event_size, chisq

        loss_and_grad = jax.pmap(jax.value_and_grad(loss, has_aux=True))

        def update(params, opt_state):
            splt_params = jnp.array(jnp.split(params, dev_cnt, axis=0))
            (_, chisq), grads = loss_and_grad(splt_params)
            grads = jnp.concatenate(grads, axis=0)
            chisq = jnp.concatenate(chisq, axis=0)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return chisq, new_params, opt_state

        with trange(num_steps) as pbar:
            for _ in pbar:
                loss, params, opt_state = update(params, opt_state)
                pbar.set_description(
                    f"Chi-squared: {float(jnp.nanmin(loss)):.3f}"
                )
        log_prob, chi_sq = self.prob_model.log_prob(lens_sim, params)
        best_z = params[jnp.nanargmax(log_prob)]
        best_x = self.prob_model.bij.forward([best_z])
        return best_x, chi_sq[jnp.nanargmax(log_prob)]

    def SVI(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt,
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * dev_cnt), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            z = qz.sample(n_vi // dev_cnt, seed=seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))

        @functools.partial(pmap, axis_name="num_devices")
        def get_update(qz_params, seed):
            val, grad = elbo_and_grad(qz_params, seed)
            return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
                grad, axis_name="num_devices"
            )

        opt_state = optimizer.init(replicated_params)
        loss_hist = []
        with trange(num_steps) as pbar:
            for step in pbar:
                loss, (grads,) = get_update(replicated_params, seeds)
                loss = float(jnp.mean(loss))
                seeds = jax.random.split(seeds[0], dev_cnt)
                updates, opt_state = optimizer.update(grads, opt_state)
                replicated_params = optax.apply_updates(replicated_params, updates)
                pbar.set_description(f"ELBO: {loss:.3f}")
                loss_hist.append(loss)
        mean = replicated_params[0, :n_params]
        cov = cov_bij.forward(replicated_params[0, n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist

    def HMC(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                trace_fn=lambda _, pkr: None,
                seed=seed,
                kernel=mc_kernel,
            )

        start = time.time()
        ret = run_chain(seeds)
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        return ret

    def SMC(self,
            start=None,
            num_particles=1000,
            num_ensembles=1,
            num_leapfrog_steps=10,
            post_sampling_steps=100,
            ess_threshold_ratio=0.5,
            max_sampling_per_stage=8,
            sampler='HMC',
            seed=1):
        dev_cnt = jax.device_count()

        # cannot split particles, only ensembles
        if num_ensembles % dev_cnt != 0:
            raise ValueError(f"num_ensembles ({num_ensembles}) must be divisible by device count ({dev_cnt})")
        else:
            num_ensembles = num_ensembles // dev_cnt
        n_smc_samples = num_particles * num_ensembles

        seed_0, seed_1, seed_2 = jax.random.split(jax.random.PRNGKey(seed), 3)
        seeds_1 = jax.random.split(seed_1, dev_cnt)
        seeds_2 = jax.random.split(seed_2, dev_cnt)

        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_smc_samples
        )

        if start is None:
            start = self.prob_model.prior.sample((dev_cnt, num_particles, num_ensembles), seed=seed_0)
            start = self.prob_model.bij.inverse(start)
        else:
            n_dim = start.shape[-1]
            start = jnp.reshape(start, (-1, n_dim))
            start = jax.random.choice(seed_0, start, (dev_cnt, num_particles, num_ensembles), replace=True)
        n_dim = start.shape[-1]

        if sampler == 'HMC':
            make_kernel_fn = tfe.mcmc.gen_make_hmc_kernel_fn(num_leapfrog_steps=num_leapfrog_steps)
            tunning_fn = lambda ns, ls, la: tfe.mcmc.simple_heuristic_tuning(ns, ls, la, optimal_accept=0.651)
        elif sampler == 'RWMH':
            make_kernel_fn = tfe.mcmc.make_rwmh_kernel_fn
            tunning_fn = tfe.mcmc.simple_heuristic_tuning
        else:
            raise ValueError(f"Unknown sampler: {sampler}, must be 'HMC' or 'RWMH'")

        @jit
        def log_like_fn(z):
            z = jnp.reshape(z, (n_smc_samples, -1))
            ll = self.prob_model.log_like(lens_sim, z)
            return jnp.reshape(ll, (num_particles, num_ensembles))

        @jit
        def log_prob_fn(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def sample_smc(start_z, seed_):
            _, samples_, final_kernel_results = tfe.mcmc.sample_sequential_monte_carlo(
                prior_log_prob_fn=self.prob_model.log_prior,
                likelihood_log_prob_fn=log_like_fn,
                current_state=start_z,
                min_num_steps=1,
                max_num_steps=max_sampling_per_stage,
                max_stage=100,
                make_kernel_fn=make_kernel_fn,
                tuning_fn=tunning_fn,
                resample_fn=tfe.mcmc.resample_systematic,
                ess_threshold_ratio=ess_threshold_ratio,
                seed=seed_,
                name="SMC"
            )
            scalings_ = jnp.exp(final_kernel_results.particle_info.log_scalings)
            log_prob_ = final_kernel_results.particle_info.tempered_log_prob

            return samples_, scalings_, log_prob_

        @pmap
        def sample_mcmc(start_z, scalings_, seed_):
            kernel = make_kernel_fn(
                log_prob_fn,
                [start_z],
                scalings_
            )
            samples_, log_prob_ = tfp.mcmc.sample_chain(
                num_results=post_sampling_steps,
                num_burnin_steps=0,
                current_state=start_z,
                kernel=kernel,
                trace_fn=lambda _, pkr: pkr.accepted_results.target_log_prob,
                seed=seed_,
            )
            return samples_, log_prob_

        t = time.time()
        print("starting SMC")
        samples, scalings, log_prob = sample_smc(start, seeds_1)
        samples = jnp.reshape(samples, (dev_cnt, -1, n_dim))
        scalings = jnp.reshape(scalings, (dev_cnt, -1))
        log_prob = jnp.reshape(log_prob, (dev_cnt, -1))
        t_sample = time.time() - t
        print(f'SMC completed, time: {t_sample / 60:.1f} min')
        if post_sampling_steps > 0:
            t = time.time()
            print("starting MCMC sampling")
            samples, log_prob = sample_mcmc(samples, scalings, seeds_2)
            t_sample = time.time() - t
            print(f'MCMC completed, time: {t_sample / 60:.1f} min')
            # shape (dev_cnt, post_sampling_steps, n_particles * n_ensembles / dev_cnt, n_dim)
            # -> (post_sampling_steps, n_particles * n_ensembles, n_dim)
            samples = jnp.swapaxes(samples, 0, 1)
            samples = jnp.reshape(samples, (post_sampling_steps, -1, n_dim))
            log_prob = jnp.swapaxes(log_prob, 0, 1)
            log_prob = jnp.reshape(log_prob, (post_sampling_steps, -1))
        else:
            # shape (dev_cnt, n_particles * n_ensembles / dev_cnt, n_dim)
            # -> (1, n_particles * n_ensembles, n_dim)
            samples = jnp.reshape(samples, (1, -1, n_dim))
            log_prob = jnp.reshape(log_prob, (1, -1))
        map_idx = jnp.unravel_index(jnp.argmax(log_prob), log_prob.shape)
        best_model = jnp.expand_dims(samples[map_idx], axis=0)
        return samples, best_model
