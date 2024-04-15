import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange, tqdm
import time

import gigalens.inference
import gigalens.model
import gigalens.tf.simulator as sim


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(self, optimizer, start=None, n_samples=500, num_steps=350, seed=0):
        tf.random.set_seed(seed)
        start = self.prob_model.prior.sample(n_samples) if start is None else start
        trial = tf.Variable(self.prob_model.bij.inverse(start))
        lens_sim = sim.LensSimulator(
            self.phys_model, self.sim_config, bs=n_samples
        )

        event_size = tf.zeros(1)
        if self.prob_model.include_pixels:
            event_size += tf.cast(tf.math.count_nonzero(lens_sim.img_region), tf.float32)
        if self.prob_model.include_positions:
            event_size += self.prob_model.n_position

        def train_step():
            with tf.GradientTape() as tape:
                log_prob, red_chi2 = self.prob_model.log_prob(lens_sim, trial)
                agg_loss = tf.reduce_mean(-log_prob / event_size)
            gradients = tape.gradient(agg_loss, [trial])
            optimizer.apply_gradients(zip(gradients, [trial]))
            return red_chi2

        with trange(num_steps) as pbar:
            for _ in pbar:
                square_err = train_step()
                pbar.set_description(f"Chi Squared: {np.nanmin(square_err):.4f}")
        return trial

    def SVI(self, optimizer, start_mean, n_vi=250, init_scales=1e-3, num_steps=500, seed=2,
            full_rank=True):
        tf.random.set_seed(seed)
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi,
        )

        start_mean = tf.squeeze(start_mean)

        scale = (
            np.eye(len(start_mean), len(start_mean)).astype(np.float32) * init_scales
            if np.size(init_scales) == 1
            else init_scales
        )

        if full_rank:

            q_z = tfd.MultivariateNormalTriL(
                loc=tf.Variable(start_mean),
                scale_tril=tfp.util.TransformedVariable(
                    scale,
                    tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6),
                    name="stddev",
                ),
            )
        else:
            q_z = tfd.MultivariateNormalDiag(
                loc=tf.Variable(start_mean),
                scale_diag=tfp.util.TransformedVariable(
                    np.diag(scale),
                    tfp.bijectors.Exp(),
                    name="stddev",
                ),
            )

        losses = tfp.vi.fit_surrogate_posterior(
            lambda z: self.prob_model.log_prob(lens_sim, z)[0],
            surrogate_posterior=q_z,
            sample_size=n_vi,
            optimizer=optimizer,
            num_steps=num_steps,
        )

        return q_z, losses

    def HMC(
        self,
        q_z,
        init_eps=0.3,
        init_l=3,
        n_hmc=50,
        num_burnin_steps=250,
        num_results=750,
        max_leapfrog_steps=30,
        adapt_rate=0.05,
        adapt_mode='dual',
        seed=3,
    ):

        # def trace_fn(_, pkr):
        #     return (
        #         pkr.inner_results.inner_results.log_accept_ratio,
        #         pkr.inner_results.inner_results.is_accepted,
        #         pkr.inner_results.inner_results.accepted_results.step_size,
        #         pkr.inner_results.inner_results.accepted_results.num_leapfrog_steps
        #     )

        def tqdm_progress_bar_fn(num_steps):
            return iter(tqdm(range(num_steps), desc="", leave=True))

        tf.random.set_seed(seed)
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc,
        )

        mc_start = q_z.sample(n_hmc)
        cov_estimate = q_z.covariance()

        momentum_distribution = (
            tfe.distributions.MultivariateNormalPrecisionFactorLinearOperator(
                precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                    tf.linalg.cholesky(cov_estimate),
                ),
                precision=tf.linalg.LinearOperatorFullMatrix(cov_estimate),
            )
        )

        num_adaptation_steps = int(num_burnin_steps * 0.8)
        start = tf.identity(mc_start)

        mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=lambda z: self.prob_model.log_prob(lens_sim, z)[0],
            momentum_distribution=momentum_distribution,
            step_size=init_eps,
            num_leapfrog_steps=init_l,
        )

        mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
            mc_kernel,
            num_adaptation_steps=num_adaptation_steps,
            max_leapfrog_steps=max_leapfrog_steps,
        )
        if adapt_mode == 'dual':
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )
        elif adapt_mode == 'simple':
            mc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps, adaptation_rate=adapt_rate,
            )
        else:
            raise ValueError(f"Invalid adaptation mode {adapt_mode}, the options are 'simple' and 'dual'")

        pbar = tfe.mcmc.ProgressBarReducer(
            num_results + num_burnin_steps - 1, progress_bar_fn=tqdm_progress_bar_fn
        )
        mc_kernel = tfe.mcmc.WithReductions(mc_kernel, pbar)

        def run_chain():

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                trace_fn=None,
                seed=seed,
            )

        return run_chain()

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

        n_smc_samples = num_particles * num_ensembles

        if start is None:
            start = self.prob_model.prior.sample((num_particles, num_ensembles), seed=seed)
            start = self.prob_model.bij.inverse(start)
        else:
            start_size = tf.math.reduce_prod(start.shape[:-1])
            select = tf.random.categorical(tf.zeros((1, start_size)), n_smc_samples, dtype=tf.int32)
            start = tf.gather(tf.reshape(start, (n_smc_samples, -1)), select[0])
            start = tf.reshape(start, (num_particles, num_ensembles, -1))
        n_dim = start.shape[-1]

        lens_sim = sim.LensSimulator(self.phys_model, self.sim_config, bs=n_smc_samples)

        @tf.function
        def log_like_fn(z):
            z = tf.reshape(z, (n_smc_samples, -1))
            ll = self.prob_model.log_like(lens_sim, z)
            return tf.reshape(ll, (num_particles, num_ensembles))

        @tf.function
        def log_prob_fn(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        tfe.mcmc.sample_sequential_monte_carlo.__globals__['PRINT_DEBUG'] = True

        def sample_smc(start_z):
            if sampler == 'HMC':
                make_kernel_fn = tfe.mcmc.gen_make_hmc_kernel_fn(num_leapfrog_steps=num_leapfrog_steps)
                tunning_fn = lambda ns, ls, la: tfe.mcmc.simple_heuristic_tuning(ns, ls, la, optimal_accept=0.651)
            elif sampler == 'RWMH':
                make_kernel_fn = tfe.mcmc.make_rwmh_kernel_fn
                tunning_fn = tfe.mcmc.simple_heuristic_tuning
            else:
                raise ValueError(f"Unknown sampler: {sampler}, must be 'HMC' or 'RWMH'")

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
                seed=seed,
                name="SMC"
            )
            scalings = tf.math.exp(final_kernel_results.particle_info.log_scalings)

            kernel = make_kernel_fn(
                log_prob_fn,
                [tf.reshape(samples_, (-1, n_dim))],
                tf.reshape(scalings, (-1,)))

            return samples_, kernel

        t = time.time()
        print("starting SMC")
        samples, kernel = sample_smc(start)
        t_sample = time.time() - t
        print(f'SMC completed, time: {t_sample / 60:.1f} min')
        if post_sampling_steps > 0:
            t = time.time()
            print("starting MCMC sampling")
            samples = tfp.mcmc.sample_chain(
                num_results=post_sampling_steps,
                num_burnin_steps=0,
                current_state=tf.reshape(samples, (-1, n_dim)),
                kernel=kernel,
                trace_fn=None,
                seed=seed,
            )
            t_sample = time.time() - t
            print(f'SMC completed, time: {t_sample / 60:.1f} min')
        return samples
