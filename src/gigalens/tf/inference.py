import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange, tqdm

import gigalens.inference
import gigalens.model
import gigalens.tf.simulator


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(self, optimizer, start=None, n_samples=500, num_steps=350, seed=0):
        tf.random.set_seed(seed)
        start = self.prob_model.prior.sample(n_samples) if start is None else start
        trial = tf.Variable(self.prob_model.bij.inverse(start))
        self.prob_model.init_centroids(bs=n_samples)
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model, self.sim_config, bs=n_samples
        )

        event_size = tf.zeros(1)
        if self.prob_model.include_pixels:
            event_size += tf.math.count_nonzero(lens_sim.img_region, out_type=tf.float32)
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
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi,
        )
        self.prob_model.init_centroids(bs=n_vi)

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
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc,
        )
        self.prob_model.init_centroids(bs=n_hmc)

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

    def SMC(self):
        pass