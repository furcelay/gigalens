import functools

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb

import gigalens.jax.simulator as sim
import gigalens.model


class ForwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
            self,
            prior: tfd.Distribution,
            observed_image=None,
            background_rms=None,
            exp_time=None,
            error_map=None,
            centroids_x=None,
            centroids_y=None,
            centroids_errors_x=None,
            centroids_errors_y=None,
            include_pixels=True,
            include_positions=True,
    ):
        super(ForwardProbModel, self).__init__(prior)
        self.include_pixels = include_pixels
        self.include_positions = include_positions

        self.observed_image = None
        self.error_map = None
        self.background_rms = None
        self.exp_time = None
        self.centroids_x = None
        self.centroids_y = None
        self.centroids_errors_x = None
        self.centroids_errors_y = None

        if self.include_pixels:
            self.observed_image = jnp.array(observed_image)
            if error_map is not None:
                self.error_map = jnp.array(error_map)
            else:
                self.background_rms = jnp.float32(background_rms)
                self.exp_time = jnp.float32(exp_time)
        if self.include_positions:
            self.centroids_x = [jnp.array(cx) for cx in centroids_x]
            self.centroids_y = [jnp.array(cy) for cy in centroids_y]
            self.centroids_errors_x = [jnp.array(cex) for cex in centroids_errors_x]
            self.centroids_errors_y = [jnp.array(cey) for cey in centroids_errors_y]
            # self.n_position = 2 * tf.size(tf.concat(self.centroids_x, axis=0), out_type=tf.float32)

        example = prior.sample(seed=random.PRNGKey(0))
        self.pack_bij = tfb.pack_sequence_as(example)
        self.bij = tfb.Chain(
            [
                prior.experimental_default_event_space_bijector(),
                self.pack_bij,
            ]
        )

    @functools.partial(jit, static_argnums=(0, 1))
    def stats_pixels(self, simulator: sim.LensSimulator, params):
        im_sim = simulator.simulate(params)
        if self.error_map is not None:
            err_map = self.error_map
        else:
            err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        obs_img = self.observed_image
        # TODO: check if better than current
        # log_like = tfd.Independent(
        #     tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        # ).log_prob(self.observed_image)
        chi2 = jnp.sum(((im_sim - obs_img) / err_map) ** 2 * simulator.img_region, axis=(-2, -1))
        normalization = jnp.sum(jnp.log(2 * np.pi * err_map ** 2) * simulator.img_region, axis=(-2, -1))
        log_like = -1 / 2 * (chi2 + normalization)
        red_chi2 = chi2 / jnp.count_nonzero(simulator.img_region)
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def stats_positions(self, simulator: sim.LensSimulator, params):
        chi2 = 0.
        log_like = 0.
        for cx, cy, cex, cey in zip(self.centroids_x, self.centroids_y,
                                    self.centroids_errors_x, self.centroids_errors_y):
            # TODO: see if need to batch centroids or add dimension
            beta_centroids = tf.stack(simulator.beta(cx, cy, params['lens_mass']), axis=0)
            beta_centroids = tf.transpose(beta_centroids, (2, 0, 1))  # batch size, xy, images
            beta_barycentre = tf.math.reduce_mean(beta_centroids, axis=2, keepdims=True)
            beta_barycentre = tf.repeat(beta_barycentre, beta_centroids.shape[2], axis=2)

            if self.use_magnification:
                magnifications = simulator.magnification(cx, cy, params['lens_mass'])
            else:
                magnifications = tf.ones_like(cx, dtype=tf.float32)
            magnifications = tf.transpose(magnifications, (1, 0))  # batch size, images

            err_map = tf.stack([cex / magnifications, cey / magnifications],
                               axis=1)  # batch size, xy, images
            chi2_i = tf.reduce_sum(((beta_centroids - beta_barycentre) / err_map) ** 2, axis=(-2, -1))
            normalization_i = tf.reduce_sum(tf.math.log(2 * np.pi * err_map ** 2), axis=(-2, -1))
            log_like += -1/2 * (chi2_i + normalization_i)
            chi2 += chi2_i
        red_chi2 = chi2 / self.n_position
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        log_like = tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(self.observed_image)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, jnp.mean(
            ((im_sim - self.observed_image) / err_map) ** 2, axis=(-2, -1)
        )


class BackwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
            self, prior: tfd.Distribution, observed_image, background_rms, exp_time
    ):
        super(BackwardProbModel, self).__init__(prior)
        err_map = jnp.sqrt(
            background_rms ** 2 + jnp.clip(observed_image, 0, np.inf) / exp_time
        )
        self.observed_dist = tfd.Independent(
            tfd.Normal(observed_image, err_map), reinterpreted_batch_ndims=2
        )
        self.observed_image = jnp.array(observed_image)
        self.err_map = jnp.array(err_map)
        example = prior.sample(seed=random.PRNGKey(0))
        self.pack_bij = tfb.pack_sequence_as(example)
        self.bij = tfb.Chain(
            [
                prior.experimental_default_event_space_bijector(),
                self.pack_bij,
            ]
        )

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(x, self.observed_image, self.err_map)
        log_like = self.observed_dist.log_prob(im_sim)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, jnp.mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
