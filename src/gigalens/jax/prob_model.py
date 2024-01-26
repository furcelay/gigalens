import functools

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_flatten
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
        super(ForwardProbModel, self).__init__(prior, include_pixels, include_positions)

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
            self.n_position = 2 * jnp.size(jnp.concatenate(self.centroids_x, axis=0))

        example = prior.sample(seed=random.PRNGKey(0))
        size = int(jnp.size(tree_flatten(example)[0]))
        self.pack_bij = tfb.Chain(
            [
                tfb.pack_sequence_as(example),
                tfb.Split(size),
                tfb.Reshape(event_shape_out=(-1,), event_shape_in=(size, -1)),
                tfb.Transpose(perm=(1, 0)),
            ]
        )
        self.unconstraining_bij = prior.experimental_default_event_space_bijector()
        self.bij = tfb.Chain([self.unconstraining_bij, self.pack_bij])

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
        for cx, cy, cex, cey in zip(self.centroids_x_batch, self.centroids_y_batch,
                                    self.centroids_errors_x, self.centroids_errors_y):
            # TODO: see if need to batch centroids or add dimension
            beta_centroids = jnp.stack(simulator.beta(cx, cy, params['lens_mass']), axis=0)
            beta_centroids = jnp.transpose(beta_centroids, (2, 0, 1))  # batch size, xy, images
            beta_barycentre = jnp.mean(beta_centroids, axis=2, keepdims=True)
            beta_barycentre = jnp.repeat(beta_barycentre, beta_centroids.shape[2], axis=2)

            magnifications = simulator.magnification(cx, cy, params['lens_mass'])
            magnifications = jnp.transpose(magnifications, (1, 0))  # batch size, images

            err_map = jnp.stack([cex / magnifications, cey / magnifications],
                                axis=1)  # batch size, xy, images
            chi2_i = jnp.sum(((beta_centroids - beta_barycentre) / err_map) ** 2, axis=(-2, -1))
            normalization_i = jnp.sum(jnp.log(2 * np.pi * err_map ** 2), axis=(-2, -1))
            log_like += -1/2 * (chi2_i + normalization_i)
            chi2 += chi2_i
        red_chi2 = chi2 / self.n_position
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        n_chi = 0

        x = self.bij.forward(z)

        if self.include_pixels:
            log_like_pix, red_chi2_pix = self.stats_pixels(simulator, x)
            log_like += log_like_pix
            red_chi2 += red_chi2_pix
            n_chi += 1
        if self.include_positions:
            log_like_pos, red_chi2_pos = self.stats_positions(simulator, x)
            log_like += log_like_pos
            red_chi2 += red_chi2_pos
            n_chi += 1
        red_chi2 /= n_chi

        log_prior = self.prior.log_prob(
            x
        ) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_like(self, simulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        x = self.bij.forward(z)

        if self.include_pixels:
            log_like_pix, _ = self.stats_pixels(simulator, x)
            log_like += log_like_pix
        if self.include_positions:
            log_like_pos, _ = self.stats_positions(simulator, x)
            log_like += log_like_pos
        return log_like

    def log_prior(self, z):
        x = self.bij.forward(z)
        return self.prior.log_prob(x) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))

    def init_centroids(self, bs):
        if self.include_positions:
            self.centroids_x_batch = [jnp.array(
                jnp.repeat(cx[..., jnp.newaxis], bs, axis=-1)) for cx in self.centroids_x]
            self.centroids_y_batch = [jnp.array(
                jnp.repeat(cy[..., jnp.newaxis], bs, axis=-1)) for cy in self.centroids_y]


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
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, jnp.mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
