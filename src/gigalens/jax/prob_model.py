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
            mask=None,
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

        self.event_size = jnp.array(0)

        if self.include_pixels:
            self.observed_image = jnp.array(observed_image)
            if mask is not None:
                self.mask = jnp.array(mask, dtype=bool)
            else:
                self.mask = jnp.ones_like(self.observed_image, dtype=bool)
            if error_map is not None:
                self.error_map = jnp.array(error_map)
            else:
                self.background_rms = jnp.float32(background_rms)
                self.exp_time = jnp.float32(exp_time)
            self.event_size += jnp.count_nonzero(self.mask)
        if self.include_positions:
            self.centroids_x = [jnp.expand_dims(jnp.array(cx), -1) for cx in centroids_x]
            self.centroids_y = [jnp.expand_dims(jnp.array(cy), -1) for cy in centroids_y]
            self.centroids_errors_x = [jnp.array(cex) for cex in centroids_errors_x]
            self.centroids_errors_y = [jnp.array(cey) for cey in centroids_errors_y]
            self.n_position = 2 * jnp.size(jnp.concatenate(self.centroids_x, axis=0))
            self.event_size += self.n_position

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
        chi2 = jnp.sum(((im_sim - obs_img) / err_map) ** 2 * self.mask, axis=(-2, -1))
        normalization = jnp.sum(jnp.log(2 * np.pi * err_map ** 2) * self.mask, axis=(-2, -1))
        log_like = -1 / 2 * (chi2 + normalization)
        red_chi2 = chi2 / jnp.count_nonzero(self.mask)
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def stats_positions(self, simulator: sim.LensSimulator, params):
        chi2 = 0.
        log_like = 0.
        # TODO: see if need to batch centroids or add dimension
        beta_points, beta_barycentre = simulator.points_beta_barycentre(self.centroids_x,
                                                                        self.centroids_y,
                                                                        params)
        magnifications = simulator.points_magnification(self.centroids_x,
                                                        self.centroids_y,
                                                        params)
        for points, barycentre, cex, cey, mag in zip(beta_points, beta_barycentre,
                                                     self.centroids_errors_x, self.centroids_errors_y,
                                                     magnifications):

            barycentre = jnp.repeat(barycentre, points.shape[2], axis=2)

            mag = jnp.transpose(mag, (1, 0))  # batch size, images

            err_map = jnp.stack([cex / mag, cey / mag],
                               axis=1)  # batch size, xy, images
            chi2_i = jnp.sum(((points - barycentre) / err_map) ** 2, axis=(-2, -1))
            normalization_i = jnp.sum(jnp.log(2 * np.pi * err_map ** 2), axis=(-2, -1))
            log_like += -1/2 * (chi2_i + normalization_i)
            chi2 += chi2_i
        red_chi2 = chi2 / self.n_position
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        n_chi = 0
        z = list(z.T)
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
        ) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_like(self, simulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        z = list(z.T)
        x = self.bij.forward(z)

        if self.include_pixels:
            log_like_pix, _ = self.stats_pixels(simulator, x)
            log_like += log_like_pix
        if self.include_positions:
            log_like_pos, _ = self.stats_positions(simulator, x)
            log_like += log_like_pos
        return log_like

    def log_prior(self, z):
        z = list(z.T)
        x = self.bij.forward(z)
        return self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)


class BackwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
            self,
            prior: tfd.Distribution,
            observed_image=None,
            mask=None,
            background_rms=None,
            exp_time=None,
            error_map=None,
            centroids_x=None,
            centroids_y=None,
            centroids_errors_x=None,
            centroids_errors_y=None,
            include_positions=False,
    ):
        super(BackwardProbModel, self).__init__(prior, include_pixels=True, include_positions=include_positions)

        self.event_size = jnp.array(0)

        self.observed_image = jnp.array(observed_image)
        if mask is not None:
            self.mask = jnp.array(mask, dtype=bool)
        else:
            self.mask = jnp.ones_like(observed_image, dtype=bool)
        if error_map is not None:
            self.error_map = jnp.array(error_map)
        else:
            self.background_rms = jnp.float32(background_rms)
            self.exp_time = jnp.float32(exp_time)
            self.error_map = jnp.sqrt(
                self.background_rms ** 2 + jnp.clip(self.observed_image, 0, np.inf) / self.exp_time
            )
        self.observed_dist = tfd.Independent(
            tfd.Normal(self.observed_image[self.mask], self.error_map[self.mask]), reinterpreted_batch_ndims=1
        )
        self.event_size += jnp.count_nonzero(self.mask)
        if self.include_positions:
            self.centroids_x = [jnp.expand_dims(jnp.array(cx), -1) for cx in centroids_x]
            self.centroids_y = [jnp.expand_dims(jnp.array(cy), -1) for cy in centroids_y]
            self.centroids_errors_x = [jnp.array(cex) for cex in centroids_errors_x]
            self.centroids_errors_y = [jnp.array(cey) for cey in centroids_errors_y]
            self.n_position = 2 * jnp.size(jnp.concatenate(self.centroids_x, axis=0))
            self.event_size += self.n_position

        example = prior.sample(seed=random.PRNGKey(0))
        size = int(jnp.size(tree_flatten(example)[0]))
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
        im_sim = simulator.lstsq_simulate(params, self.observed_image, self.error_map)
        log_like = self.observed_dist.log_prob(im_sim[:, self.mask])
        red_chi2 = jnp.mean(
            ((im_sim - self.observed_image) / self.error_map)[:, self.mask] ** 2, axis=-1
        )
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def stats_positions(self, simulator: sim.LensSimulator, params):
        chi2 = 0.
        log_like = 0.
        beta_points, beta_barycentre = simulator.points_beta_barycentre(self.centroids_x,
                                                                        self.centroids_y,
                                                                        params)
        magnifications = simulator.points_magnification(self.centroids_x,
                                                        self.centroids_y,
                                                        params)
        for points, barycentre, cex, cey, mag in zip(beta_points, beta_barycentre,
                                                     self.centroids_errors_x, self.centroids_errors_y,
                                                     magnifications):
            barycentre = jnp.repeat(barycentre, points.shape[2], axis=2)

            mag = jnp.transpose(mag, (1, 0))  # batch size, images

            err_map = jnp.stack([cex / mag, cey / mag],
                                axis=1)  # batch size, xy, images
            chi2_i = jnp.sum(((points - barycentre) / err_map) ** 2, axis=(-2, -1))
            normalization_i = jnp.sum(jnp.log(2 * np.pi * err_map ** 2), axis=(-2, -1))
            log_like += -1 / 2 * (chi2_i + normalization_i)
            chi2 += chi2_i
        red_chi2 = chi2 / self.n_position
        return log_like, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        n_chi = 0

        z = list(z.T)
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
        ) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, red_chi2

    @functools.partial(jit, static_argnums=(0, 1))
    def log_like(self, simulator, z):
        log_like, red_chi2 = jnp.zeros(z.shape[0]), jnp.zeros(z.shape[0])
        z = list(z.T)
        x = self.bij.forward(z)

        if self.include_pixels:
            log_like_pix, _ = self.stats_pixels(simulator, x)
            log_like += log_like_pix
        if self.include_positions:
            log_like_pos, _ = self.stats_positions(simulator, x)
            log_like += log_like_pos
        return log_like

    def log_prior(self, z):
        z = list(z.T)
        x = self.bij.forward(z)
        return self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
