import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

import gigalens.prob_model
import gigalens.tf.simulator


class ForwardProbModel(gigalens.prob_model.ProbabilisticModel):
    """
    Probabilistic model defined using the simulated image as an estimator for the noise variance map. Linear parameters
    *are not* automatically solved for using least squares.

    Attributes:
        observed_image (:obj:`tf.Tensor` or :obj:`numpy.array`): The observed image.
        background_rms (float): The estimated background Gaussian noise level
        exp_time (float): The exposure time (used for calculating Poisson shot noise)
        pack_bij (:obj:`tfp.bijectors.Bijector`): A bijector that reshapes from a tensor to a structured parameter
            object (i.e., dictionaries of parameter values). Does not change the input parameters whatsoever, it only
            reshapes them.
        unconstraining_bij (:obj:`tfp.bijectors.Bijector`): A bijector that maps from physical parameter space to
            unconstrained parameter space. Outputs an identical structure as the input, only maps values to
            unconstrained space.
        bij (:obj:`tfp.bijectors.Bijector`): The composition of :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` and
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`. The inverse method of ``bij`` will
            unconstrain parameter values and then flatten the entire structure into one tensor.
    """

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
        include_positions=True
    ):
        super(ForwardProbModel, self).__init__(prior, include_pixels, include_positions)

        if self.include_pixels:
            self.observed_image = tf.constant(observed_image, dtype=tf.float32)
            if error_map is not None:
                self.error_map = tf.constant(error_map, dtype=tf.float32)
            else:
                self.background_rms = tf.constant(background_rms, dtype=tf.float32)
                self.exp_time = tf.constant(exp_time, dtype=tf.float32)
        if self.include_positions:
            self.centroids_x = [tf.expand_dims(tf.constant(cx, dtype=tf.float32), -1) for cx in centroids_x]
            self.centroids_y = [tf.expand_dims(tf.constant(cy, dtype=tf.float32), -1) for cy in centroids_y]
            self.centroids_errors_x = [tf.convert_to_tensor(cex, dtype=tf.float32) for cex in centroids_errors_x]
            self.centroids_errors_y = [tf.convert_to_tensor(cey, dtype=tf.float32) for cey in centroids_errors_y]
            self.n_position = 2 * tf.size(tf.concat(self.centroids_x, axis=0), out_type=tf.float32)

        example = prior.sample(seed=0)
        size = int(tf.size(tf.nest.flatten(example)))
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

    @tf.function
    def stats_pixels(self, simulator: gigalens.tf.simulator.LensSimulator, params):
        im_sim = simulator.simulate(params)
        if self.error_map is not None:
            err_map = self.error_map
        else:
            err_map = tf.math.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        obs_img = self.observed_image
        chi2 = tf.reduce_sum(((im_sim - obs_img) / err_map) ** 2 * simulator.img_region, axis=(-2, -1))
        normalization = tf.reduce_sum(tf.math.log(2 * np.pi * err_map ** 2) * simulator.img_region, axis=(-2, -1))
        log_like = -1 / 2 * (chi2 + normalization)
        red_chi2 = chi2 / tf.math.count_nonzero(simulator.img_region, dtype=tf.float32)
        return log_like, red_chi2

    @tf.function
    def stats_positions(self, simulator: gigalens.tf.simulator.LensSimulator, params):
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
            barycentre = tf.repeat(barycentre, points.shape[2], axis=2)

            mag = tf.transpose(mag, (1, 0))  # batch size, images

            err_map = tf.stack([cex / mag, cey / mag],
                               axis=1)  # batch size, xy, images
            chi2_i = tf.reduce_sum(((points - barycentre) / err_map) ** 2, axis=(-2, -1))
            normalization_i = tf.reduce_sum(tf.math.log(2 * np.pi * err_map ** 2), axis=(-2, -1))
            log_like += -1 / 2 * (chi2_i + normalization_i)
            chi2 += chi2_i
        red_chi2 = chi2 / self.n_position
        return log_like, red_chi2

    @tf.function
    def log_prob(self, simulator: gigalens.tf.simulator.LensSimulator, z):
        """Evaluate the reparameterized log posterior density for a batch of unconstrained lens parameters ``z``.
        Simulates the lenses with parameters ``z`` to evaluate the log likelihood. The log prior is calculated by
        calculating the log prior in physical parameter space and adding the log determinant of the Jacobian.

        Notes:
            The log determinant for the default bijector will output a scalar regardless of the batch size of ``z`` due
            to the flattening behavior of :obj:`tfp.bijectors.Split`. To get around this, we use the fact that
            :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` has determinant 1 (it simply reshapes/restructures the
            input), and the determinant comes purely from
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`.

        Args:
            simulator (:obj:`~gigalens.tf.simulator.LensSimulator`): A simulator object that has the same batch size
                as ``z``.
            z (:obj:`tf.Tensor`): Parameters in unconstrained space with shape ``(bs, d)``, where ``bs`` is the batch
                size and ``d`` is the number of parameters per lens.

        Returns:
            The reparameterized log posterior density, with shape ``(bs,)``.
        """
        params = self.bij.forward(z)

        log_like, red_chi2 = tf.zeros(z.shape[0]), tf.zeros(z.shape[0])
        n_chi = 0
        if self.include_pixels:
            log_like_pix, red_chi2_pix = self.stats_pixels(simulator, params)
            log_like += log_like_pix
            red_chi2 += red_chi2_pix
            n_chi += 1
        if self.include_positions:
            log_like_pos, red_chi2_pos = self.stats_positions(simulator, params)
            log_like += log_like_pos
            red_chi2 += red_chi2_pos
            n_chi += 1
        red_chi2 /= n_chi

        log_prior = self.prior.log_prob(
            params
        ) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, red_chi2

    @tf.function
    def log_like(self, simulator, z):
        params = self.bij.forward(z)

        log_like = tf.zeros(z.shape[0])
        if self.include_pixels:
            log_like_pix, _ = self.stats_pixels(simulator, params)
            log_like += log_like_pix
        if self.include_positions:
            log_like_pos, _ = self.stats_positions(simulator, params)
            log_like += log_like_pos
        return log_like

    @tf.function
    def log_prior(self, z):
        params = self.bij.forward(z)
        return self.prior.log_prob(params) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))


class BackwardProbModel(gigalens.prob_model.ProbabilisticModel):  # TODO: update BackwardProbModel
    """
    Probabilistic model defined using the observed image as an estimator for the noise variance map. Linear parameters
    *are* automatically solved for using least squares.

    Attributes:
        observed_image (:obj:`tf.Tensor` or :obj:`numpy.array`): The observed image.
        # background_rms (float): The estimated background Gaussian noise level
        # exp_time (float): The exposure time (used for calculating Poisson shot noise)
        pack_bij (:obj:`tfp.bijectors.Bijector`): A bijector that reshapes from a tensor to a structured parameter
            object (i.e., dictionaries of parameter values). Does not change the input parameters whatsoever, it only
            reshapes them.
        unconstraining_bij (:obj:`tfp.bijectors.Bijector`): A bijector that maps from physical parameter space to
            unconstrained parameter space. Outputs an identical structure as the input, only maps values to
            unconstrained space.
        bij (:obj:`tfp.bijectors.Bijector`): The composition of :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` and
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`. The inverse method of ``bij`` will
            unconstrain parameter values and then flatten the entire structure into one tensor.
    """

    def __init__(
        self, prior: tfd.Distribution, observed_image, background_rms, exp_time
    ):
        super(BackwardProbModel, self).__init__(prior)
        err_map = tf.math.sqrt(
            background_rms ** 2 + tf.clip_by_value(observed_image, 0, np.inf) / exp_time
        )
        self.observed_dist = tfd.Independent(
            tfd.Normal(observed_image, err_map), reinterpreted_batch_ndims=2
        )
        self.observed_image = tf.constant(observed_image, dtype=tf.float32)
        self.err_map = tf.constant(err_map, dtype=tf.float32)
        example = prior.sample(seed=0)
        size = int(tf.size(tf.nest.flatten(example)))
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

    @tf.function
    def log_prob(self, simulator: gigalens.tf.simulator.LensSimulator, z):
        """Evaluate the reparameterized log posterior density for a batch of unconstrained lens parameters ``z``.
        Simulates the lenses with parameters ``z`` (using least squares to solve for linear light parameters) to
        evaluate the log likelihood. The log prior is calculated by  calculating the log prior in physical parameter
        space and adding the log determinant of the Jacobian.

        Notes:
            The log determinant for the default bijector will output a scalar regardless of the batch size of ``z`` due
            to the flattening behavior of :obj:`tfp.bijectors.Split`. To get around this, we use the fact that
            :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` has determinant 1 (it simply reshapes/restructures the
            input), and the determinant comes purely from
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`.

        Args:
            simulator (:obj:`~gigalens.tf.simulator.LensSimulator`): A simulator object that has the same batch size
                as ``z``.
            z (:obj:`tf.Tensor`): Parameters in unconstrained space with shape ``(bs, d)``, where ``bs`` is the batch
                size and ``d`` is the number of parameters per lens.

        Returns:
            The reparameterized log posterior density, with shape ``(bs,)``.
        """
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(x, self.observed_image, self.err_map)
        log_like = self.observed_dist.log_prob(im_sim)
        log_prior = self.prior.log_prob(
            x
        ) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, tf.reduce_mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
