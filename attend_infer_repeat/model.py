import functools

import tensorflow as tf
from tensorflow.contrib.distributions import Normal


import ops
from cell import AIRCell
from evaluation import gradient_summaries
from prior import NumStepsDistribution
from modules import AIRDecoder


# TODO: implement IWAE
# TODO: implement VIMCO as a mixin


class AIRModel(object):
    """Generic AIR model

    :param analytic_kl_expectation: bool, computes expectation over conditional-KL analytically if True
    """

    analytic_kl_expectation = False

    def __init__(self, obs, max_steps, glimpse_size,
                 n_what, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor, n_samples=1,
                 output_std=1., discrete_steps=True, output_multiplier=1.,
                 debug=False, **cell_kwargs):
        """Creates the model.

        :param obs: tf.Tensor, images
        :param max_steps: int, maximum number of steps to take (or objects in the image)
        :param glimpse_size: tuple of ints, size of the attention glimpse
        :param n_what: int, number of latent variables describing an object
        :param transition: see :class: AIRCell
        :param input_encoder: see :class: AIRCell
        :param glimpse_encoder: see :class: AIRCell
        :param glimpse_decoder: callable, decodes the glimpse from latent representation
        :param transform_estimator: see :class: AIRCell
        :param steps_predictor: see :class: AIRCell
        :param output_std: float, std. dev. of the output Gaussian distribution
        :param discrete_steps: see :class: AIRCell
        :param output_multiplier: float, a factor that multiplies the reconstructed glimpses
        :param debug: see :class: AIRCell
        :param **cell_kwargs: all other parameters are passed to AIRCell
        """

        self.obs = obs
        self.max_steps = max_steps
        self.glimpse_size = glimpse_size
        self.n_what = n_what
        self.n_samples = n_samples
        self.output_std = output_std
        self.discrete_steps = discrete_steps
        self.debug = debug

        print 'n_samples', 1

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False, name='canvas_multiplier')

            shape = self.obs.get_shape().as_list()
            self.batch_size = shape[0]
            self.img_size = shape[1:]
            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                        steps_predictor, cell_kwargs)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
               steps_predictor, cell_kwargs):
        """Build the model. See __init__ for argument description"""
        # save existing variables to know later what we've created
        previous_vars = tf.trainable_variables()

        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder, batch_dims=3)
        self.cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, self.n_samples, transition,
                            input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                            discrete_steps=self.discrete_steps,
                            debug=self.debug,
                            **cell_kwargs)

        initial_state = self.cell.initial_state(self.obs)

        dummy_sequence = tf.zeros((self.max_steps, self.batch_size, 1), name='dummy_sequence')
        outputs, state = tf.nn.dynamic_rnn(self.cell, dummy_sequence, initial_state=initial_state, time_major=True)

        many = [0, 1, 2, 3, len(outputs) - 1]
        for i, (name, output) in enumerate(zip(self.cell.output_names, outputs)):
            if i in many:
                output = tf.reshape(output, (self.max_steps, self.batch_size, self.n_samples, -1))
            else:
                output = tf.reshape(output, (self.max_steps, self.batch_size, 1, -1))
            output = tf.transpose(output, (1, 2, 0, 3))
            setattr(self, name, output)

        self.canvas, self.glimpse = self.decoder(self.what, self.where, self.presence)

        self.final_state = state[1]
        self.num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(self.presence), -1))
        self.num_step = tf.reduce_mean(self.num_step_per_sample)
        tf.summary.scalar('num_step', self.num_step)

        self.output_distrib, self.num_steps_posterior, self.scale_posterior, self.shift_posterior, self.what_posterior\
            = self._make_posteriors()

        self.num_step_prior_prob, self.num_step_prior,\
        self.scale_prior, self.shift_prior, self.what_prior = self._make_priors()

        # group variables
        model_vars = set(tf.trainable_variables()) - set(previous_vars)
        self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.decoder.variable_scope.name)
        self.encoder_vars = list(model_vars - set(self.decoder_vars))
        self.model_vars = list(model_vars)

    def _make_posteriors(self):
        """Builds posterior distributions. This is fairly standard and shouldn't be changed.

        :return:
        """
        output_distrib = Normal(self.canvas, self.output_std)

        posterior_step_probs = tf.squeeze(self.presence_prob)
        num_steps_posterior = NumStepsDistribution(posterior_step_probs)

        ax = self.where_loc.shape.ndims - 1
        us, ut = tf.split(self.where_loc, 2, ax)
        ss, st = tf.split(self.where_scale, 2, ax)
        scale_posterior = Normal(us, ss)
        shift_posterior = Normal(ut, st)
        what_posterior = Normal(self.what_loc, self.what_scale)

        return output_distrib, num_steps_posterior, scale_posterior, shift_posterior, what_posterior

    def train_step(self, learning_rate, l2_weight=0., nums=None,
                   optimizer=tf.train.RMSPropOptimizer, opt_kwargs=dict(momentum=.9, centered=True)):
        """Creates the train step and the global_step

        :param learning_rate: float or tf.Tensor
        :param l2_weight: float or tf.Tensor, if > 0. then adds l2 regularisation to the model
        :param use_reinforce: boolean, if False doesn't compute gradients for the number of steps
        :param baseline: callable or None, baseline for variance reduction of REINFORCE
        :param decay_rate: float, decay rate to use for exp-moving average for NVIL
        :param nums: tf.Tensor, number of objects in images
        :return: train step and global step
        """

        self.l2_weight = l2_weight

        with tf.variable_scope('loss'):
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            make_opt = functools.partial(optimizer, **opt_kwargs)

            # Reconstruction Loss, - \E_q [ p(x | z, n) ]
            self.rec_loss = ops.Loss()
            rec_loss, rec_loss_per_sample = self._log_likelihood()
            tf.summary.scalar('rec', rec_loss)
            self.rec_loss.add(rec_loss, rec_loss_per_sample)

            # Prior Loss, KL[ q(z, n | x) || p(z, n) ]
            self.kl_div = self._kl_divergence()
            tf.summary.scalar('prior', self.kl_div.value)

        with tf.variable_scope('grad'):
            self._train_step, self.gvs = self._make_train_step(make_opt, self.rec_loss, self.kl_div)

        # Metrics
        gradient_summaries(self.gvs)
        if nums is not None:
            self.gt_num_steps = tf.squeeze(tf.reduce_sum(nums, 0))
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, self.num_step_per_sample)))

        print 'rec_loss', self.rec_loss.per_sample.shape
        print 'kl_div', self.kl_div.per_sample.shape

        # negative ELBO
        self.nelbo = self.rec_loss.value + self.kl_div.value
        return self._train_step, tf.train.get_or_create_global_step()

    def _l2_loss(self):

        l2_loss = 0.
        # L2 reg
        if self.l2_weight > 0.:
            # don't penalise biases
            weights = [w for w in self.model_vars if len(w.get_shape()) == 2]
            l2_loss = self.l2_weight * sum(map(tf.nn.l2_loss, weights))
            tf.summary.scalar('l2', l2_loss)
        return l2_loss

    def _kl_divergence(self):
        """Creates KL-divergence term of the loss"""

        with tf.variable_scope('KL_divergence'):
            kl_divergence = ops.Loss()

            with tf.variable_scope('num_steps'):

                self.kl_num_steps, self.kl_num_steps_per_sample = self._kl_num_steps()
                tf.summary.scalar('kl_num_steps', self.kl_num_steps)
                kl_divergence.add(self.kl_num_steps, self.kl_num_steps_per_sample[:, tf.newaxis])

            self.ordered_step_prob = self._ordered_step_prob()
            with tf.variable_scope('what'):
                self.kl_what, self.kl_what_per_sample = self._kl_what()
                tf.summary.scalar('kl_what', self.kl_what)
                kl_divergence.add(self.kl_what, self.kl_what_per_sample)

            with tf.variable_scope('where'):
                self.kl_where, self.kl_where_per_sample = self._kl_where()
                tf.summary.scalar('kl_where', self.kl_where)
                kl_divergence.add(self.kl_where, self.kl_where_per_sample)

        return kl_divergence