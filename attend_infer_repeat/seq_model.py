import functools

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, NormalWithSoftplusScale
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.distributions.python.ops.kullback_leibler import kl as _kl

import ops
from evaluation import gradient_summaries, print_vars
from modules import SpatialTransformer, ParametrisedGaussian
from prior import geometric_prior, NumStepsDistribution, tabular_kl


def identity_transiton(inpt, state):
    return inpt


class SeqAIRCell(snt.RNNCore):
    """RNN cell that implements the core features of Attend, Infer, Repeat, as described here:
       https://arxiv.org/abs/1603.08575
       """
    _n_where = 4

    def __init__(self, max_steps, img_size, glimpse_size, n_what, n_samples,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 what_transiton=identity_transiton, where_transition=identity_transiton, presence_transition=identity_transiton,
                 discrete_steps=True, debug=False):
        """Creates the cell

        :param max_steps: int, number of maximum steps per image
        :param img_size: int tuple, size of the image
        :param glimpse_size: int tuple, size of the attention glimpse
        :param n_what: number of latent units describing the "what"
        :param n_samples: number of samples per input for importance-weighted ELBO
        :param transition: an RNN cell for maintaining the internal hidden state
        :param input_encoder: callable, encodes the original input image before passing it into the transition
        :param glimpse_encoder: callable, encodes the glimpse into latent representation
        :param transform_estimator: callabe, transforms the hidden state into parameters for the spatial transformer
        :param steps_predictor: callable, predicts whether to take a step
        :param debug: boolean, adds checks for NaNs in the inputs to distributions
        """

        super(SeqAIRCell, self).__init__(self.__class__.__name__)
        self._max_steps = max_steps
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._glimpse_size = glimpse_size
        self._n_what = n_what
        self._n_samples = n_samples
        self._transition = transition
        self._what_transition = what_transiton
        self._where_transition = where_transition
        self._presence_transition = presence_transition
        self._n_hidden = int(self._transition.output_size[0])
        self._discrete_steps = discrete_steps
        if not self._discrete_steps:
            assert self._n_samples == 1, "There's not point in IWAE if there's no sampling"

        self._debug = debug

        with self._enter_variable_scope():
            self._spatial_transformer = SpatialTransformer(self._img_size, self._glimpse_size)
            self._transform_estimator = transform_estimator(self._n_where)
            self._input_encoder = input_encoder()
            self._glimpse_encoder = glimpse_encoder()
            self._what_distrib = ParametrisedGaussian(n_what, scale_offset=0.5,
                                                      validate_args=self._debug, allow_nan_stats=not self._debug)
            self._steps_predictor = steps_predictor()
            
        self._n_flat = self._max_steps * self._n_samples

    @property
    def state_size(self):
        return [
            self._transition.state_size,  # hidden state of the rnn, learnable but same at the start of every timestep
            self._n_flat * self._n_what,  # what
            self._n_flat * self._n_where,  # where
            self._n_flat * 1,  # presence
        ]

    @property
    def output_size(self):
        return [
            self._n_flat * self._n_what,  # what code
            self._n_flat * self._n_what,  # what loc
            self._n_flat * self._n_what,  # what scale
            self._n_flat * self._n_where,  # where code
            self._max_steps * self._n_where,  # where loc
            self._max_steps * self._n_where,  # where scale
            self._max_steps * 1,  # presence logit
            self._n_flat * 1,  # presence
            self._transition.state_size  # last state of rnn at every timestep
        ]

    @property
    def output_names(self):
        return 'what what_loc what_scale where where_loc where_scale presence_logit presence final_state'.split()

    def initial_state(self, img, trainable_z=False):
        batch_size = img.get_shape().as_list()[0]
        hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        if trainable_z:
            def state_var(name, ndim):
                var = tf.get_variable(name, [1, ndim], tf.float32)
                return tf.tile(var, (batch_size, self._n_flat))

            where_code = state_var('where_init', self._n_where)
            what_code = state_var('what_init', self._n_what)
        else:
            where_code = tf.zeros([self._n_flat * self._n_where], dtype=tf.float32, name='where_init')
            what_code = tf.zeros([self._n_flat * self._n_what], dtype=tf.float32, name='what_init')
            where_code, what_code = (tf.tile(i[tf.newaxis], (batch_size,) + tuple([1] * len(i.get_shape())))
                                                  for i in (where_code, what_code))

        init_presence_logit = tf.ones((batch_size, self._n_flat), dtype=tf.float32) * 1e3
        state = [hidden_state, what_code, where_code, init_presence_logit]
        return state

    def _build(self, img, state):
        """Input is unused; it's only to force a maximum number of steps"""
        # TODO: presence should depend on previous timesteps
        # TODO: the model should be able to "reset" object registers (z, p) to store a new object at a particular index i
        # TODO: more sophisticated 'what' transition

        batch_size = img.get_shape().as_list()[0]
        raw_hidden_state = state[0]
        what, where, presence = (tf.reshape(s, (batch_size, self._n_flat, -1)) for s in state[1:])

        def _sample_flat(distrib):
            samples = distrib.sample(self._n_samples)
            # transpose so that the shape is (batch_size, n_steps, n_samples, -1) and then flatten
            trailing_dim = int(samples.shape[-1])
            return tf.reshape(tf.transpose(samples, (1, 2, 0)), (batch_size * self._n_flat, trailing_dim))

        inpt_encoding = self._input_encoder(img)
        with tf.variable_scope('inner_rnn'):
            hidden_outputs = []

            hidden_state = raw_hidden_state
            # if self._timestep_transition_indicator:
            #     embedding_size = int(inpt_encoding.shape[-1])
            #     timestep_indicator = tf.get_variable('timestep_indicator', [1, embedding_size], tf.float32)
            #     timestep_indicator = tf.tile(timestep_indicator, (batch_size, 1))
            #     _, hidden_state = self._transition(timestep_indicator, hidden_state)

            for i in xrange(self._max_steps):
                hidden_output, hidden_state = self._transition(inpt_encoding, hidden_state)
                hidden_outputs.append(hidden_output)

            hidden_output = tf.stack(hidden_outputs, 1)
            hidden_output = tf.reshape(hidden_output, (-1, self._n_hidden))

        where_param = self._transform_estimator(hidden_output)
        where_distrib = NormalWithSoftplusScale(*where_param,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)
        where_loc, where_scale = where_distrib.loc, where_distrib.scale
        where_delta = _sample_flat(where_distrib)
        where = self._where_transition(where_delta, where)

        # reshape to avoid tiling images
        shaped_where_code = tf.reshape(where, (batch_size, self._n_flat, self._n_where))
        cropped = self._spatial_transformer(img, shaped_where_code)
        cropped = tf.reshape(cropped, (batch_size * self._n_flat,) + tuple(self._glimpse_size))

        with tf.variable_scope('presence'):
            presence_logit = self._steps_predictor(hidden_output)
            if self._discrete_steps:
                presence_distrib = Bernoulli(logits=presence_logit, dtype=tf.float32,
                                             validate_args=self._debug, allow_nan_stats=not self._debug)
                new_presence = _sample_flat(presence_distrib)
            else:
                new_presence = tf.nn.sigmoid(presence_logit)

            new_presence = tf.reshape(new_presence, (batch_size, self._max_steps, self._n_samples))
            if self._discrete_steps:
                new_presence = tf.cumprod(new_presence, axis=1)
            new_presence = tf.reshape(new_presence, (batch_size, self._n_flat, 1))
            presence = self._presence_transition(new_presence, presence)

        what_params = self._glimpse_encoder(cropped)
        what_distrib = self._what_distrib(what_params)
        what_loc, what_scale = what_distrib.loc, what_distrib.scale
        what_delta = what_distrib.sample()
        what = self._what_transition(what_delta, what)
        with tf.variable_scope('rnn_outputs'):
            # flattening
            flat = [what, what_loc, what_scale, where, where_loc, where_scale, presence_logit, presence]
            flat = [tf.reshape(p, (batch_size, -1)) for p in flat]

        output = flat + [hidden_state]
        state = [raw_hidden_state,
                 flat[0],  # what
                 flat[3],  # where_code
                 flat[-1],  # presence
                ]

        return output, state


class AIRDecoder(snt.AbstractModule):

    def __init__(self, img_size, glimpse_size, glimpse_decoder):
        super(AIRDecoder, self).__init__(self.__class__.__name__)
        self._inverse_transformer = SpatialTransformer(img_size, glimpse_size, inverse=True)

        with self._enter_variable_scope():
            self._glimpse_decoder = glimpse_decoder(glimpse_size)

    def _build(self, what, where, presence):
        batch = functools.partial(snt.BatchApply, n_dims=4)
        glimpse = batch(self._glimpse_decoder)(what)
        inversed = batch(self._inverse_transformer)(glimpse, where)
        canvas = tf.reduce_sum(presence[..., tf.newaxis, tf.newaxis] * inversed, axis=2)[..., 0]
        return canvas, glimpse


class SeqAIRModel(object):
    """Generic AIR model"""

    def __init__(self, obs, max_steps, glimpse_size, n_what, n_samples,
                 transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor,
                 output_std=1., output_multiplier=1., discrete_steps=True,
                 debug=False, **kwargs):
        """Creates the model.

        :param obs: tf.Tensor, images
        :param max_steps: int, maximum number of steps to take (or objects in the image)
        :param glimpse_size: tuple of ints, size of the attention glimpse
        :param n_what: int, number of latent variables describing an object
        :param n_samples: int, number of samples per input for importance-weighted ELBO
        :param transition: see :class: AIRCell
        :param input_encoder: see :class: AIRCell
        :param glimpse_encoder: see :class: AIRCell
        :param glimpse_decoder: callable, decodes the glimpse from latent representation
        :param transform_estimator: see :class: AIRCell
        :param steps_predictor: see :class: AIRCell
        :param output_std: float, std. dev. of the output Gaussian distribution
        :param output_multiplier: float, a factor that multiplies the reconstructed glimpses
        :param debug: see :class: AIRCell
        :param **kwargs: all other parameters are passed to AIRCell
        """

        self.obs = obs
        self.max_steps = max_steps
        self.glimpse_size = glimpse_size

        self.n_what = n_what
        self.n_samples = n_samples

        self.output_std = output_std
        self.discrete_steps = discrete_steps
        self.debug = debug

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False, name='canvas_multiplier')

            shape = self.obs.get_shape().as_list()
            self.batch_size = shape[1]
            self.img_size = shape[2:]
            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                        steps_predictor, kwargs)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
               steps_predictor, kwargs):
        """Build the model. See __init__ for argument description"""

        self.n_timesteps = tf.shape(self.obs)[0]
        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder)
        self.cell = SeqAIRCell(self.max_steps, self.img_size, self.glimpse_size, self.n_what, self.n_samples,
                               transition,
                               input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                               discrete_steps=self.discrete_steps, debug=self.debug,
                               **kwargs)

        initial_state = self.cell.initial_state(self.obs[0])
        outputs, state = tf.nn.dynamic_rnn(self.cell, self.obs, initial_state=initial_state, time_major=True)

        n_timesteps = tf.shape(self.obs)[0]
        sample_counts = [self.n_samples] * 4 + [1] * 3 + [self.n_samples]
        for i, (o, n_samples) in enumerate(zip(outputs[:-1], sample_counts)):
            trailing_dim = int(o.get_shape()[-1]) / self.max_steps / n_samples
            outputs[i] = tf.reshape(o, (n_timesteps, self.batch_size, self.max_steps, n_samples, trailing_dim))

        for name, output in zip(self.cell.output_names, outputs):
            setattr(self, name, output)

        self.presence_prob = tf.nn.sigmoid(self.presence_logit)

        self.canvas, self.glimpse = self.decoder(self.what, self.where, self.presence)
        self.canvas *= self.output_multiplier

        self.output_distrib = Normal(self.canvas, self.output_std)
        self.num_steps_distrib = NumStepsDistribution(self.presence_prob, step_axis=-3)

        rounded_presence = tf.round(self.presence)
        self.num_step_per_sample = tf.to_float(tf.reduce_sum(rounded_presence, 2))
        self.num_step = tf.reduce_mean(self.num_step_per_sample)

    def train_step(self, learning_rate, l2_weight=0., what_prior=None, where_scale_prior=None,
                   where_shift_prior=None,
                   num_steps_prior=None,
                   use_reinforce=True, baseline=None, decay_rate=None, nums=None, nums_xe_weight=0.,
                   importance_weighted=False,
                   opt=tf.train.RMSPropOptimizer, opt_kwargs=dict(momentum=.9, centered=True)):
        """Creates the train step and the global_step

        :param learning_rate: float or tf.Tensor
        :param l2_weight: float or tf.Tensor, if > 0. then adds l2 regularisation to the model
        :param what_prior: AttrDict or similar, with `loc` and `scale`, both floats
        :param where_scale_prior: AttrDict or similar, with `loc` and `scale`, both floats
        :param where_shift_prior: AttrDict or similar, with `loc` and `scale`, both floats
        :param num_steps_prior: AttrDict or similar, described as an example:

            >>> num_steps_prior = AttrDict(
            >>> anneal='exp',   # type of annealing of the prior; can be 'exp', 'linear' or None
            >>> init=1. - 1e-7, # initial value of the prior
            >>> final=1e-5,     # final value of the prior
            >>> steps_div=1e4,  # relevant for exponential annealing, see :func: tf.exponential_decay
            >>> steps=1e5,      # number of steps for annealing
            >>> analytic=True
            >>> )

        `init` and `final` describe success probability values in a geometric distribution; for example `init=.9` means
        that the probability of taking a single step is .9, two steps is .9**2 etc.

        :param use_prior: boolean, if False sets the KL-divergence loss term to 0
        :param use_reinforce: boolean, if False doesn't compute gradients for the number of steps
        :param baseline: callable or None, baseline for variance reduction of REINFORCE
        :param decay_rate: float, decay rate to use for exp-moving average for NVIL
        :param nums: tf.Tensor, number of objects in images
        :return: train step and global step
        """

        if num_steps_prior is not None:
            num_steps_prior['analytic'] = getattr(num_steps_prior, 'analytic', True)

        self.l2_weight = l2_weight
        self.what_prior = what_prior
        self.where_scale_prior = where_scale_prior
        self.where_shift_prior = where_shift_prior
        self.num_steps_prior = num_steps_prior
        self.nums = nums
        self.nums_xe_weight = nums_xe_weight

        if not hasattr(self, 'baseline'):
            self.baseline = baseline

        priors = what_prior, where_shift_prior, where_scale_prior, num_steps_prior
        self.use_prior = any([p is not None for p in priors])
        self.use_reinforce = use_reinforce and self.discrete_steps
        make_opt = functools.partial(opt, **opt_kwargs)

        with tf.variable_scope('loss'):
            global_step = tf.train.get_or_create_global_step()
            loss = ops.Loss()
            self._train_step = []
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            # Reconstruction Loss, - \E_q [ p(x | z, n) ]
            rec_loss_per_sample = -self.output_distrib.log_prob(self.obs[:, :, tf.newaxis])
            shape = [self.n_timesteps] + rec_loss_per_sample.shape.as_list()[1:3] + [np.prod(self.img_size)]

            self.rec_loss_per_pixel = tf.reshape(rec_loss_per_sample, shape)
            self.rec_loss_per_sample = tf.reduce_sum(rec_loss_per_sample, axis=(3, 4))
            self.rec_loss = tf.reduce_mean(self.rec_loss_per_sample)
            tf.summary.scalar('rec', self.rec_loss)
            # loss.add(self.rec_loss, self.rec_loss_per_sample)
            loss.add(self.rec_loss_per_pixel)

            # Prior Loss, KL[ q(z, n | x) || p(z, n) ]
            if self.use_prior:
                self.prior_loss = self._prior_loss(what_prior, where_scale_prior,
                                                   where_shift_prior, num_steps_prior, global_step)
                tf.summary.scalar('prior', self.prior_loss.value)
                self.prior_weight = tf.to_float(tf.equal(self.use_prior, True))
                loss.add(self.prior_loss, weight=self.prior_weight)

            # REINFORCE
            if importance_weighted:

                elbo_per_sample = -loss.raw
                imp_weight_per_sample = tf.exp(elbo_per_sample)
                summed_imp_weight = tf.reduce_sum(imp_weight_per_sample, -2)
                summed_imp_weight_per_sample = tf.reduce_sum(tf.log(summed_imp_weight), -1)
                imp_weighted_elbo = tf.reduce_mean(summed_imp_weight_per_sample) - tf.log(float(self.n_samples))
                opt_loss = -imp_weighted_elbo

            else:
                opt_loss = loss.value

            if self.use_reinforce:

                # self.reinforce_imp_weight = self.rec_loss_per_sample
                # if num_steps_prior is not None and not num_steps_prior.analytic:
                #     self.reinforce_imp_weight += self.prior_loss.per_sample
                self.reinforce_imp_weight = self.prior_loss.per_sample

                reinforce_loss = self._reinforce(self.reinforce_imp_weight, decay_rate)
                opt_loss += reinforce_loss

            if self.nums_xe_weight > 0.:
                nums_xe = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.nums, logits=self.presence_logit[..., 0])
                self.nums_xe = tf.reduce_mean(tf.reduce_sum(nums_xe, -1))
                opt_loss += self.nums_xe_weight * self.nums_xe

            all_vars = set(tf.trainable_variables())
            baseline_vars = getattr(self, 'baseline_vars', [])
            model_vars = list(all_vars - set(baseline_vars))
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope=self.decoder.variable_scope.name)

            encoder_vars = list(all_vars - set(baseline_vars) - set(decoder_vars))

            print_vars(encoder_vars, 'encoder')
            print_vars(decoder_vars, 'decoder')
            if len(baseline_vars) > 0:
                print_vars(baseline_vars, 'baseline')

            # L2 reg
            if l2_weight > 0.:
                # don't penalise biases
                weights = [w for w in model_vars if len(w.get_shape()) == 2]
                self.l2_loss = l2_weight * sum(map(tf.nn.l2_loss, weights))
                opt_loss += self.l2_loss
                tf.summary.scalar('l2', self.l2_loss)

            opt = make_opt(self.learning_rate)
            gvs = opt.compute_gradients(opt_loss, var_list=model_vars)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_step = opt.apply_gradients(gvs, global_step=global_step)

            if self.use_reinforce and self.baseline is not None:
                baseline_opt = make_opt(10 * learning_rate)
                self._baseline_train_step = self._make_baseline_train_step(baseline_opt, self.reinforce_imp_weight,
                                                                           self.baseline, self.baseline_vars)
                self._true_train_step = self._train_step
                self._train_step = tf.group(self._true_train_step, self._baseline_train_step)

            tf.summary.scalar('num_step', self.num_step)
        # Metrics
        gradient_summaries(gvs)
        if self.nums is not None:
            self.gt_num_steps = tf.reduce_sum(self.nums, 2)
            gt_num_steps = tf.tile(self.gt_num_steps[..., tf.newaxis], (1, 1, self.n_samples))
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(gt_num_steps, self.num_step_per_sample[..., 0])))

        self.loss = loss
        self.opt_loss = opt_loss
        return self._train_step, global_step

    def _prior_loss(self, what_prior, where_scale_prior, where_shift_prior,
                    num_steps_prior, global_step):
        """Creates KL-divergence term of the loss"""

        with tf.variable_scope('KL_divergence'):
            prior_loss = ops.Loss()
            if num_steps_prior is not None:
                if num_steps_prior.anneal is not None:
                    with tf.variable_scope('num_steps_prior'):
                        nsp = num_steps_prior

                        hold_init = getattr(nsp, 'hold_init', 0.)
                        steps_div = getattr(nsp, 'steps_div', 1.)
                        steps_prior_success_prob = ops.anneal_weight(nsp.init, nsp.final, nsp.anneal, global_step,
                                                                    nsp.steps, hold_init, steps_div)
                else:
                    steps_prior_success_prob = num_steps_prior.init
                self.steps_prior_success_prob = steps_prior_success_prob

                with tf.variable_scope('num_steps'):
                    self.num_steps_prior_prob = geometric_prior(steps_prior_success_prob, self.max_steps)
                    self.num_steps_posterior_prob = self.num_steps_distrib.prob()[..., 0, 0]
                    steps_kl = tabular_kl(self.num_steps_posterior_prob, self.num_steps_prior_prob)
                    self.kl_num_steps_per_step = steps_kl

                    self.kl_num_steps_per_sample = tf.reduce_sum(steps_kl, 2)

                    self.kl_num_steps = tf.reduce_mean(self.kl_num_steps_per_sample)
                    tf.summary.scalar('kl_num_steps', self.kl_num_steps)

                    weight = getattr(num_steps_prior, 'weight', 1.)
                    self.kl_num_steps_per_sample = tf.tile(self.kl_num_steps_per_sample[..., tf.newaxis], (1, 1, self.n_samples))

                    # prior_loss.add(self.kl_num_steps, self.kl_num_steps_per_sample, weight=weight)
                    steps_kl = tf.tile(steps_kl[..., tf.newaxis, :], (1, 1, self.n_samples, 1))
                    prior_loss.add(steps_kl)

                if num_steps_prior.analytic:
                    # reverse cumsum of q(n) needed to compute \E_{q(n)} [ KL[ q(z|n) || p(z|n) ]]
                    step_weight = self.num_steps_posterior_prob[..., 1:]
                    step_weight = tf.cumsum(step_weight, axis=-1, reverse=True)[..., tf.newaxis]
                    step_weight = tf.tile(step_weight, (1, 1, 1, self.n_samples))
                else:
                    step_weight = self.presence[..., 0]

            else:
                step_weight = self.presence[..., 0]

            self.prior_step_weight = step_weight

            # # this prevents optimising the expectation with respect to q(n)
            # # it's similar to the maximisation step of EM: we have a pre-computed expectation
            # # from the E step, and now we're maximising with respect to the argument of the expectation.
            # self.prior_step_weight = tf.stop_gradient(self.prior_step_weight)

            if what_prior is not None:
                with tf.variable_scope('what'):

                    prior = Normal(what_prior.loc, what_prior.scale)
                    posterior = Normal(self.what_loc, self.what_scale)

                    what_kl_raw = _kl(posterior, prior) * self.prior_step_weight[..., tf.newaxis]
                    print 'what_kl', what_kl_raw
                    what_kl = tf.reduce_sum(what_kl_raw, -1)
                    what_kl_per_sample = tf.reduce_sum(what_kl, 2)

                    self.kl_what = tf.reduce_mean(what_kl_per_sample)
                    tf.summary.scalar('kl_what', self.kl_what)
                    # prior_loss.add(self.kl_what, what_kl_per_sample, weight=conditional_kl_weight)

                    shape = [self.n_timesteps, self.batch_size, self.n_samples, self.max_steps * self.n_what]
                    what_kl_raw = tf.transpose(what_kl_raw, (0, 1, 3, 4, 2))
                    what_kl_raw = tf.reshape(what_kl_raw, shape)
                    prior_loss.add(what_kl_raw)

            if where_scale_prior is not None and where_shift_prior is not None:
                with tf.variable_scope('where'):
                    usx, utx, usy, uty = tf.split(self.where_loc[..., 0, :], 4, 3)
                    ssx, stx, ssy, sty = tf.split(self.where_scale[..., 0, :], 4, 3)
                    us = tf.concat((usx, usy), -1)
                    ss = tf.concat((ssx, ssy), -1)

                    scale_distrib = Normal(us, ss)
                    scale_prior = Normal(where_scale_prior.loc, where_scale_prior.scale)
                    scale_kl = _kl(scale_distrib, scale_prior)

                    ut = tf.concat((utx, uty), -1)
                    st = tf.concat((stx, sty), -1)
                    shift_distrib = Normal(ut, st)

                    if 'loc' in where_shift_prior:
                        shift_mean = where_shift_prior.loc
                    else:
                        shift_mean = ut
                    shift_prior = Normal(shift_mean, where_shift_prior.scale)

                    shift_kl = _kl(shift_distrib, shift_prior)
                    where_kl = tf.concat([shift_kl, scale_kl], -1)
                    where_kl = where_kl[..., tf.newaxis, :] * step_weight[..., tf.newaxis]
                    # where_kl = tf.reduce_sum(scale_kl + shift_kl, -1)[..., tf.newaxis] * self.prior_step_weight
                    where_kl_per_sample = tf.reduce_sum(where_kl, 2)
                    self.kl_where = tf.reduce_mean(where_kl_per_sample)
                    tf.summary.scalar('kl_where', self.kl_where)
                    # prior_loss.add(self.kl_where, where_kl_per_sample, weight=conditional_kl_weight)

                    where_kl = tf.transpose(where_kl, (0, 1, 3, 4, 2))
                    where_kl = tf.reshape(where_kl, (self.n_timesteps, self.batch_size, self.n_samples, self.max_steps * 4))
                    prior_loss.add(where_kl)

        return prior_loss

    def _reinforce(self, importance_weight, decay_rate):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        log_prob = self.num_steps_distrib.log_prob(self.num_step_per_sample)

        if self.baseline is not None:
            if not isinstance(self.baseline, tf.Tensor):
                self.baseline_module = self.baseline
                what = tf.reduce_mean(self.what_loc, -1)
                self.baseline = self.baseline_module(self.obs, what, self.where_loc,
                                                     self.presence_prob, self.final_state)
                self.baseline = self.baseline[..., tf.newaxis]
                self.baseline_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope=self.baseline_module.variable_scope.name)

            importance_weight -= self.baseline

        axes = range(len(importance_weight.get_shape()))
        if decay_rate is not None:
            mean, var = tf.nn.moments(importance_weight, axes=axes)
            self.imp_weight_moving_mean = ops.make_moving_average('imp_weight_moving_mean', mean, 0., decay_rate)
            self.imp_weight_moving_var = ops.make_moving_average('imp_weight_moving_var', var, 1., decay_rate)

            factor = tf.maximum(tf.sqrt(self.imp_weight_moving_var), 1.)
            importance_weight = (importance_weight - self.imp_weight_moving_mean) / factor

        self.importance_weight = importance_weight
        imp_weight_mean, imp_weight_var = tf.nn.moments(self.importance_weight, axes)
        tf.summary.scalar('imp_weight_mean', imp_weight_mean)
        tf.summary.scalar('imp_weight_var', imp_weight_var)

        reinforce_loss_per_sample = tf.stop_gradient(self.importance_weight) * log_prob
        self.reinforce_loss = tf.reduce_mean(reinforce_loss_per_sample)
        tf.summary.scalar('reinforce_loss', self.reinforce_loss)

        return self.reinforce_loss

    def _make_baseline_train_step(self, opt, loss, baseline, baseline_vars):
        baseline_target = tf.stop_gradient(loss)

        self.baseline_loss = .5 * tf.reduce_mean(tf.square(baseline_target - baseline))
        tf.summary.scalar('baseline_loss', self.baseline_loss)
        train_step = opt.minimize(self.baseline_loss, var_list=baseline_vars)
        return train_step