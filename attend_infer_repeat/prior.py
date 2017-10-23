import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Geometric

from ops import clip_preserve, sample_from_tensor


def masked_apply(tensor, op, mask):
    """Applies `op` to tensor only at locations indicated by `mask` and sets the rest to zero.

    Similar to doing `tensor = tf.where(mask, op(tensor), tf.zeros_like(tensor))` but it behaves correctly
    when `op(tensor)` is NaN or inf while tf.where does not.

    :param tensor: tf.Tensor
    :param op: tf.Op
    :param mask: tf.Tensor with dtype == bool
    :return: tf.Tensor
    """
    chosen = tf.boolean_mask(tensor, mask)
    applied = op(chosen)
    idx = tf.to_int32(tf.where(mask))
    result = tf.scatter_nd(idx, applied, tf.shape(tensor))
    return result


def geometric_prior(success_prob, n_steps):
    # clipping here is ok since we don't compute gradient wrt success_prob
    success_prob = tf.clip_by_value(success_prob, 1e-7, 1. - 1e-15)
    geom = Geometric(probs=1. - success_prob)
    events = tf.range(n_steps + 1, dtype=geom.dtype)
    probs = geom.prob(events)
    return probs


def _cumprod(tensor, axis=0):
    """A custom version of cumprod to prevent NaN gradients when there are zeros in `tensor`
    as reported here: https://github.com/tensorflow/tensorflow/issues/3862

    :param tensor: tf.Tensor
    :return: tf.Tensor
    """
    transpose_permutation = None
    n_dim = int(tensor.shape.ndims)
    if n_dim > 1 and axis != 0:

        if axis < 0:
            axis += n_dim

        transpose_permutation = np.arange(n_dim)
        transpose_permutation[axis], transpose_permutation[0] = 0, axis

    tensor = tf.transpose(tensor, transpose_permutation)

    def prod(acc, x):
        return acc * x

    prob = tf.scan(prod, tensor)
    tensor = tf.transpose(prob, transpose_permutation)
    return tensor


def bernoulli_to_modified_geometric(presence_prob, axis=-1):
    """Converts probabilities of independent Bernoulli events into a NumStepsDistribution
    via stick-braking construction.

    :param presence_prob: tf.Tensor, prob of independent Bernoulli events
    :param axis: int, axis along which step probs should be computed
    :return: tf.Tensor of shape = presence.prob_shape; shape[axis] += 1
    """
    presence_prob = tf.cast(presence_prob, tf.float64)
    inv = 1. - presence_prob
    prob = _cumprod(presence_prob, axis=axis)

    try:
        n_steps = int(prob.shape[axis])
    except TypeError:
        n_steps = tf.shape(prob)[axis]

    if axis < 0:
        axis = presence_prob.shape.ndims + axis

    inv_first, inv_remaining = tf.split(inv, (1, n_steps-1), axis=axis)
    prob_until_last, prob_last = tf.split(prob, (n_steps -1, 1), axis=axis)

    modified_prob = tf.concat([inv_first, inv_remaining * prob_until_last, prob_last], axis)
    modified_prob /= tf.reduce_sum(modified_prob, axis, keep_dims=True)
    return tf.cast(modified_prob, tf.float32)


def tabular_kl(p, q, zero_prob_value=0., logarg_clip=None):
    """Computes KL-divergence KL(p||q) for two probability mass functions (pmf) given in a tabular form.

    :param p: iterable
    :param q: iterable
    :param zero_prob_value: float; values below this threshold are treated as zero
    :param logarg_clip: float or None, clips the argument to the log to lie in [-logarg_clip, logarg_clip] if not None
    :return: iterable of brodcasted shape of (p * q), per-coordinate value of KL(p||q)
    """
    p, q = (tf.cast(i, tf.float64) for i in (p, q))
    non_zero = tf.greater(p, zero_prob_value)
    logarg = p / q

    if logarg_clip is not None:
        logarg = clip_preserve(logarg, 1. / logarg_clip, logarg_clip)

    log = masked_apply(logarg, tf.log, non_zero)
    kl = p * log

    return tf.cast(kl, tf.float32)


class NumStepsDistribution(object):
    """Probability distribution used for the number of steps

    Transforms Bernoulli probabilities of an event = 1 into p(n) where n is the number of steps
    as described in the AIR paper."""

    def __init__(self, steps_probs, step_axis=-1):
        """

        :param steps_probs: tensor; Bernoulli success probabilities
        :param step_axis: int, axis along which to compute step probabilities
        """
        self._steps_probs = steps_probs
        self._step_axis = step_axis
        self._joint = bernoulli_to_modified_geometric(steps_probs, step_axis)
        self._bernoulli = None

    def sample(self, n=None):
        if self._bernoulli is None:
            self._bernoulli = Bernoulli(self._steps_probs)

        sample = self._bernoulli.sample(n)
        sample = tf.cumprod(sample, tf.rank(sample) - 1)
        sample = tf.reduce_sum(sample, -1)
        return sample

    def prob(self, samples=None):
        if samples is None:
            return self._joint

        probs = sample_from_tensor(self._joint, samples, self._step_axis)
        return probs

    def log_prob(self, samples):
        prob = self.prob(samples)
        prob = clip_preserve(prob, 1e-32, prob)
        return tf.log(prob)





