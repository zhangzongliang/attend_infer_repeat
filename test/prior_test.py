import numpy as np
import unittest

from numpy.testing import assert_array_equal, assert_array_almost_equal
from testing_tools import TFTestBase

from attend_infer_repeat.prior import *


_N_STRESS_ITER = 100


class GeometricPriorTest(TFTestBase):

    def test(self):
        prob = .75
        n_steps = 10
        expected = (1. - prob) * prob ** np.arange(n_steps + 1)
        p = geometric_prior(prob, n_steps)
        p = self.eval(p)
        print p, p.sum()
        assert_array_almost_equal(p, expected)


class TabularKLTest(TFTestBase):

    vars = {
        'x': [tf.float32, [None, None]],
        'y': [tf.float32, [None, None]]
    }

    @classmethod
    def setUpClass(cls):
        super(TabularKLTest, cls).setUpClass()

        cls.kl = tabular_kl(cls.x, cls.y, 0.)

    def test_same(self):
        p = np.asarray([.25] * 4).reshape((1, 4))
        kl = self.eval(self.kl, p, p)
        self.assertEqual(kl.shape, (1, 4))
        self.assertEqual(kl.sum(), 0.)

    def test_zero(self):
        p = [0., .25, .25, .5]
        q = [.25] * 4
        p, q = (np.asarray(i).reshape((1, 4)) for i in (p, q))

        kl = self.eval(self.kl, p, q)
        self.assertGreater(kl.sum(), 0.)

    def test_one(self):
        p = [0., 1., 0., 0.]
        q = [1. - 1e-7, 1e-7, 0., 0.]
        p, q = (np.asarray(i).reshape((1, 4)) for i in (p, q))

        kl = self.eval(self.kl, p, q)
        self.assertGreater(kl.sum(), 0.)

    def test_always_positive_on_random(self):

        def gen():
            a = abs(np.random.rand(1, 4))
            a /= a.sum()
            return a

        for i in xrange(_N_STRESS_ITER):
            p = gen()
            q = gen()

            kl = self.eval(self.kl, p, q)
            self.assertGreater(kl.sum(), 0.)


class ConditionalPresencePosteriorTest(TFTestBase):

    vars = {'x': [tf.float32, [None]]}

    @classmethod
    def setUpClass(cls):
        super(ConditionalPresencePosteriorTest, cls).setUpClass()
        cls.probs = bernoulli_to_modified_geometric(cls.x)

    def test_shape(self):

        x = tf.placeholder(tf.float32, [3])
        probs = bernoulli_to_modified_geometric(x)
        self.assertEqual(tuple(probs.get_shape().as_list()), (4,))

        x = tf.placeholder(tf.float32, [7, 3])
        probs = bernoulli_to_modified_geometric(x)
        self.assertEqual(tuple(probs.get_shape().as_list()), (7, 4,))

        x = tf.placeholder(tf.float32, [7, 11, 3])
        probs = bernoulli_to_modified_geometric(x)
        self.assertEqual(tuple(probs.get_shape().as_list()), (7, 11, 4,))

    def test_obvious(self):
        p = [0., 0., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [1., 0., 0., 0.])

        p = [1., 0., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 1., 0., 0.])

        p = [1., 1., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 0., 1., 0.])

        p = [1., 1., 1.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 0., 0., 1.])

    def test_geom(self):
        p = [.5, .5, .5]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [.5, .5**2, .5**3, .5**3])


class BernoulliToModifiedGeometricTest(TFTestBase):

    vars = {
        'x': [tf.float32, [None, 3]],
        'y': [tf.float32, [3]]
    }

    @classmethod
    def setUpClass(cls):
        super(BernoulliToModifiedGeometricTest, cls).setUpClass()
        cls.geom = bernoulli_to_modified_geometric(cls.x)
        cls.geom_1d = bernoulli_to_modified_geometric(cls.y)

    def test_1d(self):
        self.assertEqual(self.geom.shape.ndims, 2)
        self.assertEqual(self.geom_1d.shape.ndims, 1)

        expected = [.9, .09, .009, .001]
        probs = self.eval(self.geom_1d, yy=[.1, .1, .1])
        self.assertEqual(probs.shape, (4,))
        assert_array_almost_equal(probs, expected)

    def test_tensor(self):

        x = tf.placeholder(tf.float32, [None]*3)
        geom = bernoulli_to_modified_geometric(x)
        self.assertEqual(geom.shape.ndims, 3)

        expected = np.asarray([.9, .09, .009, .001]).reshape(1, 1, 4)
        y = np.asarray([.1, .1, .1]).reshape((1, 1, 3))
        probs = self.eval(geom, feed_dict={x: y})
        self.assertEqual(probs.shape, (1, 1, 4))
        assert_array_almost_equal(probs, expected)

    def test_axis(self):
        x = tf.placeholder(tf.float32, [1, 3, 1])
        geom = bernoulli_to_modified_geometric(x, axis=-2)

        expected = np.asarray([.9, .09, .009, .001]).reshape(1, 4, 1)
        y = np.asarray([.1, .1, .1]).reshape((1, 3, 1))
        probs = self.eval(geom, feed_dict={x: y})
        self.assertEqual(probs.shape, (1, 4, 1))
        assert_array_almost_equal(probs, expected)


class NumStepsKLTest(TFTestBase):

    vars = {'x': [tf.float32, [None, None]]}

    @classmethod
    def setUpClass(cls):
        super(NumStepsKLTest, cls).setUpClass()

        cls.prior = geometric_prior(.005, 3)

        cls.posterior = bernoulli_to_modified_geometric(cls.x)
        cls.posterior_grad = tf.gradients(cls.posterior, cls.x)

        cls.posterior_kl = tabular_kl(cls.posterior, cls.prior, 0.)
        cls.posterior_kl_grad = tf.gradients(tf.reduce_sum(cls.posterior_kl), cls.x)

        cls.free_kl = tabular_kl(cls.x, cls.prior, 0.)
        cls.free_kl_grad = tf.gradients(tf.reduce_sum(cls.free_kl), cls.x)

    def test_free_stress(self):
        for i in xrange(_N_STRESS_ITER):
            p = abs(np.random.rand(1, 4))
            p /= p.sum()

            kl = self.eval(self.free_kl, p)
            self.assertGreater(kl.sum(), 0)
            self.assertFalse(np.isnan(kl).any())
            self.assertTrue(np.isfinite(kl).all())

            grad = self.eval(self.free_kl_grad, p)
            self.assertFalse(np.isnan(grad).any())
            self.assertTrue(np.isfinite(grad).all())

    def test_posterior_stress(self):
        batch_size = 1

        for i in xrange(_N_STRESS_ITER):
            p = np.random.rand(batch_size, 3)
            kl = self.eval(self.posterior_kl, p)
            self.assertGreater(kl.sum(), 0), '{}'.format(kl)
            self.assertFalse(np.isnan(kl).any())
            self.assertTrue(np.isfinite(kl).all())

            grad = self.eval(self.posterior_kl_grad, p)
            self.assertFalse(np.isnan(grad).any())
            self.assertTrue(np.isfinite(grad).all())

    def test_posterior_zeros(self):
        p = np.asarray([.5, 0., 0.]).reshape((1, 3))

        posterior = self.eval(self.posterior, p)
        print 'posterior', posterior
        posterior_grad = self.eval(self.posterior_grad, p)
        print 'posterior grad', posterior_grad

        kl = self.eval(self.posterior_kl, p)
        print kl
        self.assertGreater(kl.sum(), 0)
        self.assertFalse(np.isnan(kl).any())
        self.assertTrue(np.isfinite(kl).all())

        grad = self.eval(self.posterior_kl_grad, p)
        print grad
        self.assertFalse(np.isnan(grad).any())
        self.assertTrue(np.isfinite(grad).all())