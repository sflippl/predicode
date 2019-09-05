"""Tests predicode.hierarchical.regimen."""

import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import predicode as pc

class TestSimpleOptimizerRegimen(unittest.TestCase):
    """Tests a simple optimizer regimen."""

    def setUp(self):
        """Sets up an optimizer regimen."""
        self.regimen = pc.SimpleOptimizerRegimen(
            optimizer=keras.optimizers.SGD(learning_rate=0.1), max_steps=3
        )
        self.variable = [tf.Variable(np.array([1.]), dtype=tf.float32)]
        self.fun = lambda: tf.math.pow(self.variable[0], 2)

    def test_start_batch(self):
        """Start batch increments the step count by one and sets _grads to be
        true."""
        self.regimen.start_batch()
        self.assertEqual(self.regimen.n_steps, 1)

    def test_training_step(self):
        """Tests that the training step changes the value by 0.2."""
        self.regimen.training_step(self.fun, self.variable)
        self.assertAlmostEqual(self.variable[0].numpy()[0], 0.8)

    def test_train(self):
        """Training should go on for three steps and yield a value of 0.7 in the
        end. If allowed to go on for a longer, it should converge rather quickly
        and indicate that with the function steps_until_convergence."""
        self.regimen.train(self.fun, self.variable)
        self.assertEqual(self.regimen.n_steps, 3)
        self.assertTrue(np.isnan(self.regimen.steps_until_convergence()))
        regimen = pc.SimpleOptimizerRegimen(
            optimizer=keras.optimizers.SGD()
        )
        regimen.train(self.fun, self.variable)
        self.assertFalse(np.isnan(regimen.steps_until_convergence()))

class TestConstantRegimen(unittest.TestCase):
    """Tests the constant regimen."""

    def setUp(self):
        """Sets up a constant regimen."""
        self.regimen = pc.ConstantRegimen()

    def test_end(self):
        """Tests that the end is always true."""
        self.assertEqual(self.regimen.end(), True)

    def test_restart(self):
        """Tests the restart."""
        self.assertEqual(self.regimen.restart(), None)

    def test_train(self):
        """Tests the train method."""
        self.assertEqual(self.regimen.train(None, None), None)

    def test_training_step(self):
        """Tests that the training step does nothing."""
        self.assertEqual(self.regimen.training_step(None, None), None)

    def test_steps_until_convergence(self):
        """Tests that the steps until convergence are always 0."""
        self.assertEqual(self.regimen.steps_until_convergence(), 0)

class TestExpectationMaximizationRegimen(unittest.TestCase):
    """Tests that the expectation maximization regimen works."""

    def setUp(self):
        """Sets up simple optimization problem with constant predictor
        regime."""
        self.state_variable = [tf.Variable([1.])]
        self.weight_variable = [tf.Variable([1.])]
        self.regimen = pc.ExpectationMaximizationRegimen(
            state_regimen=pc.SimpleOptimizerRegimen(
                keras.optimizers.SGD(), eps=1e-10
            ),
            predictor_regimen=pc.ConstantRegimen()
        )

    def loss(self):
        """Sets up loss function."""
        return self.state_variable[0]**2 + self.weight_variable[0]**2

    def test_train(self):
        """Tests whether training and restarting works appropriately."""
        self.regimen.train(self.loss, self.state_variable, self.weight_variable)
        self.assertLess(self.regimen.n_steps, 2)
        self.assertEqual(self.weight_variable[0].numpy()[0], 1)
        self.assertAlmostEqual(self.state_variable[0].numpy()[0], 0, places=5)
        self.assertTrue(self.regimen.end())
        self.regimen.restart()
        self.assertFalse(self.regimen.end())
