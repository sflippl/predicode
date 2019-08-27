"""Tests predicode.hierarchical.interfaces.test."""

import warnings

import unittest
import numpy as np
import tensorflow as tf

import predicode as pc

class TestMinimalModelState(unittest.TestCase):
    """Test state estimation of the minimal model."""

    def setUp(self):
        """Estimates an example model for ten steps and silenced messages."""
        self.model = pc.MinimalHierarchicalModel(
            np.array([[1, 1], [0, 1]]), latent_dimensions=1
        )
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.model.train(steps=10)

    def test_activate(self):
        """Tests whether activation works."""
        self.assertEqual(self.model.what, 'state')
        with self.assertRaises(ValueError):
            self.model.activate('no')

    def test_eval(self):
        """Tests whether state evaluation works."""
        self.assertAlmostEqual(np.round(self.model.evaluate()['loss'], 1), 0.5)

    def test_predict(self):
        """Tests whether prediction works."""
        prediction = self.model.predict()
        self.assertEqual(prediction.shape, (2, 2))

    def test_latent_values(self):
        """Tests whether latent values can be retained."""
        latent_values = self.model.latent_values
        self.assertEqual(latent_values.shape, (2, 1))

    def test_learning_curve(self):
        learning_curve = self.model.learning_curve(steps=10, resolution=5)
        self.assertEqual(learning_curve.shape, (2, 1, 2))
    
    def test_train(self):
        """Test alternative learning rate."""
        self.model.train(steps=10, learning_rate=10)

class TestMinimalModelWeight(unittest.TestCase):
    """Test weight estimation of the minimal model."""

    def setUp(self):
        """Sets up weight estimation stuff."""
        self.model = pc.MinimalHierarchicalModel(np.array([[1, 1], [0, 1]]),
                                                 latent_dimensions=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.train(steps=10)
        self.model.activate('weight')

    def test_train(self):
        """Test weight training."""
        with self.assertRaises(NotImplementedError):
            self.model.train()

    def test_eval(self):
        """Test weight evaluation."""
        with self.assertRaises(NotImplementedError):
            self.model.evaluate()

    def test_predict(self):
        """Test prediction after weight estimation."""
        with self.assertRaises(NotImplementedError):
            self.model.predict()
