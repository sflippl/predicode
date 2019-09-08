"""Tests predicode.prediction_errors."""

import unittest

import tensorflow as tf

import predicode as pc

class TestGet(unittest.TestCase):
    """Test getting prediction errors from identifiers.
    """

    def test_get(self):
        """Test retrieving prediction errors from identifiers.
        """
        fun = lambda states, predictions: tf.math.log(states/predictions)
        self.assertEqual(pc.prediction_errors.get(fun), fun)
        self.assertEqual(pc.prediction_errors.get('difference'),
                         pc.prediction_errors.difference)
        with self.assertRaises(ValueError):
            pc.prediction_errors.get('no_prediction_error')
