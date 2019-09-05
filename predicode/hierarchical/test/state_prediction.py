"""Tests predicode.hierarchical.state_prediction."""

import io
import unittest
import unittest.mock

import tensorflow as tf
import numpy as np

import predicode as pc

class TestStatePrediction(unittest.TestCase):
    """Tests the simple loss-driven state prediction."""
    def setUp(self):
        """Sets up a simple state prediction with the standard loss."""
        self.state_prediction = pc.StatePrediction()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, mock_stdout):
        """Test summary."""
        self.state_prediction.summary()
        expected_string = (
            'Loss-driven state prediction.\n'
            'Loss function: %s\n' % (self.state_prediction.loss.__str__(), )
        )
        self.assertEqual(
            mock_stdout.getvalue(),
            expected_string
        )

    def test_loss_computation(self):
        """Test loss computation."""
        loss = self.state_prediction.compute_loss(
            tf.constant(np.array([[1.], [1.]])),
            tf.constant(np.array([[0.], [1.]]))
        )
        self.assertEqual(loss[0], 1.)
        self.assertEqual(loss[1], 0.)
