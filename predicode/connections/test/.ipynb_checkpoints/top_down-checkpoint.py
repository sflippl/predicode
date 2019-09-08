"""Tests predicode.connections.top_down."""

import io
import unittest
import unittest.mock

import tensorflow.keras as keras

import predicode as pc

class TestTopDownSequential(unittest.TestCase):
    """Test TopDownSequential tier connections.
    """

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, mock_stdout):
        """Test the summary.
        """
        seq = pc.connections.TopDownSequential([
            keras.layers.Dense(2, input_shape=(1, ))
        ])
        seq.summary()
        summary = mock_stdout.getvalue()
        self.assertIn('## Predictive model', summary)
