"""Tests predicode.hierarchical.
"""

import io
import unittest
import unittest.mock

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import predicode as pc

def _setup_hpc():
    hpc = pc.Hierarchical()
    hpc.add_tier(shape=(2, ))
    hpc.add_tier(shape=(1, ),
                 name='latent_layer',
                 initializer=tf.initializers.constant([[0.]]))
    return hpc

class TestTierUI(unittest.TestCase):
    """Tests the user interface for adding tiers.
    """

    def setUp(self):
        """Sets up a two-layer model.
        """
        self.hpc = _setup_hpc()

    def test_add_tier_fails(self):
        """Tests if add_tier fails appropriately.
        """
        with self.assertRaises(ValueError):
            self.hpc.add_tier((2, ), 'tier_0')

    def test_correct_tiers_initialized(self):
        """Tests if the first tier is constant and the second tier is
        variable.
        """
        self.assertTrue(isinstance(self.hpc.tier(1), tf.Variable))
        self.assertTrue(isinstance(self.hpc.tier('latent_layer'), tf.Variable))

    def test_choose_connection(self):
        """Tests if the appropriate connections are chosen.
        """
        with self.assertRaises(ValueError):
            self.hpc.activate_connection(1, False)
        with self.assertRaises(ValueError):
            self.hpc.activate_connection(0)
        with self.assertRaises(ValueError):
            self.hpc.activate_connection(-1)
        with self.assertRaises(ValueError):
            self.hpc.activate_connection(2)
        with self.assertRaises(ValueError):
            self.hpc.activate_connection('tier_1')
        with self.assertRaises(TypeError):
            self.hpc.activate_connection([])

    def test_connection(self):
        """Tests if returned predictor and state prediction is appropriate."""
        self.assertIsInstance(self.hpc.connection,
                              pc.connections.NoTierConnection)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, mock_stdout):
        """Test summary."""
        self.hpc.summary()
        expected_string = ('# Tier 1: latent_layer\n\n'
                           '# Connection: latent_layer -> tier_0\n'
                           '(No tier connection defined.)\n\n'
                           '# Tier 0: tier_0\n')
        self.assertEqual(
            mock_stdout.getvalue(),
            expected_string
        )

class TestTierConnectionUI(unittest.TestCase):
    """Tests user interface for adding tier connections."""

    def setUp(self):
        """Sets up a two-layer model with a simple predictor."""
        self.hpc = _setup_hpc()
        self.hpc.connection = pc.connections.TopDownSequential()
        self.hpc.connection.add(
            keras.layers.Dense(2, input_shape=(1, ), use_bias=False)
        )

    def test_assignment_fails(self):
        """Tests if assignment is now protected."""
        with self.assertRaises(TypeError):
            self.hpc.connection = keras.Sequential()

    def test_deletion(self):
        """Tests if deleting predictors works."""
        self.hpc.delete_connection()
        self.assertIsInstance(self.hpc.connection,
                              pc.connections.NoTierConnection)

class TestNoTierConnection(unittest.TestCase):
    """Test NoModel class."""

    def setUp(self):
        """Set up NoPredictor."""
        self.no_tier_connection = pc.connections.NoTierConnection()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test(self, mock_stdout):
        """Main Test."""
        self.no_tier_connection.summary()
        self.assertEqual(mock_stdout.getvalue(),
                         '(No tier connection defined.)\n')

class TestEstimation(unittest.TestCase):
    """Tests the estimation functionality."""

    def setUp(self):
        """Set up a linear HPC model."""
        self.hpc = _setup_hpc()
        self.hpc.connection = pc.connections.TopDownSequential()
        self.hpc.connection.add(
            keras.layers.Dense(2, input_shape=(1, ), use_bias=False)
        )
        self.arr = np.array([[1., 0.], [-1., 0.]])
        self.regimen = pc.regimens.EMRegimen(
            state_regimen=pc.regimens.OptimizerRegimen(
                keras.optimizers.Adam(), max_steps=2
            ),
            predictor_regimen=pc.regimens.OptimizerRegimen(
                keras.optimizers.Adam(), max_steps=2
            )
        )
        self.metric = keras.metrics.MeanSquaredError()
        self.hpc.compile(optimizer=self.regimen, metrics=[self.metric])

    def test_as_dataset(self):
        """Tests the as_dataset fails appropriately and returns a dataset."""
        array_ds = self.hpc.as_dataset(self.arr)
        self.assertIsInstance(array_ds, tf.data.Dataset)
        entry = next(iter(array_ds))
        self.assertTrue('tier_0' in entry)
        with self.assertRaises(ValueError):
            self.hpc.as_dataset(tf.data.Dataset.from_tensors(self.arr))
        with self.assertRaises(ValueError):
            self.hpc.as_dataset({'not_a_name': self.arr})
        with self.assertRaises(ValueError):
            self.hpc.as_dataset(np.array([[1.], [1.]]))

    def test_training(self):
        """Test if train method works appopriately and yields sensible values.
        """
        self.hpc.train(self.arr, epochs=2)
        self.assertEqual(self.hpc.tier(1).shape, [2, 1])

    def test_batch_training(self):
        """Tests if we can also train in batches."""
        self.hpc.train(self.arr, batch_size=1)
        self.assertEqual(self.hpc.tier(1).shape, [1, 1])

    def test_is_ready_sp(self):
        """Tests if sensible error comes out of trying to train a model without
        tier connection.
        """
        self.hpc.delete_connection()
        with self.assertRaises(ValueError):
            self.hpc.train(self.arr, self.regimen)

    def test_metric(self):
        """Tests if the metric yields results.
        """
        self.assertGreaterEqual(self.metric.result(), 0)
