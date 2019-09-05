"""Tests predicode.hierarchical."""

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
    """Tests the user interface for adding tiers."""

    def setUp(self):
        """Sets up a two-layer model."""
        self.hpc = _setup_hpc()

    def test_add_tier_fails(self):
        """Tests if add_tier fails appropriately."""
        with self.assertRaises(ValueError):
            self.hpc.add_tier((2, ), 'tier_0')

    def test_correct_tiers_initialized(self):
        """Tests if the first tier is constant and the second tier is
        variable."""
        self.assertTrue(isinstance(self.hpc.tier(1), tf.Variable))
        self.assertTrue(isinstance(self.hpc.tier('latent_layer'), tf.Variable))

    def test_choose_connection(self):
        """Tests if the appropriate connections are chosen."""
        with self.assertRaises(ValueError):
            self.hpc.choose_connection(1, False)
        with self.assertRaises(ValueError):
            self.hpc.choose_connection(0)
        with self.assertRaises(ValueError):
            self.hpc.choose_connection(-1)
        with self.assertRaises(ValueError):
            self.hpc.choose_connection(2)
        with self.assertRaises(ValueError):
            self.hpc.choose_connection('tier_1')
        with self.assertRaises(TypeError):
            self.hpc.choose_connection([])

    def test_connection(self):
        """Tests if returned predictor and state prediction is appropriate."""
        self.assertTrue(isinstance(self.hpc.predictor, pc.NoPredictor))
        self.assertTrue(isinstance(self.hpc.state_prediction,
                                   pc.NoStatePrediction))

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_summary(self, mock_stdout):
        """Test summary."""
        self.hpc.summary()
        expected_string = ('# Tier 1: latent_layer\n'
                           '## Connecting Predictor\n'
                           '(No predictor defined.)\n'
                           '## Connecting State Prediction\n'
                           '(No state prediction defined.)\n'
                           '# Tier 0: tier_0\n')
        self.assertEqual(
            mock_stdout.getvalue(),
            expected_string
        )

class TestPredictorUI(unittest.TestCase):
    """Tests user interface for adding predictors."""

    def setUp(self):
        """Sets up a two-layer model with a simple predictor."""
        self.hpc = _setup_hpc()
        self.hpc.predictor = keras.Sequential()
        self.hpc.predictor.add(
            keras.layers.Dense(2, input_shape=(1, ), use_bias=False)
        )

    def test_assignment_fails(self):
        """Tests if assignment is now protected."""
        with self.assertRaises(TypeError):
            self.hpc.predictor = keras.Sequential()

    def test_deletion(self):
        """Tests if deleting predictors works."""
        self.hpc.delete_predictor()
        self.assertTrue(isinstance(self.hpc.predictor, pc.NoPredictor))

class TestStatePredictionUI(unittest.TestCase):
    """Tests user interface for state prediction."""

    def setUp(self):
        self.hpc = _setup_hpc()
        self.hpc.state_prediction = 'Dummy'

    def test_assignment_fails(self):
        """Tests if assignment is now protected."""
        with self.assertRaises(TypeError):
            self.hpc.state_prediction = []

    def test_deletion(self):
        """Tests if deleting state predictions works."""
        self.hpc.delete_state_prediction()
        self.assertTrue(isinstance(self.hpc.state_prediction,
                                   pc.NoStatePrediction))

    def test_is_ready(self):
        """Tests if is_ready stops when no state prediction is specified."""
        self.hpc.delete_state_prediction()
        self.hpc.predictor = keras.Sequential([
            keras.layers.Dense(2, input_shape=(1, ))
        ])

class TestNoPredictor(unittest.TestCase):
    """Test NoModel class."""

    def setUp(self):
        """Set up NoPredictor."""
        self.no_predictor = pc.NoPredictor()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test(self, mock_stdout):
        """Main Test."""
        self.no_predictor.summary()
        self.assertEqual(mock_stdout.getvalue(), '(No predictor defined.)\n')

class TestNoStatePrediction(unittest.TestCase):
    """Test NoStatePrediction class."""

    def setUp(self):
        """Set up NoStatePrediction."""
        self.no_sp = pc.NoStatePrediction()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test(self, mock_stdout):
        """Main Test."""
        self.no_sp.summary()
        self.assertEqual(mock_stdout.getvalue(),
                         '(No state prediction defined.)\n')

class TestEstimation(unittest.TestCase):
    """Tests the estimation functionality."""

    def setUp(self):
        """Set up a linear HPC model."""
        self.hpc = _setup_hpc()
        self.hpc.predictor = keras.Sequential()
        self.hpc.predictor.add(
            keras.layers.Dense(2, input_shape=(1, ), use_bias=False)
        )
        self.hpc.state_prediction = pc.StatePrediction()
        self.arr = np.array([[1., 0.], [-1., 0.]])
        self.regimen = pc.ExpectationMaximizationRegimen(
            state_regimen=pc.SimpleOptimizerRegimen(
                keras.optimizers.Adam(), max_steps=2
            ),
            predictor_regimen=pc.SimpleOptimizerRegimen(
                keras.optimizers.Adam(), max_steps=2
            ),
            max_steps=2
        )

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
        self.hpc.train(self.arr, self.regimen)
        self.assertEqual(self.hpc.tier(1).shape, [2, 1])

    def test_batch_training(self):
        """Tests if we can also train in batches."""
        self.hpc.train(self.arr, self.regimen, batch_size=1)
        self.assertEqual(self.hpc.tier(1).shape, [1, 1])

    def test_is_ready(self):
        """Tests if sensible error comes out of trying to train a model without
        state prediction."""
        self.hpc.delete_state_prediction()
        with self.assertRaises(ValueError):
            self.hpc.train(self.arr, self.regimen)
