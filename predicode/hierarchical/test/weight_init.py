"""Tests predicode.hierarchical.weight_init."""

import unittest
import numpy as np

import predicode as pc

class TestWeightInitPCA(unittest.TestCase):
    """Tests weight initialization with PCA."""

    def setUp(self):
        """Sets up the weight initialization."""
        self.weight_init = pc.weight_init_pca()

    def test_initialize(self):
        """Tests weight initialization."""
        self.assertEqual(
            self.weight_init(1, np.array([[1, 0], [0, 1]])).shape,
            (2, 1)
        )

class TestWeightInitRandom(unittest.TestCase):
    """Tests random weight initialization."""

    def setUp(self):
        """Sets up the weight initializations."""
        self.weight_init = pc.weight_init_random('orthogonal')

    def test_validate(self):
        """Tests whether non-implemented methods are appropriately handled."""
        with self.assertRaises(NotImplementedError):
            pc.weight_init_random('nomethod')

    def test_initialize(self):
        """Tests whether the function works correctly."""
        weight = self.weight_init(2, 1)
        self.assertEqual(weight.shape, (2, 1))
        self.assertAlmostEqual(np.matmul(weight.T, weight)[0, 0], 1)
        with self.assertRaises(ValueError):
            self.weight_init(2, 3)
        weight = self.weight_init(2)
        self.assertEqual(weight.shape, (2, 2))

class TestWeightInit(unittest.TestCase):
    """Tests weight_init function."""

    def test_init(self):
        """Tests whether non-character arguments are appropriately handled."""
        with self.assertRaises(NotImplementedError):
            pc.weight_init('nomethod')
        self.assertEqual(pc.weight_init(np.array([1])), np.array([1]))
        call = lambda x: x
        self.assertEqual(pc.weight_init(call), call)

    def test_random(self):
        """Tests whether random weight initialization is appropriately
        implemented."""
        weight = pc.weight_init(
            'random', input_dimensions=2, latent_dimensions=1
        )
        self.assertEqual(weight.shape, (2, 1))
        self.assertAlmostEqual(np.matmul(weight.T, weight)[0, 0], 1)

    def test_pca(self):
        """Tests whether PCA initialization is appropriately implemented."""
        weight = pc.weight_init(
            'pca',
            latent_dimensions=1,
            input_data=np.array([[1, 0], [0, 1]])
        )
        self.assertEqual(weight.shape, (2, 1))
