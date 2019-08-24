import unittest
import predicode as pc
import numpy as np

class TestWeightInitPCA(unittest.TestCase):
    def setUp(self):
        self.weight_init = pc.WeightInitPCA()
    
    def test_initialize(self):
        self.assertEqual(self.weight_init.initialize(1,np.array([[1,0],[0,1]])).shape, (2,1))

class TestWeightInitRandom(unittest.TestCase):
    def setUp(self):
        self.weight_init = pc.WeightInitRandom('orthogonal')
    
    def test_validate(self):
        with self.assertRaises(AssertionError):
            self.weight_init.validate_method('nomethod')
    
    def test_initialize(self):
        weight = self.weight_init.initialize(2,1)
        self.assertEqual(weight.shape, (2,1))
        self.assertAlmostEqual(np.matmul(weight.T, weight)[0,0], 1)
        with self.assertRaises(ValueError):
            self.weight_init.initialize(2,3)
        weight = self.weight_init.initialize(2)
        self.assertEqual(weight.shape, (2,2))
        
class TestWeightInit(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            pc.weight_init('nomethod')
        self.assertEqual(pc.weight_init(np.array([1])), np.array([1]))
    
    def test_random(self):
        weight = pc.weight_init('random', input_dimensions = 2, latent_dimensions = 1)
        self.assertEqual(weight.shape, (2,1))
        self.assertAlmostEqual(np.matmul(weight.T, weight)[0,0], 1)
    
    def test_pca(self):
        weight = pc.weight_init('pca', latent_dimensions = 1, input_data = np.array([[1,0],[0,1]]))
        self.assertEqual(weight.shape, (2,1))