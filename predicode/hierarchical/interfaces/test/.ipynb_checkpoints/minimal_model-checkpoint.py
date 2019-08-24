import unittest
import predicode as pc
import numpy as np
import warnings

class TestMinimalModelState(unittest.TestCase):
    def setUp(self):
        self.model = pc.MinimalHierarchicalModel(np.array([[1,1],[0,1]]), latent_dimensions = 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.train(steps = 10)
    
    def test_activate(self):
        self.assertEqual(self.model.what, 'state')
        with self.assertRaises(AssertionError):
            self.model.activate('no')
    
    def test_eval(self):
        self.assertAlmostEqual(np.round(self.model.evaluate()['loss'], 1), 0.5)
    
    def test_predict(self):
        prediction = self.model.predict()
        self.assertEqual(prediction.shape, (2,2))
    
    def test_latent_values(self):
        latent_values = self.model.latent_values()
        self.assertEqual(latent_values.shape, (2,1))

class TestMinimalModelWeight(unittest.TestCase):
    def setUp(self):
        self.model = pc.MinimalHierarchicalModel(np.array([[1,1],[0,1]]), latent_dimensions = 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.train(steps = 10)
        self.model.activate('weight')
    
    def test_train(self):
        with self.assertRaises(NotImplementedError):
            self.model.train()
    
    def test_eval(self):
        with self.assertRaises(NotImplementedError):
            self.model.evaluate()
    
    def test_predict(self):
        with self.assertRaises(NotImplementedError):
            self.model.predict()