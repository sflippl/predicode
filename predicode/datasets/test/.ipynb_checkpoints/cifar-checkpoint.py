import unittest
import predicode as pc

class TestCifar10(unittest.TestCase):
    def setUp(self):
        self.cifar10 = pc.Cifar10()
    
    def test_labels(self):
        self.assertTrue(self.cifar10.labels['label_text'][0] in pc.Cifar10.labels)