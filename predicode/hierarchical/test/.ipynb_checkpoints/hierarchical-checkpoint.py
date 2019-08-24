import unittest
import predicode as pc

class TestHierarchical(unittest.TestCase):
    def setUp(self):
        self.hierarchical = pc.Hierarchical()
    
    def test(self):
        self.assertIs(type(self.hierarchical), pc.Hierarchical)