import unittest
import predicode as pc

class TestDecayingMultiNormal(unittest.TestCase):
    def setUp(self):
        self.art = pc.DecayingMultiNormal(2,1)
    
    def test_init(self):
        with self.assertRaises(AssertionError):
            pc.DecayingMultiNormal(2,2,-1)
        self.assertEqual(self.art.data.shape, (1,2))