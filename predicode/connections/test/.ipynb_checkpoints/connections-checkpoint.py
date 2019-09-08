"""Tests predicode.connections.connections.
"""

import unittest

import predicode as pc

class TestNoTierConnection(unittest.TestCase):
    """Test the placeholder for a tier connection.
    """

    def test_methods_fail(self):
        """Test that NoTierConnection cannot train a model.
        """
        with self.assertRaises(ValueError):
            pc.connections.NoTierConnection().predict(None, None)
        with self.assertRaises(ValueError):
            pc.connections.NoTierConnection().prediction_error(None, None, None)
        with self.assertRaises(ValueError):
            pc.connections.NoTierConnection().compute_loss(None, None, None)
