"""Tests predicode.datasets.artificial_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import predicode as pc

class TestDecayingMultiNormal(unittest.TestCase):
    """Tests decaying_multi_normal function."""
    def setUp(self):
        """Sets up an example dataset."""
        self.art = pc.decaying_multi_normal(2, 1)

    def test_init(self):
        """Tests that initialization of decaying_multi_normal function works."""
        with self.assertRaises(ValueError):
            pc.decaying_multi_normal(2, 2, -1)
        self.assertEqual(self.art.shape, (1, 2))
