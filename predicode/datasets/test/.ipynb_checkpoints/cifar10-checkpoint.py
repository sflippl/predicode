"""Tests predicode.dataset.cifar10"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import predicode as pc

class TestCifar10(unittest.TestCase):
    """Tests CIFAR-10 API."""

    def setUp(self):
        """Import CIFAR-10."""
        self.cifar10 = pc.Cifar10()

    def test_labels(self):
        """Test that labels are correct."""
        self.assertTrue(
            self.cifar10.labels['label_text'][0] in pc.Cifar10.labels
        )
