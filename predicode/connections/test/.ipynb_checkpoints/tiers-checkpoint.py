"""Tests predicode.connections.tiers.
"""

import tempfile
import unittest

import numpy as np
import h5py

import predicode as pc

class TestTier(unittest.TestCase):
    """Tests the creation of a default tier and its transformation to an hdf5
    file.
    """

    def setUp(self):
        """Create small example tier.
        """
        self.tier = pc.connections.Tier(shape=(2, ))
        self.arr = np.array([[0.1, -0.1], [0.2, 0.1]])

    def test_shape(self):
        """If the shape is retained correctly.
        """
        self.assertEqual(self.tier.shape, (2, ))

    def test_setitem(self):
        """If an item can be set and fails appropriately.
        """
        self.tier['state'] = self.arr
        for state_entry, array_entry in zip(self.tier['state'].flatten(),
                                            self.arr.flatten()):
            self.assertEqual(state_entry, array_entry)
        with self.assertRaises(ValueError):
            self.tier['prediction'] = np.array([[0.1], [0.2]])

    def test_hdf5(self):
        """Does the hdf5 format work?
        """
        self.tier['state'] = self.arr
        filepath = '%s.hdf5' % (tempfile.mkdtemp(), )
        file = h5py.File(filepath)
        file.create_group('tier')
        self.tier.to_hdf5(file['tier'])
        self.assertEqual(file['tier'].attrs['shape'][0], 2)
        self.assertEqual(file['tier']['state'][0, 0], 0.1)
