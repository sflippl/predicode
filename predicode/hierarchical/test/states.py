"""Tests predicode.hierarchical.states.
"""

import unittest
import tempfile

import numpy as np

import predicode as pc

class TestStates(unittest.TestCase):
    """Tests the creation of states and saving them as hdf5 files.
    """

    def setUp(self):
        """Set up states or hdf5 states.
        """
        self.states = pc.States()
        self.fill_states()

    def fill_states(self):
        """Fills a rudimentary two-tier state.
        """
        self.states.add_tier((2, ))
        self.states.add_tier((1, ), 'latent_layer')
        self.states['tier_0']['state'] = np.array([[0, 0]])

    def test_state_entry(self):
        """Tests entry access options.
        """
        self.assertEqual(self.states['tier_0']['state'][0, 0], 0)
        self.assertEqual(self.states[0]['state'][0, 0], 0)
        self.assertEqual(len(self.states['latent_layer']), 0)

    def test_no_tier(self):
        """Makes sure that only tiers can be set.
        """
        with self.assertRaises(ValueError):
            self.states['latent_layer'] = 'str'

    def test_add_same_tier(self):
        """Makes sure tiers must have distinct names.
        """
        with self.assertRaises(ValueError):
            self.states.add_tier((1, ), tier_name='latent_layer')

    def test_n_tiers(self):
        """Tests n_tiers.
        """
        self.assertEqual(self.states.n_tiers, 2)

class TestHDF5States(TestStates):
    """Tests hdf5 states.
    """

    def setUp(self):
        self.path = '%s.hdf5' % (tempfile.mkdtemp(), )
        self.states = pc.HDF5States(name=self.path)
        self.fill_states()

    def test_to_hdf5(self):
        """Tests if hdf5 states raise the appropriate error when being converted
        to hdf5.
        """
        with self.assertRaises(NotImplementedError):
            self.states.to_hdf5('anyfile')

class TestSavedAsHDF5(TestStates):
    """Tests states saved as hdf5 states.
    """

    def setUp(self):
        self.path = '%s.hdf5' % (tempfile.mkdtemp(), )
        super().setUp()
        self.states = self.states.to_hdf5(self.path)