"""Tests predicode.datasets.test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import predicode as pc

class TestHex2(unittest.TestCase):
    """Tests the hex_2 function."""
    def test_base(self):
        """Tests base functionality."""
        self.assertEqual(pc.hex_2(2), '02')
        self.assertEqual(pc.hex_2(30), '1e')
        self.assertEqual(pc.hex_2(0), '00')
        with self.assertRaises(ValueError):
            pc.hex_2(16**3)
        with self.assertRaises(ValueError):
            pc.hex_2(-1)
        with self.assertRaises(TypeError):
            pc.hex_2('01')
        with self.assertRaises(TypeError):
            pc.hex_2(1.5)

class TestImageData(unittest.TestCase):
    """Tests the Imagedata class."""
    def setUp(self):
        """Sets up labelled and unlabelled examples."""
        array = np.array(
            [[[[59, 62, 63],
               [43, 46, 45],
               [50, 48, 43]],

              [[16, 20, 20],
               [0, 0, 0],
               [18, 8, 0]],

              [[25, 24, 21],
               [16, 7, 0],
               [49, 27, 8]]],


             [[[154, 177, 187],
               [126, 137, 136],
               [105, 104, 95]],

              [[140, 160, 169],
               [145, 153, 154],
               [125, 125, 118]],

              [[140, 155, 164],
               [139, 146, 149],
               [115, 115, 112]]]],
            dtype=np.uint8)
        self.labelled_data = pc.ImageData(array, labels=['a', 'b'])
        self.unlabelled_data = pc.ImageData(array)
        self.dataframe = self.labelled_data.dataframe()
        self.rgb_dataframe = self.labelled_data.rgb_dataframe()

    def test_init(self):
        """Tests appropriate handling of faulty handling."""
        with self.assertRaises(ValueError):
            pc.ImageData(np.array([], ndmin=5))
        with self.assertRaises(AttributeError):
            pc.ImageData('a')

    def test_dataframe(self):
        """Tests dataframe method."""
        self.assertEqual(len(self.dataframe), 18)
        self.assertEqual(len(self.rgb_dataframe), 18)
        self.assertEqual(len(self.labelled_data.dataframe(subset=[1])), 9)
        self.assertEqual(len(self.labelled_data.dataframe(n_random=1)), 9)
        self.assertEqual(len(self.labelled_data.dataframe(n_random=10)), 18)

    def test_picture(self):
        """Tests that pictures can be created and fail correctly.

        Does not test how these pictures look."""
        with self.assertRaises(NotImplementedError):
            self.labelled_data.pictures(mode='notimplemented')
        self.labelled_data.pictures(mode='color')
        self.unlabelled_data.pictures(mode='bw')

    def test_print(self):
        """Tests that Imagedata prints correctly."""
        msg = self.unlabelled_data.__str__()
        self.assertEqual(msg, "2 3x3-pictures.")
