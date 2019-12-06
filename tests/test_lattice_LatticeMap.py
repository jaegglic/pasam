#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the LatticeMap* classes in `pasam.lattice.lattice.py`.
"""

# Standard library
import unittest
# Third party requirements
# Local imports
from pasam.lattice import LatticeMap2D, LatticeMap3D


class TestLatticeMap(unittest.TestCase):

    def setUp(self):
        pass

    def test_LatticeMap2D(self):
        ndim = 2
        lattice_map = LatticeMap2D()

        self.assertTrue(hasattr(lattice_map, 'ndim'))
        self.assertEqual(lattice_map.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice_map, '__repr__'))
        self.assertTrue(lattice_map.__repr__())
        self.assertTrue(hasattr(lattice_map, '__str__'))
        self.assertTrue(lattice_map.__str__())

    def test_LatticeMap3D(self):
        ndim = 3
        lattice_map = LatticeMap3D()

        self.assertTrue(hasattr(lattice_map, 'ndim'))
        self.assertEqual(lattice_map.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice_map, '__repr__'))
        self.assertTrue(lattice_map.__repr__())
        self.assertTrue(hasattr(lattice_map, '__str__'))
        self.assertTrue(lattice_map.__str__())


if __name__ == '__main__':
    unittest.main()
