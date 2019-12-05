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
        self.assertEqual(lattice_map.ndim, ndim)

    def test_LatticeMap3D(self):
        ndim = 3
        lattice_map = LatticeMap3D()
        self.assertEqual(lattice_map.ndim, ndim)


if __name__ == '__main__':
    unittest.main()
