#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Lattice* classes in `pasam.lattice.lattice.py`.
"""

# Standard library
import unittest
# Third party requirements
# Local imports
from pasam.lattice import LatticeFactory


class TestLattice(unittest.TestCase):

    def setUp(self):
        self.x = [1, 2, 3, 4.5, 5, 8]
        self.y = [-1.5, -1, 0, 5.76]
        self.z = [-100.1, -1, 1998.5]

        self.lat_fact = LatticeFactory()

    def test_Lattice2D(self):
        ndim = 2
        nodes = self.x, self.y
        lattice = self.lat_fact.make_lattice(nodes)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice3D(self):
        ndim = 3
        nodes = self.x, self.y, self.z
        lattice = self.lat_fact.make_lattice(nodes)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())


if __name__ == '__main__':
    unittest.main()
