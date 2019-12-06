#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Lattice* classes in `pasam.lattice.lattice.py`.
"""

# Standard library
import unittest
# Third party requirements
import numpy as np
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
        nnodes_dim = (6, 4)
        nnodes = 24
        nodes = self.x, self.y
        lattice = self.lat_fact.make_lattice(nodes)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

        # Test ndim and nnodes* functions
        self.assertEqual(lattice.ndim, ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertEqual(lattice.nnodes, nnodes)

    def test_Lattice3D(self):
        ndim = 3
        nnodes_dim = (6, 4, 3)
        nnodes = 72
        nodes = self.x, self.y, self.z
        lattice = self.lat_fact.make_lattice(nodes)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

        # Test ndim and nnodes* functions
        self.assertEqual(lattice.ndim, ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertEqual(lattice.nnodes, nnodes)


if __name__ == '__main__':
    unittest.main()
