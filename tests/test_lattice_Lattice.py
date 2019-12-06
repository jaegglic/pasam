#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Lattice* classes in `pasam.lattice.lattice.py`.
"""

# Standard library
import unittest
# Third party requirements
# Local imports
from pasam.lattice import Lattice2D, Lattice3D


class TestLattice(unittest.TestCase):

    def setUp(self):
        pass

    def test_Lattice2D(self):
        ndim = 2
        x = [1, 2, 3, 4.5, 5, 8]
        y = [-1.5, -1, 0, 5.76]
        lattice = Lattice2D(x, y)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice3D(self):
        ndim = 3
        lattice = Lattice3D()

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())


if __name__ == '__main__':
    unittest.main()
