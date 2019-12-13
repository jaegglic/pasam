#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Lattice* classes in `pasam.lattice.py`.
"""

# -------------------------------------------------------------------------
#   Authors: Stefanie Marti and Christoph Jaeggli
#   Institute: Insel Data Science Center, Insel Gruppe AG
#
#   Copyright (c) 2020 Stefanie Marti, Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Standard library
import unittest
# Third party requirements
import numpy as np
# Local imports
from pasam.lattice import (Lattice, LatticeFactory, Lattice2D, Lattice3D,
                           LatticeMap, LatticeMapFactory)
from pasam._paths import PATH_TESTFILES


class TestLattice(unittest.TestCase):

    def setUp(self):
        self.x = [[1, 2, 3], [4.5, 5, 8]]
        self.y = [-1.5, -1, 0, 5.76]
        self.z = [-100.1, -1, 1998.5]

        self.nodes2D = [self.x, self.y]
        self.nodes3D = [self.x, self.y, self.z]

        self.x_short = [1, 2, 3, 4.5, 5]
        self.x_non_eq = [5, 5, 5, 5, 5, 5]

        self.lat_fact = LatticeFactory()
        self.latmap_fact = LatticeMapFactory()

    # Tests associated to Lattice2D
    def test_Lattice2D__eq__(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)
        lattice_is = lattice
        lattice_eq = self.lat_fact.make_lattice(self.nodes2D)
        lattice_non_eq = self.lat_fact.make_lattice([self.x_non_eq, self.y])
        lattice_short = self.lat_fact.make_lattice([self.x_short, self.y])

        self.assertTrue(lattice is lattice_is)
        self.assertTrue(lattice == lattice_eq)
        self.assertEqual(lattice, lattice_eq)
        self.assertFalse(lattice is lattice_eq)
        self.assertFalse(lattice == lattice_non_eq)
        self.assertFalse(lattice == lattice_short)

    def test_Lattice2D_ndim(self):
        ndim = 2
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)
        self.assertTrue(isinstance(lattice.ndim, int))

    def test_Lattice2D_print(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice2D_type(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)
        self.assertTrue(isinstance(lattice, Lattice))
        self.assertTrue(isinstance(lattice, Lattice2D))

    def test_Lattice2D_nnodes(self):
        ndim = 2
        nnodes_dim = (6, 4)
        nnodes = 24
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        self.assertTrue(isinstance(lattice.nnodes_dim, tuple))
        self.assertEqual(len(lattice.nnodes_dim), ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertTrue(isinstance(lattice.nnodes, int))
        self.assertEqual(lattice.nnodes, nnodes)

    # Tests associated to Lattice3D
    def test_Lattice3D__eq__(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)
        lattice_is = lattice
        lattice_eq = self.lat_fact.make_lattice(self.nodes3D)
        lattice_non_eq = self.lat_fact.make_lattice([self.x_non_eq, self.y, self.z])
        lattice_short = self.lat_fact.make_lattice([self.x_short, self.y, self.z])

        self.assertTrue(lattice is lattice_is)
        self.assertTrue(lattice == lattice_eq)
        self.assertEqual(lattice, lattice_eq)
        self.assertFalse(lattice is lattice_eq)
        self.assertFalse(lattice == lattice_non_eq)
        self.assertFalse(lattice == lattice_short)

    def test_Lattice3D_ndim(self):
        ndim = 3
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)
        self.assertTrue(isinstance(lattice.ndim, int))

    def test_Lattice3D_print(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice3D_type(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)
        self.assertTrue(isinstance(lattice, Lattice))
        self.assertTrue(isinstance(lattice, Lattice3D))

    def test_Lattice3D_nnodes(self):
        ndim = 3
        nnodes_dim = (6, 4, 3)
        nnodes = 72
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        self.assertTrue(isinstance(lattice.nnodes_dim, tuple))
        self.assertEqual(len(lattice.nnodes_dim), ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertTrue(isinstance(lattice.nnodes, int))
        self.assertEqual(lattice.nnodes, nnodes)

    # Tests associated to LatticeMap2D
    def test_LatticeMap2D_gen(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        self.assertTrue(isinstance(latticemap, LatticeMap))
        self.assertTrue(np.all(map_vals == latticemap.map_vals))
        with self.assertRaises(ValueError):
            self.latmap_fact.make_latticemap(lattice, map_vals[:-1])
        with self.assertRaises(ValueError):
            self.latmap_fact.make_latticemap(lattice, [])

    def test_LatticeMap2D__eq__(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)
        latticemap_is = latticemap
        latticemap_eq = self.latmap_fact.make_latticemap(lattice, map_vals)
        latticemap_non_eq = self.latmap_fact.make_latticemap(lattice, map_vals + 1)

        self.assertTrue(latticemap is latticemap_is)
        self.assertTrue(latticemap == latticemap_eq)
        self.assertEqual(latticemap, latticemap_eq)
        self.assertFalse(latticemap is latticemap_eq)
        self.assertFalse(latticemap == latticemap_non_eq)

    def test_LatticeMap2D_ndim(self):
        ndim = 2
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, 'ndim'))
        self.assertEqual(latticemap.ndim, ndim)
        self.assertTrue(isinstance(latticemap.ndim, int))

    def test_LatticeMap2D_print(self):
        lattice = self.lat_fact.make_lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, '__repr__'))
        self.assertTrue(latticemap.__repr__())
        self.assertTrue(hasattr(latticemap, '__str__'))
        self.assertTrue(latticemap.__str__())

    def test_LatticeMap2D_make_latticemap_from_txt(self):
        nodes = [[-1.5, 1.5, 5, 8, 9], [1, 2, 3, 4, 5, 6]]
        lattice = self.lat_fact.make_lattice(nodes)
        map_vals = [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        latticemap_true = self.latmap_fact.make_latticemap(lattice, map_vals)

        txtfile = PATH_TESTFILES + 'latticemap2d_simple.txt'
        latticemap_test = self.latmap_fact.make_latticemap_from_txt(txtfile)

        self.assertEqual(latticemap_true, latticemap_test)

    # Tests associated to LatticeMap3D
    def test_LatticeMap3D_gen(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        self.assertTrue(isinstance(latticemap, LatticeMap))
        self.assertTrue(np.all(map_vals == latticemap.map_vals))
        with self.assertRaises(ValueError):
            self.latmap_fact.make_latticemap(lattice, map_vals[:-1])
        with self.assertRaises(ValueError):
            self.latmap_fact.make_latticemap(lattice, [])

    def test_LatticeMap3D__eq__(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)
        latticemap_is = latticemap
        latticemap_eq = self.latmap_fact.make_latticemap(lattice, map_vals)
        latticemap_non_eq = self.latmap_fact.make_latticemap(lattice, map_vals + 1)

        self.assertTrue(latticemap is latticemap_is)
        self.assertTrue(latticemap == latticemap_eq)
        self.assertEqual(latticemap, latticemap_eq)
        self.assertFalse(latticemap is latticemap_eq)
        self.assertFalse(latticemap == latticemap_non_eq)

    def test_LatticeMap3D_ndim(self):
        ndim = 3
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, 'ndim'))
        self.assertEqual(latticemap.ndim, ndim)
        self.assertTrue(isinstance(latticemap.ndim, int))

    def test_LatticeMap3D_print(self):
        lattice = self.lat_fact.make_lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = self.latmap_fact.make_latticemap(lattice, map_vals)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(latticemap, '__repr__'))
        self.assertTrue(latticemap.__repr__())
        self.assertTrue(hasattr(latticemap, '__str__'))
        self.assertTrue(latticemap.__str__())


if __name__ == '__main__':
    unittest.main()
