#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Lattice* classes in `pasam.lattice.py`.
"""

# -------------------------------------------------------------------------
#   Authors: Stefanie Marti and Christoph Jaeggli
#   Institute: Insel Data Science Center, Insel Gruppe AG
#
#   MIT License
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
import pasam._settings as settings
from pasam._paths import PATH_TESTFILES
from pasam.lattice import Lattice, LatticeMap

# Constants
_NP_ORDER = settings.NP_ORDER
_NP_SEED = 458967
np.random.seed(_NP_SEED)


class TestLattice(unittest.TestCase):

    def setUp(self):
        self.x = [1, 2, 3, 4.5, 5, 8]
        self.y = [-1.5, -1, 0, 5.76]
        self.z = [-100.1, -1, 1998.5]

        self.nodes2D = [self.x, self.y]
        self.nodes3D = [self.x, self.y, self.z]

        self.x_short = [1, 2, 3, 4.5, 5]
        self.x_non_eq = [5, 6, 7, 8, 9, 10]

    # Tests associated to Lattice2D
    def test_Lattice2D_gen(self):
        lattice = Lattice(self.nodes2D)
        self.assertTrue(isinstance(lattice, Lattice))
        with self.assertRaises(ValueError):
            Lattice([[1, 2, 3], [1, -1, 2]])

    def test_Lattice2D__eq__(self):
        lattice = Lattice(self.nodes2D)
        lattice_is = lattice
        lattice_eq = Lattice(self.nodes2D)
        lattice_non_eq = Lattice([self.x_non_eq, self.y])
        lattice_short = Lattice([self.x_short, self.y])

        self.assertTrue(lattice is lattice_is)
        self.assertTrue(lattice == lattice_eq)
        self.assertEqual(lattice, lattice_eq)
        self.assertFalse(lattice is lattice_eq)
        self.assertFalse(lattice == lattice_non_eq)
        self.assertFalse(lattice == lattice_short)

    def test_Lattice2D__getitem__(self):
        lattice = Lattice(self.nodes2D)
        nodes = lattice.nodes
        nnodes_dim = lattice.nnodes_dim

        nx_st, nx_end = 2, 4
        ny_st, ny_end = 0, 3

        lattice_x = lattice[nx_st, :]
        lattice_xslice = lattice[nx_st:nx_end, :]

        lattice_y = lattice[:, ny_st]
        lattice_yslice = lattice[:, ny_st:ny_end]

        lattice_xyslice = lattice[nx_st:nx_end, ny_st:ny_end]

        self.assertTrue(lattice_x.nnodes_dim == (1, nnodes_dim[1]))
        self.assertTrue(lattice_x == Lattice([[nodes[0][nx_st]], nodes[1]]))
        self.assertTrue(lattice_xslice.nnodes_dim == (2, nnodes_dim[1]))
        self.assertTrue(lattice_xslice == Lattice([nodes[0][nx_st:nx_end], nodes[1]]))

        self.assertTrue(lattice_y.nnodes_dim == (nnodes_dim[0], 1))
        self.assertTrue(lattice_y == Lattice([nodes[0], [nodes[1][ny_st]]]))
        self.assertTrue(lattice_yslice.nnodes_dim == (nnodes_dim[0], 3))
        self.assertTrue(lattice_yslice == Lattice([nodes[0], nodes[1][ny_st:ny_end]]))

        self.assertTrue(lattice_xyslice == Lattice([nodes[0][nx_st:nx_end], nodes[1][ny_st:ny_end]]))

    def test_Lattice2D_ndim(self):
        ndim = 2
        lattice = Lattice(self.nodes2D)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)
        self.assertTrue(isinstance(lattice.ndim, int))

    def test_Lattice2D_print(self):
        lattice = Lattice(self.nodes2D)

        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice2D_nnodes(self):
        ndim = 2
        nnodes_dim = (6, 4)
        nnodes = 24
        lattice = Lattice(self.nodes2D)

        self.assertTrue(isinstance(lattice.nnodes_dim, tuple))
        self.assertEqual(len(lattice.nnodes_dim), ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertTrue(isinstance(lattice.nnodes, int))
        self.assertEqual(lattice.nnodes, nnodes)

    # Tests associated to Lattice3D
    def test_Lattice3D_gen(self):
        lattice = Lattice(self.nodes3D)
        self.assertTrue(isinstance(lattice, Lattice))
        with self.assertRaises(ValueError):
            Lattice([[1, 2, 3], [-1, 2], [1, -1, 2]])

    def test_Lattice3D__eq__(self):
        lattice = Lattice(self.nodes3D)
        lattice_is = lattice
        lattice_eq = Lattice(self.nodes3D)
        lattice_non_eq = Lattice([self.x_non_eq, self.y, self.z])
        lattice_short = Lattice([self.x_short, self.y, self.z])

        self.assertTrue(lattice is lattice_is)
        self.assertTrue(lattice == lattice_eq)
        self.assertEqual(lattice, lattice_eq)
        self.assertFalse(lattice is lattice_eq)
        self.assertFalse(lattice == lattice_non_eq)
        self.assertFalse(lattice == lattice_short)

    def test_Lattice3D__getitem__(self):
        lattice = Lattice(self.nodes3D)
        nodes = lattice.nodes
        nnodes_dim = lattice.nnodes_dim

        nx_st, nx_end = 2, 4
        ny_st, ny_end = 0, 3
        nz_st, nz_end = 2, 3

        lattice_z = lattice[:, :, nz_st]
        lattice_zslice = lattice[:, :, nz_st:nz_end]

        lattice_xyzslice = lattice[nx_st:nx_end, ny_st:ny_end, nz_st:nz_end]

        self.assertTrue(lattice_z.nnodes_dim == (nnodes_dim[0], nnodes_dim[1], 1))
        self.assertTrue(lattice_z == Lattice([nodes[0], nodes[1], [nodes[2][nz_st]]]))
        self.assertTrue(lattice_zslice.nnodes_dim == (nnodes_dim[0], nnodes_dim[1], 1))
        self.assertTrue(lattice_zslice == Lattice([nodes[0], nodes[1], nodes[2][nz_st:nz_end]]))

        self.assertTrue(lattice_xyzslice == Lattice([nodes[0][nx_st:nx_end], nodes[1][ny_st:ny_end], nodes[2][nz_st:nz_end]]))

    def test_Lattice3D_ndim(self):
        ndim = 3
        lattice = Lattice(self.nodes3D)

        self.assertTrue(hasattr(lattice, 'ndim'))
        self.assertEqual(lattice.ndim, ndim)
        self.assertTrue(isinstance(lattice.ndim, int))

    def test_Lattice3D_print(self):
        lattice = Lattice(self.nodes3D)

        self.assertTrue(hasattr(lattice, '__repr__'))
        self.assertTrue(lattice.__repr__())
        self.assertTrue(hasattr(lattice, '__str__'))
        self.assertTrue(lattice.__str__())

    def test_Lattice3D_nnodes(self):
        ndim = 3
        nnodes_dim = (6, 4, 3)
        nnodes = 72
        lattice = Lattice(self.nodes3D)

        self.assertTrue(isinstance(lattice.nnodes_dim, tuple))
        self.assertEqual(len(lattice.nnodes_dim), ndim)
        self.assertEqual(lattice.nnodes_dim, nnodes_dim)
        self.assertTrue(isinstance(lattice.nnodes, int))
        self.assertEqual(lattice.nnodes, nnodes)

    # Tests associated to LatticeMap2D
    def test_LatticeMap2D_gen(self):
        lattice = Lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        self.assertTrue(isinstance(latticemap, LatticeMap))
        self.assertTrue(np.all(map_vals == latticemap.map_vals))
        with self.assertRaises(ValueError):
            LatticeMap(lattice, map_vals[:-1])
        with self.assertRaises(ValueError):
            LatticeMap(lattice, [])

    def test_LatticeMap2D__eq__(self):
        lattice = Lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)
        latticemap_is = latticemap
        latticemap_eq = LatticeMap(lattice, map_vals)
        latticemap_non_eq = LatticeMap(lattice, map_vals + 1)

        self.assertTrue(latticemap is latticemap_is)
        self.assertTrue(latticemap == latticemap_eq)
        self.assertEqual(latticemap, latticemap_eq)
        self.assertFalse(latticemap is latticemap_eq)
        self.assertFalse(latticemap == latticemap_non_eq)

    def test_LatticeMap2D__getitem__(self):
        lattice = Lattice(self.nodes2D)
        nodes = lattice.nodes
        nnodes_dim = lattice.nnodes_dim

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        nx_st, nx_end = 2, 4
        ny_st, ny_end = 0, 3
        ind_x = np.arange(nx_st, nx_end)
        ind_y = np.arange(ny_st, ny_end)

        latticemap_x = latticemap[nx_st, :]
        lin_ind_x = np.array([nx_st + iy*nnodes_dim[0]
                              for iy in range(nnodes_dim[1])])

        latticemap_xslice = latticemap[nx_st:nx_end, :]
        lin_ind_xslice = np.array([ix + iy*nnodes_dim[0]
                                   for iy in range(nnodes_dim[1])
                                   for ix in ind_x])

        latticemap_y = latticemap[:, ny_st]
        lin_ind_y = np.array([ix + ny_st*nnodes_dim[0]
                              for ix in range(nnodes_dim[0])])


        latticemap_yslice = latticemap[:, ny_st:ny_end]
        lin_ind_yslice = np.array([ix + iy*nnodes_dim[0]
                                   for iy in ind_y
                                   for ix in range(nnodes_dim[0])])

        latticemap_xyslice = latticemap[nx_st:nx_end, ny_st:ny_end]
        lin_ind_xyslice = np.array([ix + iy*nnodes_dim[0]
                                    for iy in ind_y
                                    for ix in ind_x])

        self.assertTrue(latticemap_x == LatticeMap(lattice=lattice[nx_st, :],
                                                   map_vals=map_vals[lin_ind_x]))
        self.assertTrue(latticemap_xslice == LatticeMap(lattice=lattice[nx_st:nx_end, :],
                                                        map_vals=map_vals[lin_ind_xslice]))
        self.assertTrue(latticemap_y == LatticeMap(lattice=lattice[:, ny_st],
                                                   map_vals=map_vals[lin_ind_y]))
        self.assertTrue(latticemap_yslice == LatticeMap(lattice=lattice[:, ny_st:ny_end],
                                                        map_vals=map_vals[lin_ind_yslice]))
        self.assertTrue(latticemap_xyslice == LatticeMap(lattice=lattice[nx_st:nx_end, ny_st:ny_end],
                                                         map_vals=map_vals[lin_ind_xyslice]))

    def test_LatticeMap2D__add__(self):
        lattice = Lattice(self.nodes2D)
        nodes_short = [self.x_short, self.y]
        lattice_short = Lattice(nodes_short)
        num = -25.89

        map_vals_left = np.random.randn(lattice.nnodes)
        map_vals_right = np.random.randn(lattice.nnodes)
        map_vals_short = np.random.randn(lattice_short.nnodes)
        latticemap_left = LatticeMap(lattice, map_vals_left)
        latticemap_right = LatticeMap(lattice, map_vals_right)
        latticemap_short = LatticeMap(lattice_short, map_vals_short)
        latticmap_sum = LatticeMap(lattice, map_vals_left + map_vals_right)
        latticmap_sum_num = LatticeMap(lattice, map_vals_left + num)

        self.assertEqual(latticmap_sum, latticemap_left + latticemap_right)
        self.assertTrue(latticmap_sum == latticemap_left + latticemap_right)
        self.assertEqual(latticmap_sum_num, latticemap_left + num)
        self.assertTrue(latticmap_sum_num == latticemap_left + num)
        with self.assertRaises(ValueError):
            latticemap_left + latticemap_short
        with self.assertRaises(TypeError):
            latticemap_left + 'foobar'

    def test_LatticeMap2D__mul__(self):
        lattice = Lattice(self.nodes2D)
        num = -25.89

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)
        latticmap_mul = LatticeMap(lattice, map_vals * num)
        latticemap_squared = LatticeMap(lattice, map_vals**2)

        self.assertEqual(latticmap_mul, latticemap * num)
        self.assertTrue(latticmap_mul == latticemap * num)
        self.assertEqual(latticmap_mul, num * latticemap)
        self.assertTrue(latticmap_mul ==  num * latticemap)
        self.assertEqual(latticemap * latticemap, latticemap_squared)
        with self.assertRaises(TypeError):
            latticemap * 'foobar'

    def test_LatticeMap2D_ndim(self):
        ndim = 2
        lattice = Lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, 'ndim'))
        self.assertEqual(latticemap.ndim, ndim)
        self.assertTrue(isinstance(latticemap.ndim, int))

    def test_LatticeMap2D_print(self):
        lattice = Lattice(self.nodes2D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, '__repr__'))
        self.assertTrue(latticemap.__repr__())
        self.assertTrue(hasattr(latticemap, '__str__'))
        self.assertTrue(latticemap.__str__())

    def test_LatticeMap2D_make_latticemap_from_txt(self):
        nodes = [[-1.5, 1.5, 5, 8, 9], [1, 2, 3, 4, 5, 6]]
        lattice = Lattice(nodes)
        map_vals = [
            0.5, 0.5, 0.5, 0.5, 0.5,
            0.6, 0.6, 0.6, 0.6, 0.6,
            0.7, 0.7, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7, 0.7,
            0.6, 0.6, 0.6, 0.6, 0.6,
            0.5, 0.5, 0.5, 0.5, 0.5,
        ]
        latticemap_true = LatticeMap(lattice, map_vals)

        file = PATH_TESTFILES + 'latticemap2d_simple.txt'
        latticemap_test = LatticeMap.from_txt(file)

        self.assertTrue(isinstance(latticemap_test, LatticeMap))
        self.assertEqual(latticemap_true, latticemap_test)

    # Tests associated to LatticeMap3D
    def test_LatticeMap3D_gen(self):
        lattice = Lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        self.assertTrue(isinstance(latticemap, LatticeMap))
        self.assertTrue(np.all(map_vals == latticemap.map_vals))
        with self.assertRaises(ValueError):
            LatticeMap(lattice, map_vals[:-1])
        with self.assertRaises(ValueError):
            LatticeMap(lattice, [])

    def test_LatticeMap3D__eq__(self):
        lattice = Lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)
        latticemap_is = latticemap
        latticemap_eq = LatticeMap(lattice, map_vals)
        latticemap_non_eq = LatticeMap(lattice, map_vals + 1)

        self.assertTrue(latticemap is latticemap_is)
        self.assertTrue(latticemap == latticemap_eq)
        self.assertEqual(latticemap, latticemap_eq)
        self.assertFalse(latticemap is latticemap_eq)
        self.assertFalse(latticemap == latticemap_non_eq)

    def test_LatticeMap3D__getitem__(self):
        lattice = Lattice(self.nodes3D)
        nodes = lattice.nodes
        nnodes_dim = lattice.nnodes_dim

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        nx_st, nx_end = 2, 4
        ny_st, ny_end = 0, 3
        nz_st, nz_end = 2, 3
        ind_x = np.arange(nx_st, nx_end)
        ind_y = np.arange(ny_st, ny_end)
        ind_z = np.arange(nz_st, nz_end)

        latticemap_z = latticemap[:, :, nz_st]
        lin_ind_z = np.array([ix + iy*nnodes_dim[0] + nz_st*nnodes_dim[0]*nnodes_dim[1]
                              for iy in range(nnodes_dim[1])
                              for ix in range(nnodes_dim[0])])

        latticemap_zslice = latticemap[:, :, nz_st:nz_end]
        lin_ind_zslice = np.array([ix + iy*nnodes_dim[0] + iz*nnodes_dim[0]*nnodes_dim[1]
                                   for iz in ind_z
                                   for iy in range(nnodes_dim[1])
                                   for ix in range(nnodes_dim[0])])

        latticemap_xyzslice = latticemap[nx_st:nx_end, ny_st:ny_end, nz_st:nz_end]
        lin_ind_xyzslice = np.array([ix + iy*nnodes_dim[0] + iz*nnodes_dim[0]*nnodes_dim[1]
                                     for iz in ind_z
                                     for iy in ind_y
                                     for ix in ind_x])
        self.assertTrue(latticemap_z == LatticeMap(lattice=lattice[:, :, nz_st],
                                                   map_vals=map_vals[lin_ind_z]))
        self.assertTrue(latticemap_zslice == LatticeMap(lattice=lattice[:, :, nz_st:nz_end],
                                                        map_vals=map_vals[lin_ind_zslice]))
        self.assertTrue(latticemap_xyzslice == LatticeMap(lattice=lattice[nx_st:nx_end, ny_st:ny_end, nz_st:nz_end],
                                                          map_vals=map_vals[lin_ind_xyzslice]))

    def test_LatticeMap3D__add__(self):
        lattice = Lattice(self.nodes3D)
        nodes_short = [self.x_short, self.y, self.z]
        lattice_short = Lattice(nodes_short)
        num = -25.89

        map_vals_left = np.random.randn(lattice.nnodes)
        map_vals_right = np.random.randn(lattice.nnodes)
        map_vals_short = np.random.randn(lattice_short.nnodes)
        latticemap_left = LatticeMap(lattice, map_vals_left)
        latticemap_right = LatticeMap(lattice, map_vals_right)
        latticemap_short = LatticeMap(lattice_short, map_vals_short)
        latticmap_sum = LatticeMap(lattice, map_vals_left + map_vals_right)
        latticmap_sum_num = LatticeMap(lattice, map_vals_left + num)

        self.assertEqual(latticmap_sum, latticemap_left + latticemap_right)
        self.assertTrue(latticmap_sum == latticemap_left + latticemap_right)
        self.assertEqual(latticmap_sum_num, latticemap_left + num)
        self.assertTrue(latticmap_sum_num == latticemap_left + num)
        with self.assertRaises(ValueError):
            latticemap_left + latticemap_short
        with self.assertRaises(TypeError):
            latticemap_left + 'foobar'

    def test_LatticeMap3D__mul__(self):
        lattice = Lattice(self.nodes3D)
        num = -25.89

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)
        latticmap_mul = LatticeMap(lattice, map_vals * num)
        latticemap_squared = LatticeMap(lattice, map_vals**2)

        self.assertEqual(latticmap_mul, latticemap * num)
        self.assertTrue(latticmap_mul == latticemap * num)
        self.assertEqual(latticmap_mul, num * latticemap)
        self.assertTrue(latticmap_mul ==  num * latticemap)
        self.assertEqual(latticemap * latticemap, latticemap_squared)
        with self.assertRaises(TypeError):
            latticemap * 'foobar'

    def test_LatticeMap3D_ndim(self):
        ndim = 3
        lattice = Lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        self.assertTrue(hasattr(latticemap, 'ndim'))
        self.assertEqual(latticemap.ndim, ndim)
        self.assertTrue(isinstance(latticemap.ndim, int))

    def test_LatticeMap3D_print(self):
        lattice = Lattice(self.nodes3D)

        map_vals = np.random.randn(lattice.nnodes)
        latticemap = LatticeMap(lattice, map_vals)

        # Force the implementation of __repr__() and __str__()
        self.assertTrue(hasattr(latticemap, '__repr__'))
        self.assertTrue(latticemap.__repr__())
        self.assertTrue(hasattr(latticemap, '__str__'))
        self.assertTrue(latticemap.__str__())

    def test_LatticeMap3D_make_latticemap_from_txt(self):
        nodes = [[-1.5, 1.5], [5, 8, 9], [-2, 3]]
        lattice = Lattice(nodes)
        map_vals = [
            0.5, 0.5,
            0.8, 0.8,
            0.1, 0.1,
            0.6, 0.6,
            0.9, 0.9,
            0.2, 0.2,
        ]
        latticemap_true = LatticeMap(lattice, map_vals)

        file = PATH_TESTFILES + 'latticemap3d_simple.txt'
        latticemap_test = LatticeMap.from_txt(file)

        self.assertTrue(isinstance(latticemap_test, LatticeMap))
        self.assertEqual(latticemap_true, latticemap_test)


if __name__ == '__main__':
    unittest.main()
