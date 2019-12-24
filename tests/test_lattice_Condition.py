#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the utilities in `pasam.utils.py`.
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
from pasam.lattice import (Lattice, LatticeMap,
                           Condition, ConditionFile, ConditionPoint)
from pasam._paths import PATH_TESTFILES


class TestCondition(unittest.TestCase):

    def setUp(self):
        pass

    # Tests associated to ConditionText
    def test_ConditionFile_print(self):
        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        cond_file = ConditionFile(file)

        self.assertTrue(hasattr(cond_file, '__repr__'))
        self.assertTrue(cond_file.__repr__())
        self.assertTrue(hasattr(cond_file, '__str__'))
        self.assertTrue(cond_file.__str__())

    def test_ConditionFile_gen(self):
        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        cond_file = ConditionFile(file)
        self.assertTrue(isinstance(cond_file, Condition))
        self.assertTrue(isinstance(cond_file, ConditionFile))

    def test_ConditionFile_make_latticemap(self):
        nodes = [np.arange(-179, 181, 2), np.arange(-89, 91, 2)]
        nodes_wrong = [np.arange(-89, 90, 1), np.arange(-44, 44.5, 0.5)]
        lattice = Lattice(nodes)
        lattice_wrong = Lattice(nodes_wrong)

        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        cond_file = ConditionFile(file)
        perm_map = cond_file.permission_map(lattice)
        self.assertTrue(isinstance(perm_map, LatticeMap))
        self.assertTrue(isinstance(cond_file, Condition))
        self.assertTrue(isinstance(cond_file, ConditionFile))
        with self.assertRaises(ValueError):
            cond_file.permission_map(lattice_wrong)

    def test_ConditionFile_condmap2D_simple(self):
        nodes2D = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
        lattice2D = Lattice(nodes2D)
        map_vals2D = [
            True, True, True,
            True, True, True,
            True, True, True,
            True, True, False,
            True, False, False,
            False, False, False,
            False, False, False,
        ]
        latticemap2D_true = LatticeMap(lattice2D, map_vals2D)

        file = PATH_TESTFILES + 'condmap2d_simple.txt'
        cond_file = ConditionFile(file)
        perm_map = cond_file.permission_map(lattice2D)

        self.assertTrue(isinstance(perm_map, LatticeMap))
        self.assertTrue(perm_map.map_vals.dtype == 'bool')
        self.assertEqual(latticemap2D_true, perm_map)
        self.assertTrue(latticemap2D_true == perm_map)

    def test_ConditionFile_condmap3D_simple(self):
        nodes3D = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11]]
        lattice3D = Lattice(nodes3D)
        map_vals3D = [
            True, True, True,
            False, True, False,
            False, False, True,
            False, True, False,
            False, False, False,
            True, True, False,
            False, True, True
        ]
        latticemap3D_true = LatticeMap(lattice3D, map_vals3D)

        file = PATH_TESTFILES + 'condmap3d_simple.txt'
        cond_file = ConditionFile(file)
        perm_map = cond_file.permission_map(lattice3D)

        self.assertTrue(isinstance(perm_map, LatticeMap))
        self.assertTrue(perm_map.map_vals.dtype == 'bool')
        self.assertEqual(latticemap3D_true, perm_map)
        self.assertTrue(latticemap3D_true == perm_map)

    # Tests associated to ConditionPoint
    def test_ConditionPoint_print(self):
        components = (0, 1, -1)
        cond_pnt = ConditionPoint(components)

        self.assertTrue(hasattr(cond_pnt, '__repr__'))
        self.assertTrue(cond_pnt.__repr__())
        self.assertTrue(hasattr(cond_pnt, '__str__'))
        self.assertTrue(cond_pnt.__str__())

    def test_ConditionPoint_gen(self):
        components = (0, 1, -1)
        cond_pnt = ConditionPoint(components)

        self.assertTrue(isinstance(cond_pnt, Condition))
        self.assertTrue(isinstance(cond_pnt, ConditionPoint))

    def test_ConditionPoint__eq__(self):
        components = (0, 1, -1)
        cond_pnt = ConditionPoint(components)
        condition_is = cond_pnt
        condition_eq = ConditionPoint(components)
        condition_noneq = ConditionPoint([5, 1, -1])

        self.assertTrue(cond_pnt is condition_is)
        self.assertTrue(cond_pnt == condition_is)
        self.assertTrue(cond_pnt == condition_eq)
        self.assertTrue(cond_pnt == (0, 1, -1))
        self.assertTrue(cond_pnt == [0, 1, -1])
        self.assertFalse(cond_pnt is condition_eq)
        self.assertFalse(cond_pnt is condition_noneq)
        self.assertFalse(cond_pnt == condition_noneq)
        self.assertFalse(cond_pnt == (5, 1, -1))
        self.assertFalse(cond_pnt == [5, 1, -1])

    def test_ConditionPoint__len__(self):
        cond_pnt_1D_tuple = ConditionPoint((1,))
        cond_pnt_2D = ConditionPoint((1, 2.))
        cond_pnt_3D = ConditionPoint((3, 4.5, 6))

        self.assertTrue(len(cond_pnt_1D_tuple) == 1)
        self.assertTrue(len(cond_pnt_2D) == 2)
        self.assertTrue(len(cond_pnt_3D) == 3)


if __name__ == '__main__':
    unittest.main()
