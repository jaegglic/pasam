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
    def test_ConditionText_print(self):
        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        condition = ConditionFile(file)

        self.assertTrue(hasattr(condition, '__repr__'))
        self.assertTrue(condition.__repr__())
        self.assertTrue(hasattr(condition, '__str__'))
        self.assertTrue(condition.__str__())

    def test_ConditionText_gen(self):
        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        condition = ConditionFile(file)
        self.assertTrue(isinstance(condition, Condition))
        self.assertTrue(isinstance(condition, ConditionFile))

    def test_ConditionText_make_latticemap(self):
        nodes = [np.arange(-179, 181, 2), np.arange(-89, 91, 2)]
        nodes_wrong = [np.arange(-89, 90, 1), np.arange(-44, 44.5, 0.5)]
        lattice = Lattice(nodes)
        lattice_wrong = Lattice(nodes_wrong)

        file = PATH_TESTFILES + 'latticemap2d_int.txt'
        condition = ConditionFile(file)
        cond_latmap = condition.make_condmap(lattice)
        self.assertTrue(isinstance(cond_latmap, LatticeMap))
        self.assertTrue(isinstance(condition, Condition))
        self.assertTrue(isinstance(condition, ConditionFile))
        with self.assertRaises(ValueError):
            condition.make_condmap(lattice_wrong)

    def test_ConditionText_condmap2D_simple(self):
        nodes2D = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
        lattice2D = Lattice(nodes2D)
        map_vals2D = [
            [True, False, False, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, False, False, True],
        ]
        latticemap2D_true = LatticeMap(lattice2D, map_vals2D)

        file = PATH_TESTFILES + 'condmap2d_simple.txt'
        condition = ConditionFile(file)
        cond_latmap = condition.make_condmap(lattice2D)

        self.assertTrue(isinstance(cond_latmap, LatticeMap))
        self.assertTrue(cond_latmap.map_vals.dtype == 'bool')
        self.assertEqual(latticemap2D_true, cond_latmap)
        self.assertTrue(latticemap2D_true == cond_latmap)

    def test_ConditionText_latticemap3D_object(self):
        nodes3D = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11]]
        lattice3D = Lattice(nodes3D)
        map_vals3D = [
            [True, False, False, False, False, True, False],
            [True, True, False, True, False, True, True],
            [True, False, True, False, False, False, True],
        ]
        latticemap3D_true = LatticeMap(lattice3D, map_vals3D)

        file = PATH_TESTFILES + 'condmap3d_simple.txt'
        condition = ConditionFile(file)
        cond_latmap = condition.make_condmap(lattice3D)

        self.assertTrue(isinstance(cond_latmap, LatticeMap))
        self.assertTrue(cond_latmap.map_vals.dtype == 'bool')
        self.assertEqual(latticemap3D_true, cond_latmap)
        self.assertTrue(latticemap3D_true == cond_latmap)

    # Tests associated to ConditionPoint
    def test_ConditionPoint_print(self):
        components = (0, 1, -1)
        condition = ConditionPoint(components)

        self.assertTrue(hasattr(condition, '__repr__'))
        self.assertTrue(condition.__repr__())
        self.assertTrue(hasattr(condition, '__str__'))
        self.assertTrue(condition.__str__())

    def test_ConditionPoint_gen(self):
        components = (0, 1, -1)
        condition = ConditionPoint(components)

        self.assertTrue(isinstance(condition, Condition))
        self.assertTrue(isinstance(condition, ConditionPoint))

    def test_ConditionPoint__eq__(self):
        components = (0, 1, -1)
        condition = ConditionPoint(components)
        condition_is = condition
        condition_eq = ConditionPoint(components)
        condition_noneq = ConditionPoint([5, 1, -1])

        self.assertTrue(condition is condition_is)
        self.assertTrue(condition == condition_is)
        self.assertTrue(condition == condition_eq)
        self.assertTrue(condition == (0, 1, -1))
        self.assertTrue(condition == [0, 1, -1])
        self.assertFalse(condition is condition_eq)
        self.assertFalse(condition is condition_noneq)
        self.assertFalse(condition == condition_noneq)
        self.assertFalse(condition == (5, 1, -1))
        self.assertFalse(condition == [5, 1, -1])

    def test_ConditionPoint__len__(self):
        condition1D_single = ConditionPoint(1)
        condition1D_tuple = ConditionPoint((1,))
        condition2D = ConditionPoint((1, 2.))
        condition3D = ConditionPoint((3, 4.5, 6))

        self.assertTrue(len(condition1D_single) == 1)
        self.assertTrue(len(condition1D_tuple) == 1)
        self.assertTrue(len(condition2D) == 2)
        self.assertTrue(len(condition3D) == 3)


if __name__ == '__main__':
    unittest.main()
