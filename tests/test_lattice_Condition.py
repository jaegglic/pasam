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


if __name__ == '__main__':
    unittest.main()
