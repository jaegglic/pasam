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
        cond_latmap = condition.make_latticemap(lattice)
        self.assertTrue(isinstance(cond_latmap, LatticeMap))
        self.assertTrue(isinstance(condition, Condition))
        self.assertTrue(isinstance(condition, ConditionFile))
        with self.assertRaises(ValueError):
            condition.make_latticemap(lattice_wrong)

    def test_ConditionText_latticemap2D_object(self):
        nodes2D = [[-1.5, 1.5, 5, 8, 9], [1, 2, 3, 4, 5, 6]]
        lattice2D = Lattice(nodes2D)
        map_vals2D = [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        latticemap2D_true = LatticeMap(lattice2D, map_vals2D)

        file = PATH_TESTFILES + 'latticemap2d_simple.txt'
        condition = ConditionFile(file)
        cond_latmap = condition.make_latticemap(lattice2D)

        self.assertEqual(latticemap2D_true, cond_latmap)
        self.assertTrue(latticemap2D_true == cond_latmap)
        self.assertTrue(isinstance(cond_latmap, LatticeMap))

    def test_ConditionText_latticemap3D_object(self):
        nodes3D = [[-1.5, 1.5], [5, 8, 9], [-2, 3]]
        lattice3D = Lattice(nodes3D)
        map_vals3D = [
            [0.5, 0.5],
            [0.8, 0.8],
            [0.1, 0.1],
            [0.6, 0.6],
            [0.9, 0.9],
            [0.2, 0.2],
        ]
        latticemap3D_true = LatticeMap(lattice3D, map_vals3D)

        file = PATH_TESTFILES + 'latticemap3d_simple.txt'
        condition = ConditionFile(file)
        cond_latmap = condition.make_latticemap(lattice3D)

        self.assertEqual(latticemap3D_true, cond_latmap)
        self.assertTrue(latticemap3D_true == cond_latmap)
        self.assertTrue(isinstance(cond_latmap, LatticeMap))


if __name__ == '__main__':
    unittest.main()
