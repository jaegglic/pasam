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
from pasam.lattice import Lattice, Condition, ConditionFile, ConditionPoint
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
        cond_map = condition.make_latticemap(lattice)
        with self.assertRaises(ValueError):
            condition.make_latticemap(lattice_wrong)


if __name__ == '__main__':
    unittest.main()
