#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Trajectory* classes in `pasam.lattice`.
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
import os
# Third party requirements
import numpy as np
# Local imports
from pasam.lattice import Lattice, LatticeMap
from pasam.sampling import Trajectory

# Constants
_LOC = os.path.dirname(os.path.abspath(__file__))
PATH_TESTFILES = os.path.join(_LOC, 'testfiles', '')


class TestTrajectory(unittest.TestCase):

    def setUp(self):
        pass

    def test_Trajectory_print(self):
        pts = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        traj = Trajectory(pts)

        self.assertTrue(hasattr(traj, '__repr__'))
        self.assertTrue(traj.__repr__())
        self.assertTrue(hasattr(traj, '__str__'))
        self.assertTrue(traj.__str__())
        self.assertTrue(isinstance(traj.__str__(), str))

    def test_Trajectory_write_read_txt(self):
        pts = [(1.123, 3.453, 4.5),
               (5.675, 4, 89.9),
               (6.543, 47.89, -898.009)]
        traj_str_true = '3\n' \
                        '1.123\t3.453\t4.5\n' \
                        '5.675\t4\t89.9\n' \
                        '6.543\t47.89\t-898.009\n'

        traj_true = Trajectory(pts)
        fname = PATH_TESTFILES + 'traj_test.txt'

        traj_true.to_txt(fname)
        with open(fname, 'r') as tfile:
            traj_str_test = ''.join(tfile.readlines())

        self.assertEqual(traj_str_true, traj_str_test)

        traj_read = Trajectory.from_txt(fname)
        self.assertTrue(traj_read == traj_true)     # Also tests __eq__

    def test_Trajectory_to_latticemap(self):
        nodes = [[-2, -1, 0, 1, 2], [-3, 1, 5, 9]]
        lattice = Lattice(nodes)
        pts = [(-2, 1), (-1, -4), (0, 3), (1, 4), (2, 8.8)]

        values = np.array([
            0, 1, 0, 0, 0,
            1, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1,
        ], dtype=bool)

        map_test = Trajectory(pts).to_latticemap(lattice)
        map_true = LatticeMap(lattice, values)
        self.assertEqual(map_true, map_test)


if __name__ == '__main__':
    unittest.main()
