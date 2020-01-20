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
# Local imports
from pasam._paths import PATH_TESTFILES
from pasam.pathgen import Trajectory


class TestTrajectory(unittest.TestCase):

    def setUp(self):
        self.points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

    def test_Trajectory_print(self):
        traj = Trajectory(self.points)

        self.assertTrue(hasattr(traj, '__repr__'))
        self.assertTrue(traj.__repr__())
        self.assertTrue(hasattr(traj, '__str__'))
        self.assertTrue(traj.__str__())
        self.assertTrue(isinstance(traj.__str__(), str))

    def test_Trajectory_write_to_txt(self):
        pts = [(1.123, 3.453, 4.5),
               (5.675, 4, 89.9),
               (6.543, 47.89, -898.009)]
        traj_str_true = '3\n' \
                        '1.123\t3.453\t4.5\n' \
                        '5.675\t4\t89.9\n' \
                        '6.543\t47.89\t-898.009\n'

        traj = Trajectory(pts)
        fname = PATH_TESTFILES + 'traj_test.txt'

        traj.to_txt(fname)
        with open(fname, 'r') as tfile:
            traj_str_test = ''.join(tfile.readlines())

        self.assertEqual(traj_str_true, traj_str_test)


if __name__ == '__main__':
    unittest.main()
