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
# Third party requirements
import numpy as np
# Local imports
from pasam._paths import PATH_TESTFILES
from pasam.lattice import Lattice, LatticeMap
from pasam.lattice import Trajectory, TrajectoryPermission,\
    TrajectoryPermissionFactory, TrajectoryPermissionGantryDominant2D


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

    def test_TrajectoryPermissionFactory(self):
        specs = {
            'traj_type': 'GantryDominant2D',
            'ratio': 2.0,
        }
        traj_fact = TrajectoryPermissionFactory()
        traj_perm = traj_fact.make(**specs)

        self.assertTrue(isinstance(traj_perm, TrajectoryPermission))
        self.assertTrue(isinstance(traj_perm, TrajectoryPermissionGantryDominant2D))

    def test_TrajectoryPermissionGantryDominant2D_adjacency_graph(self):
        ratio = 1.0
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)

        nodes = [[1, 2, 3], [-1, 0, 1]]
        lattice = Lattice(nodes)

        graph_true = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        graph_test = traj_perm.adjacency_graph(lattice)
        self.assertTrue(np.all(graph_true == graph_test))

        ratio = 2.0
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)

        nodes = [[1, 2, 3], [-1, 0, 1]]
        lattice = Lattice(nodes)

        graph_true = np.array([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        graph_test = traj_perm.adjacency_graph(lattice)
        self.assertTrue(np.all(graph_true == graph_test))

        ratio = 1.0
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)

        nodes = [[1, 2, 3], [-0.5, 0, 0.5]]
        lattice = Lattice(nodes)

        graph_true = np.array([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        graph_test = traj_perm.adjacency_graph(lattice)
        self.assertTrue(np.all(graph_true == graph_test))

        ratio = 0.5
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)

        nodes = [[0, 2, 8, 9], [-3, -2, 0]]
        lattice = Lattice(nodes)

        graph_true = np.array([
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        graph_test = traj_perm.adjacency_graph(lattice)
        self.assertTrue(np.all(graph_true == graph_test))

    def test_TrajectoryPermissionGantryDominant2D_permission_from_map(self):
        nodes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5]]
        values = np.array([
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        ], dtype=bool)

        ratio = 2.0
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)
        latticemap = LatticeMap(nodes, values)
        values_true = np.array([
            1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        ], dtype=bool)

        values_test = traj_perm.permission_from_map(latticemap).map_vals
        self.assertTrue(np.all(values_test == values_true))

        ratio = 1.0
        traj_perm = TrajectoryPermissionGantryDominant2D(ratio)
        latticemap = LatticeMap(nodes, values)
        values_true = np.array([
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        ], dtype=bool)

        values_test = traj_perm.permission_from_map(latticemap).map_vals
        self.assertTrue(np.all(values_test == values_true))


if __name__ == '__main__':
    unittest.main()
