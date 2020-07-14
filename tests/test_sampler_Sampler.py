#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the Sampler* classes in `pasam.sampling.py`.
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
from pasam._settings import NP_ATOL, NP_RTOL
from pasam.lattice import Lattice, LatticeMap
from pasam.sampling import Sampler, GantryDominant2D

# Constants
_LOC = os.path.dirname(os.path.abspath(__file__))
PATH_TESTFILES = os.path.join(_LOC, 'testfiles', '')


class TestSampler(unittest.TestCase):

    def setUp(self):
        pass

    def test_GantryDominant2D_gen(self):
        nodes = [[-1.5, 1.5, 5, 8, 9], [1, 2, 3, 4, 5, 6]]
        lattice = Lattice(nodes)
        sampler = GantryDominant2D(lattice, ratio=2.0)

        self.assertTrue(isinstance(sampler, Sampler))
        self.assertTrue(isinstance(sampler, GantryDominant2D))

    def test_GantryDominant2D_adjacency_graph(self):
        nodes = [[1, 2, 3], [-1, 0, 1]]
        lattice = Lattice(nodes)

        ratio = 1.0
        sampler = GantryDominant2D(lattice, ratio=ratio)

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
        graph_test = sampler._adjacency_graph()
        self.assertTrue(np.all(graph_true == graph_test))

        ratio = 2.0
        sampler = GantryDominant2D(lattice, ratio=ratio)

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
        graph_test = sampler._adjacency_graph()
        self.assertTrue(np.all(graph_true == graph_test))

        nodes = [[1, 2, 3], [-0.5, 0, 0.5]]
        lattice = Lattice(nodes)

        ratio = 1.0
        sampler = GantryDominant2D(lattice, ratio=ratio)

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
        graph_test = sampler._adjacency_graph()
        self.assertTrue(np.all(graph_true == graph_test))

    def test_GantryDominant2D_set_prior_cond_with_map(self):
        nodes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5]]
        lattice = Lattice(nodes)
        values = np.array([
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        ], dtype=bool)
        conditions = [LatticeMap(nodes, values)]

        values_true = np.array([
            1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 2.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions, validate=True)
        self.assertEqual(sampler._prior_cond, prior_cond_true)

        values_true = np.array([
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 1.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions, validate=True)
        self.assertEqual(sampler._prior_cond, prior_cond_true)

    def test_GantryDominant2D_set_prior_cond_with_str(self):
        nodes = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
        lattice = Lattice(nodes)
        conditions = [PATH_TESTFILES + 'condmap2d_simple.txt']

        values_true = np.array([
            1,  1,  1,
            1,  1,  1,
            1,  1,  1,
            1,  1,  0,
            1,  0,  0,
            0,  0,  0,
            0,  0,  0,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 1.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions)

        self.assertEqual(sampler._prior_cond, prior_cond_true)

        # import matplotlib.pyplot as plt
        # from pasam._paths import PATH_EXAMPLES
        # nodes = [np.arange(-179, 181, 2), np.arange(-89, 91, 2)]
        # lattice = Lattice(nodes)
        # conditions = [PATH_EXAMPLES + 'restrictions_180x90.txt']
        #
        # ratio = 0.5
        # sampler = GantryDominant2D(lattice, ratio=ratio)
        # sampler.set_prior_cond(conditions)
        # plt_values = np.reshape(sampler._prior_cond.values,
        #                         lattice.nnodes_dim, order='F').transpose()
        # plt.imshow(plt_values, origin='lower')
        # plt.show()

    def test_GantryDominant2D__cond_map_from_point(self):
        nodes = [[-2, -1, 0, 1, 2], [-2, 0, 2]]
        lattice = Lattice(nodes)

        conditions = [(0, 0)]
        values_true = np.array([
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 2.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions)

        self.assertEqual(sampler._prior_cond, prior_cond_true)

        conditions = [(-2, -2)]
        values_true = np.array([
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 1,
            0, 0, 1, 1, 1,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 2.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions)

        self.assertEqual(sampler._prior_cond, prior_cond_true)

        conditions = [(1, -2)]
        values_true = np.array([
            1, 1, 1, 1, 1,
            1, 1, 1, 0, 1,
            1, 1, 1, 0, 1,
        ], dtype=bool)
        prior_cond_true = LatticeMap(nodes, values_true)

        ratio = 4.0
        sampler = GantryDominant2D(lattice, ratio=ratio)
        sampler.set_prior_cond(conditions)

        self.assertEqual(sampler._prior_cond, prior_cond_true)

    def test_GantryDominant2D__call__conditioning(self):
        # Computational lattice
        nodes = [
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 6],
        ]
        lattice = Lattice(nodes)

        # Sampler
        ratio = 2.0
        sampler = GantryDominant2D(lattice=lattice, ratio=ratio)

        # Sample Trajectory
        trajectory = sampler(conditions=[(0, 3)])

        # Check length
        self.assertTrue(len(trajectory), len(nodes[0]))

        # Check conditioing
        self.assertEqual(trajectory[4], (0, 3))

        # Check max trajectory spreading
        pt_last = trajectory[0]
        self.assertEqual(pt_last[0], nodes[0][0])
        for pt_next, nd_next in zip(trajectory[1:], nodes[0][1:]):
            self.assertEqual(pt_next[0], nd_next)
            max_spread = abs(pt_next[0]-pt_last[0])*ratio
            max_spread += NP_ATOL + NP_RTOL*max_spread
            self.assertTrue(abs(pt_next[1]-pt_last[1]) <= max_spread)
            pt_last = pt_next
