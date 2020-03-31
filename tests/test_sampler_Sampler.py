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
# Third party requirements
import numpy as np
# Local imports
from pasam._paths import PATH_TESTFILES
from pasam._settings import NP_ORDER
from pasam.lattice import Lattice, LatticeMap
from pasam.sampling import Sampler, GantryDominant2D


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
        sampler.set_prior_cond(conditions)
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
        sampler.set_prior_cond(conditions)
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
        # conditions = [PATH_EXAMPLES + 'example_condition_2D.txt']
        #
        # ratio = 0.5
        # sampler = GantryDominant2D(lattice, ratio=ratio)
        # sampler.set_prior_cond(conditions)
        # plt_values = np.reshape(sampler._prior_cond.map_vals,
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

    def test_SamplerFactory_simple(self):
        # TODO make this test happen
        pass
        # import matplotlib.pylab as plt
        # file = PATH_TESTFILES + 'latticemap2d_sampler.txt'
        # prior_map = LatticeMap.from_txt(file)
        #
        # factory = SamplerFactory
        # settings = {'traj_type': 'GantryDominant2D', 'ratio': 2.0}
        # sampler = factory.make(prior_map, **settings)
        # self.assertTrue(isinstance(sampler, Sampler))
        # self.assertTrue(isinstance(sampler, SamplerGantryDominant2D))
        #
        # traj = sampler()
        # print(traj.points)
        #
        # plt.imshow(np.reshape(prior_map.map_vals,
        #                       prior_map.lattice.nnodes_dim,
        #                       order=NP_ORDER).transpose(),
        #            origin='lower')
        #
        # traj_x = []
        # traj_y = []
        # for pt in traj.points:
        #     ind = prior_map.lattice.indices_from_point(pt)
        #     traj_x.append(ind[0])
        #     traj_y.append(ind[1])
        # plt.plot(traj_x, traj_y, 'r')
        #
        # plt.show()
        #
        # with self.assertRaises(ValueError):
        #     factory.make(prior_map, traj_type='NotDefined')

