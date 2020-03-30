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
from pasam.lattice import LatticeMap
from pasam.sampling import Sampler, SamplerFactory, SamplerGantryDominant2D


class TestSampler(unittest.TestCase):

    def setUp(self):
        self.points = [(1, 2, 3), (4, 5, 6)]

    def test_SamplerFactory_simple(self):
        pass
        # import matplotlib.pylab as plt
        # file = PATH_TESTFILES + 'latticemap2d_sampler.txt'
        # prior_map = LatticeMap.from_txt(file)
        #
        # factory = SamplerFactory
        # # TODO refactor traj_type to type_
        # sampler_settings = {'traj_type': 'GantryDominant2D', 'ratio': 2.0}
        # sampler = factory.make(prior_map, **sampler_settings)
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

