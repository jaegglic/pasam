#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXAMPLE FILE

This is a very simple 2D sampling example for the geometry

    ------------------------------------------------------
    |                                                    |
    |                                                    |
    |                                                    |
    |                                                    |
    |                                                    |
    |                                                    |
    |                                                    |
    |                                                    |
    ------------------------------------------------------

with uniform prior distribution.
"""
# TODO Write docstring of module

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
# Third party requirements
import numpy as np
import matplotlib.pylab as plt
# Local imports
import pasam as ps

# Computational Lattice and Prior Distribution
file_prior_energy = ps.PATH_EXAMPLES + 'example_prior_energy_2D.txt'
_, nodes, prior_energy_values = ps.readfile_latticemap(file_prior_energy)
prior_map = ps.LatticeMap(nodes, np.exp(-prior_energy_values))

# Prior Conditioning
file_prior_cond = ps.PATH_EXAMPLES + 'example_condition_2D.txt'
conditions = [ps.ConditionFile(file_prior_cond)]

# Sampler Definitions and Specifications
sampler_settings = {
    'traj_type': 'GantryDominant2D',
    'ratio': 2.0,
}
sampler = ps.SamplerFactory.make(prior_map, **sampler_settings)

# Sample Trajectory
trajectory = sampler(conditions)

# Plot the Result
lattice = prior_map.lattice
permission_map = conditions[0].permission_map(lattice)
values = np.reshape((permission_map * prior_map).map_vals, lattice.nnodes_dim, order='F')

# _, ax = plt.subplots()
# min_x, max_x = nodes[0][0], nodes[0][-1]
# min_y, max_y = nodes[1][0], nodes[1][-1]
# ax.imshow(values.transpose(),
#           origin='lower',
#           cmap='viridis',
#           extent=[min_x, max_x, min_y, max_y])
# ax.set(xticks=np.arange(min_x, max_x+1, (max_x - min_x)//2),
#        yticks=np.arange(min_y, max_y+1, (max_y - min_y)//2))
# ax.plot(*np.array(trajectory.points).transpose(), 'r')
#
# plt.show()
