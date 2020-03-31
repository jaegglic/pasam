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


# Computational lattice
nodes = [np.arange(-179, 180, 2), np.arange(-89, 90, 2)]
lattice = ps.Lattice(nodes)

# Sampler
ratio = 2.0
sampler = ps.GantryDominant2D(lattice=lattice, ratio=ratio)

# Set prior probability
file_energy = ps.PATH_EXAMPLES + 'example_prior_energy_2D.txt'
prior_energy = ps.readfile_latticemap(file_energy)
sampler.set_prior_prob(prior_energy, energy=True)

# Set and check validity of prior conditioning
file_cond = ps.PATH_EXAMPLES + 'example_condition_2D.txt'
sampler.set_prior_cond([file_cond])


# # Sample Trajectory
# trajectory = sampler(inspect=False)
#
# # Plot the Result
# map_ = sampler._prior_cond * sampler._prior_prob
# values = np.reshape(map_.map_vals, lattice.nnodes_dim, order='F')
#
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
