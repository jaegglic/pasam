#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXAMPLE FOR SAMPLING A TRAJECTORY

This is a 2D example for a 180 x 90 grid representing the following setup::

           +89° --------------------------------------
                |            ****        0000000000  |
                | ***      **    *       0000000000  |
                **   ******       *        000000    |
    DIM_GANTRY  |                  *         00      |
                |      00           *                |
                |    000000          *****        **** Trajectory
                |  0000000000             ***  ***   |
                |  0000000000                **      |
           -89° --------------------------------------
             -179°            DIM_GANTRY            +179°

where the nodes spacing is 2° in each direction. The map values represent the
prior conditioning with permitted (`1`) and blocked (`0`) nodes. These consider
mechanical restrictions taken from `restrictions_180x90.txt` while the prior
density is loaded from the file `prior_energy_180x90.txt`.
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
import os
# Third party requirements
import numpy as np
import matplotlib.pylab as plt
# Local imports
import pasam as ps

# Without the inspection/adaption of the prior conditioning (c.f.
# `sampler.set_prior_cond(conditions, inspect=False)`) the mechanical map does
# interfere with the trajectory permission (c.f. `ratio`) because it might
# happen that some trajectory points are so close to the blocked region such
# that it is no longer possible to avoid the forbidden zone. Some 'problematic'
# seeds are::
#
#   - trajectory = sampler(..., seed=155819)
#   - trajectory = sampler(..., seed=98407703)
#   - trajectory = sampler(..., seed=138667415).
#
# However, setting `inspect=True` avoids these kind of problems. See also the
# example in `plot_prior_cond.py`.

# Constants
_LOC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Computational lattice
nodes = [np.arange(-179, 180, 2), np.arange(-89, 90, 2)]
lattice = ps.Lattice(nodes)

# Sampler
ratio = 2.0
sampler = ps.GantryDominant2D(lattice=lattice, ratio=ratio, order='max_random')

# Set prior probability
# TODO remove comment here
# file_energy = os.path.join(_LOC_DIR, 'data', 'prior_energy_180x90.txt')
file_energy = os.path.join(_LOC_DIR, 'data', 'prior_energy_sin_180x90.txt')
prior_energy = ps.readfile_latticemap(file_energy)
sampler.set_prior_prob(prior_energy, energy=True)

# Set and check validity of prior conditioning
file_cond = os.path.join(_LOC_DIR, 'data', 'restrictions_180x90.txt')
conditions = [file_cond]
sampler.set_prior_cond(conditions, validate=True)

# Sample Trajectory
# TODO remove seed
# TODO remove control figures in folder examples\figures
trajectory = sampler(seed=154586)

# Plot the Result
map_ = sampler.prior_cond * sampler.prior_prob
values = np.reshape(map_.values, lattice.nnodes_dim, order='F')

_, ax = plt.subplots()
min_x, max_x = nodes[0][0], nodes[0][-1]
min_y, max_y = nodes[1][0], nodes[1][-1]
ax.imshow(values.transpose(),
          origin='lower',
          cmap='viridis',
          extent=[min_x, max_x, min_y, max_y])
ax.set(xticks=np.arange(min_x, max_x+1, (max_x - min_x)//2),
       yticks=np.arange(min_y, max_y+1, (max_y - min_y)//2))
ax.plot(*np.array(trajectory.points).transpose(), 'r')

# Plot settings
font_size = 12
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)

# Save the results
file_fig = os.path.join(_LOC_DIR, 'figures', 'trajectory.png')
plt.savefig(file_fig)
print(f'\nThe result is saved under {file_fig}')
