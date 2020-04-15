#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXAMPLE FOR EXAMINING THE INFLUENCE OF THE PRIOR DISTRIBUTION

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
import functools
# Third party requirements
import numpy as np
import matplotlib.pylab as plt
# Local imports
import pasam as ps

# TODO sample a large number of trajectories and show how the prior is
#  represented

# TODO play with the `order` in sampler=GrantryDominan2D(..., order='random')
#  we expect that
#  - 'random' reproduces the prior prob (but is not very sensitive to it) !!!!
#  - 'max_val' samples very closely the points which looks strange
#  - 'max_random' Should give the best representation of the


# Constants
_LOC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSAMPLE = 10000

# Computational lattice
nodes = [np.arange(-179, 180, 2), np.arange(-89, 90, 2)]
lattice = ps.Lattice(nodes)

# Sampler
ratio = 2.0
order = 'random'
sampler = ps.GantryDominant2D(lattice=lattice, ratio=ratio, order=order)

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

# Sample Trajectories
# TODO remove seed
# TODO remove control figures in folder examples\figures
trajectories = []
for i, seed in enumerate(range(_NSAMPLE)):
    print(f'\rSample {i+1:4d} / {_NSAMPLE:4d}...', end='')
    traj = sampler(seed=seed, validate=False)
    trajectories.append(traj)
print('done')

# Plot the Result
_, ax = plt.subplots(3, 1, figsize=(4, 8))
min_x, max_x = nodes[0][0], nodes[0][-1]
min_y, max_y = nodes[1][0], nodes[1][-1]
im_args = {
    'origin':   'lower',
    'cmap':     'viridis',
    'extent':   [min_x, max_x, min_y, max_y],
}

# Plot prior map
values = np.reshape(prior_energy.values, lattice.nnodes_dim, order='F')
ax[0].imshow(values.transpose(), **im_args)
ax[0].set(xticks=np.arange(min_x, max_x+1, (max_x - min_x)//2),
          yticks=np.arange(min_y, max_y+1, (max_y - min_y)//2))
ax[0].set_title('Energy')

# Plot prior probability
map_ = (sampler.prior_cond * sampler.prior_prob).normalize_sum(axis=1)
values = np.reshape(map_.values, lattice.nnodes_dim, order='F')
ax[1].imshow(values.transpose(), **im_args)
ax[1].set(xticks=np.arange(min_x, max_x+1, (max_x - min_x)//2),
          yticks=np.arange(min_y, max_y+1, (max_y - min_y)//2))
ax[1].set_title('Prior')

# Plot empirical prior probability
map_ = trajectories[0].to_latticemap(lattice, dtype='int')
for traj in trajectories[1:]:
    map_ += traj.to_latticemap(lattice, dtype='int')
values = np.reshape(map_.values, lattice.nnodes_dim, order='F')
ax[2].imshow(values.transpose(), **im_args)
ax[2].set(xticks=np.arange(min_x, max_x+1, (max_x - min_x)//2),
          yticks=np.arange(min_y, max_y+1, (max_y - min_y)//2))
ax[2].set_title(f"Posterior (order='{order}')")

# Plot settings
font_size = 12
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)

# Save the results
file_fig = os.path.join(_LOC_DIR, 'figures', 'trajectory_prior_prob.png')
plt.savefig(file_fig)
print(f'\nThe result is saved under {file_fig}')
