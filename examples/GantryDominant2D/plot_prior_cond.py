#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXAMPLE FOR VISUALIZING THE CONDITIONING REFINEMENT

This module visualized the refinement of the prior conditioning for different
sampler settings.

We use two-dimensional 50 x 25 grid representing the following setup::

           +25° --------------------------------------
                |                             00     |
                |             00              00     |
                |             00                     |
    DIM_GANTRY  |            0000                    |
                |    **      0000                 000|
                |  ***  *  *X00000                0  |
                ***      ** 000000                   |
             0° --------------------------------------
                0°            DIM_GANTRY            50°

where the nodes spacing is 1° in each direction. The map values represent the
prior conditioning with permitted (`1`) and blocked (`0`) nodes. These consider
mechanical restrictions taken from `restrictions_180x90.txt`.

Depending on the sampling settings, the prior conditioning might need a
refinement. Otherwise it can happen that due to the restricted mobility (e.g.
the movement ratio in `ratio`) the trajectory can be stuck close to a
conditioning area.
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

# Constants
_LOC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Computational lattice
nodes = [np.arange(0, 50), np.arange(0, 25)]
lattice = ps.Lattice(nodes)

# Samplers
sampler_ratio_10 = ps.GantryDominant2D(lattice=lattice, ratio=1.0)
sampler_ratio_20 = ps.GantryDominant2D(lattice=lattice, ratio=2.0)

# Compute conditioning maps with and without refinement
file_cond = os.path.join(_LOC_DIR, 'data', 'restrictions_50x25.txt')
conditions = [file_cond]
cond_map_raw = sampler_ratio_10.compute_condition_map(conditions, validate=False)
cond_map_ref_10 = sampler_ratio_10.compute_condition_map(conditions, validate=True)
cond_map_ref_20 = sampler_ratio_20.compute_condition_map(conditions, validate=True)

# Plot the Result
nnodes_dim = lattice.nnodes_dim
values_raw_10 = np.reshape(cond_map_raw.values, nnodes_dim, order='F')
values_ref_10 = np.reshape(cond_map_ref_10.values, nnodes_dim, order='F')
values_ref_20 = np.reshape(cond_map_ref_20.values, nnodes_dim, order='F')

_, ax = plt.subplots(3, 1, figsize=(5, 8))
min_x, max_x = nodes[0][0], nodes[0][-1]
min_y, max_y = nodes[1][0], nodes[1][-1]
im_args = {
    'origin':   'lower',
    'cmap':     'viridis',
    'extent':   [min_x, max_x, min_y, max_y],
}
ax[0].imshow(values_raw_10.transpose(), **im_args)
ax[1].imshow((0.5*values_ref_10 + 0.5*values_raw_10).transpose(), **im_args)
ax[2].imshow((0.5*values_ref_20 + 0.5*values_raw_10).transpose(), **im_args)

# Plot settings
font_size = 12
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('axes', titlesize=font_size)

ax[0].set(xticks=[], yticks=[min_y, max_y])
ax[1].set(xticks=[], yticks=[min_y, max_y])
ax[2].set(xticks=[min_x, max_x], yticks=[min_y, max_y])

ax[0].set_title('No refinement')
ax[1].set_title('ratio = 1.0')
ax[2].set_title('ratio = 2.0')

# Save the results
file_fig = os.path.join(_LOC_DIR, 'figures', 'prior_cond.png')
plt.savefig(file_fig)
print(f'\nThe result is saved under {file_fig}')
