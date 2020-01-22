# -*- coding: utf-8 -*-
"""Definitions of the trajectory sampling algorithms.

Classes
-------
    - :class:`Sampler`: (abstract) Parent class for each sampling algorithm.
    - :class:`SamplerFactory`: Factory for the different samplers.
    - :class:`SamplerGantryDominant2D`: Gantry dominant 2D trajectory sampler.

Methods
-------

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
import abc
import reprlib
import warnings
# Third party requirements
import numpy as np
# Local imports
import pasam._settings as settings
import pasam._messages as msg
from pasam.lattice import ConditionPoint
from pasam.pathgen import Trajectory

# Constants and Variables
_NP_ORDER = settings.NP_ORDER
_NP_SEED = 4541285
_DIM_GANTRY = settings.DIM_GANTRY

_rlib = reprlib.Repr()
_rlib.maxlist = settings.RLIB_MAXLIST
np.random.seed(settings.NP_SEED)


class Sampler(abc.ABC):
    """Parent class for all sampling algorithms.

    Args:
        prior_map (LatticeMap): Prior sampling map for trajectory points.

    """

    def __init__(self, prior_map):
        self._prior_map = prior_map

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(prior_map={self._prior_map})'

    def __str__(self):
        return self.__repr__()

    @abc.abstractmethod
    def __call__(self, conditions):
        """Start the sampling algorithm.

        Args:
            conditions (list of Condition): Set of conditions.

        Returns:
            Trajectory: Sampled trajectory
        """


class SamplerFactory:
    """Factory for creating the sampling algorithm objects.
    """

    @staticmethod
    def make(prior_map, traj_type):
        """Creates :class:`Sampler` objects.

        Args:
            prior_map (LatticeMap): Computational lattice and it's prior map.
            traj_type (str): Trajectory type.

        Returns:
            Sampler: Sampling algorithm object.
        """

        if traj_type == 'GantryDominant2D':
            return SamplerGantryDominant2D(prior_map)
        else:
            raise ValueError(msg.err3000(traj_type))


class SamplerGantryDominant2D(Sampler):
    """`SamplerGantryDominant2D` is the usual 2D gantry dominated trajectory
    movement sampler class.

    Attributes:
        ntrajectory (int): Length of trajectory.
    """

    def __init__(self, prior_map):
        super().__init__(prior_map)
        self.ntrajectory = self._prior_map.lattice.nnodes_dim[_DIM_GANTRY]

    def __call__(self, conditions):
        """Executes the sampling algorithm.

        Args:
            conditions (list of Condition): Set of conditions.

        Returns:
            Trajectory: Sampled trajectory
        """
        trajectory = np.array([None for _ in range(self.ntrajectory)])
        ind_to_do = np.array([True for _ in range(self.ntrajectory)])

        self._fix_traj_points_cond(trajectory, ind_to_do, conditions)
        self._fix_traj_points_smpl(trajectory, ind_to_do)

        return Trajectory(trajectory)

    def _fix_traj_points_cond(self, trajectory, ind_to_do, conditions):
        lattice = self._prior_map.lattice
        for cond in conditions:
            if isinstance(cond, ConditionPoint):
                pos = lattice.indices_from_point(cond.components)[_DIM_GANTRY]
                ind_to_do[pos] = False
                trajectory[pos] = cond.components

    def _fix_traj_points_smpl(self, trajectory, ind_to_do):
        perm_map = self._prior_map
        ndim = perm_map.ndim
        nnodes_dim = perm_map.lattice.nnodes_dim
        nnodes_dim_red = tuple([n for dim, n in enumerate(nnodes_dim) if dim != _DIM_GANTRY])
        positions = np.arange(len(ind_to_do))
        while np.any(ind_to_do):
            pos = np.random.choice(positions[ind_to_do])
            ind_to_do[pos] = False

            pos_slice = [slice(None) for _ in range(ndim)]
            pos_slice[_DIM_GANTRY] = pos
            map_slice = perm_map[tuple(pos_slice)]


            distribution = map_slice.map_vals
            try:
                distribution = distribution / np.sum(distribution)
            except (ZeroDivisionError, FloatingPointError,
                    RuntimeWarning, RuntimeError):
                distribution = np.ones_like(distribution) / len(distribution)
                warnings.warn(msg.warn3000)

            # use a command like: (maybe put the slice outside of the while)
            # use nnodes_dim_red of above
            # slice_ind = [(i, j) for j in range(len(nodes_colli)) for i in range(len(nodes))]
            # np.random.choice(slice_ind, p=distribution)


            trajectory[pos] = 1