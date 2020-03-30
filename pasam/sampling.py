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
import functools
# Third party requirements
import numpy as np
# Local imports
from pasam._settings import DIM_GANTRY, RLIB_MAXLIST, NP_SEED
import pasam._messages as msg
import pasam.utils as utl
from pasam.lattice import LatticeMap, ConditionPoint,\
    Trajectory, TrajectoryPermissionFactory

# Constants and Variables
_rlib = reprlib.Repr()
_rlib.maxlist = RLIB_MAXLIST
np.random.seed(NP_SEED)

# Problematic Seeds
# TODO fix this shit here
np.random.seed(155819)
# np.random.seed(98407703)
# np.random.seed(138667415)

# Test Random Seeds
# SEED = np.random.randint(158521456)
# print('Seed: ', SEED)
# np.random.seed(SEED)


class Sampler(abc.ABC):
    """Parent class for all sampling algorithms.

    Args:
        prior_map (LatticeMap): Prior sampling map.

    """

    def __init__(self, prior_map):
        self._prior_map = prior_map

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(prior_map={self._prior_map})'

    def __str__(self):
        return self.__repr__()

    @abc.abstractmethod
    def __call__(self, conditions=None):
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
    # TODO switch prior_map and traj_type
    # TODO refactor traj_type into type_
    def make(prior_map, traj_type, **kwargs):
        """Creates :class:`Sampler` objects.

        Args:
            traj_type (str): Trajectory type.
            prior_map (LatticeMap): Prior sampling map.
            kwargs (dict): Type specific arguments.

        Returns:
            Sampler: Sampling algorithm object.
        """

        if traj_type == 'GantryDominant2D':
            return SamplerGantryDominant2D(prior_map, traj_type, **kwargs)
        else:
            raise ValueError(msg.err3000(traj_type))


class SamplerGantryDominant2D(Sampler):
    """`SamplerGantryDominant2D` is the usual 2D gantry dominated trajectory
    movement sampler class.

    Attributes:
        ntrajectory (int): Length of trajectory.
    """
    # TODO refactor _TRAJ_TYPE into _TYPE_
    _TRAJ_TYPE = 'GantryDominant2D'

    # TODO refactor all of this class

    def __call__(self, conditions=None):
        """Executes the sampling algorithm.

        Args:
            conditions (list of Condition, optional): Set of conditions.

        Returns:
            Trajectory: Sampled trajectory
        """
        # Check conditions
        perm_map = self._permission_map_from_conditons(conditions)
        # perm_map = self._condition_set_validity(perm_map)

        # Sample path
        trajectory_points = self._sample_trajectory_points(perm_map)
        return Trajectory(trajectory_points)

    def __init__(self, prior_map, traj_type, ratio=1):
        super().__init__(prior_map)
        # TODO refactor traj_type into type_
        self._traj_type = traj_type
        self._ratio = ratio
        kwargs = {'traj_type': self._traj_type, 'ratio': self._ratio}
        self._traj_perm = TrajectoryPermissionFactory().make(**kwargs)

    def _sample_trajectory_points(self, perm_map):

        lattice = self._prior_map.lattice
        ndim = lattice.ndim
        ntraj = lattice.nnodes_dim[DIM_GANTRY]
        traj_points = np.array([None for _ in range(ntraj)])
        ind_to_do = np.array([True for _ in range(ntraj)], dtype='bool')

        gantry_ind = np.arange(len(ind_to_do))
        rem_nodes = [n for i, n in enumerate(lattice.nodes)
                     if i != DIM_GANTRY]
        rem_nodes = utl.cartesian_product(*rem_nodes)
        while np.any(ind_to_do):
            ind_gantry_pos = np.random.choice(gantry_ind[ind_to_do])

            prior_slice = self._prior_map.slice(DIM_GANTRY, ind_gantry_pos)
            perm_slice = perm_map.slice(DIM_GANTRY, ind_gantry_pos)

            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape((self._prior_map * perm_map).map_vals, (180, 90),
            #                       order='F').transpose(), origin='lower')
            # plt.plot([ind_gantry_pos,]*2, [0, 89], 'r--')
            # plt.show()

            if np.sum(perm_slice.map_vals) >= 1:
                distribution = (prior_slice * perm_slice).map_vals
                try:
                    distribution = distribution / np.sum(distribution)
                except (ZeroDivisionError, FloatingPointError,
                        RuntimeWarning, RuntimeError):
                    raise ValueError('HERE WE SHOULD TAKE THE PERMITTED REGION WITHOUT'
                                     'THE PRIOR DISTRIBUTION AND SAMPLE UNIFORMLY IN THERE')
                distribution = distribution / np.sum(distribution)
            else:
                import matplotlib.pyplot as plt
                plt.imshow(np.reshape((self._prior_map * perm_map).map_vals, (180, 90),
                                      order='F').transpose(), origin='lower')
                plt.show()
                raise ValueError('TODO Error message for not possible settings '
                                 '(because there is no connected trajectory possible anymore)')

            ind = np.random.choice(np.arange(len(distribution)),
                                   p=distribution)
            pos_slice = rem_nodes[ind]

            traj_point = np.array([None,] * ndim)
            traj_point[DIM_GANTRY] = lattice.nodes[DIM_GANTRY][ind_gantry_pos]
            traj_point[[i for i in range(ndim) if i != DIM_GANTRY]] = pos_slice

            traj_point = tuple(traj_point)
            cond_point = ConditionPoint(traj_point)
            perm_map *= cond_point.permission_map(lattice, self._traj_perm)

            traj_points[ind_gantry_pos] = traj_point
            ind_to_do[ind_gantry_pos] = False
        return traj_points

    def _condition_set_validity(self, perm_map):
        # TODO check condition set -> we can use a technique similar as the one
        #  for checking the condition map in
        #  TrajectoryPermissionGantryDominant2D._perm_from_map but this time we
        #  retain the graph and then check if there a combination of
        #  ingoing/outgoing from left to right. BUT MAKE SURE TO REUSE CODE
        graph = self._condition_set_graph(perm_map)
        pass
        # return valid_perm_map

    def _condition_set_graph(self, perm_map):
        pass

    def _permission_map_from_conditons(self, conditions):
        lattice = self._prior_map.lattice
        traj_perm = self._traj_perm
        if conditions:
            maps = [c.permission_map(lattice, traj_perm) for c in conditions]
            permission_map = functools.reduce(lambda a, b: a * b, maps)
        else:
            values = np.ones(lattice.nnodes, dtype='int')
            permission_map = LatticeMap(lattice, values)
        return permission_map

    @property
    def _kwargs(self):
        return {'traj_type': self._traj_type, 'ratio': self._ratio}
