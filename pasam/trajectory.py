# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements and the associated objects.

Classes
-------
    - :class:`Trajectory`: Definition of dynamic trajectories.
    - :class:`TrajectoryPermissionFactory`: Factory for defining trajectory
        permission
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
# Third party requirements
import numpy as np
# Local imports
import pasam._messages as msg
import pasam._settings as settings
import pasam.utils as utl

# Constants and Variables
_NP_ORDER = settings.NP_ORDER
_rlib = reprlib.Repr()
_rlib.maxlist = settings.RLIB_MAXLIST


class Trajectory:
    """Definition of dynamic trajectories.

    Args:
        points (array_like): Sequence of trajectory points.

    Attributes:
        points (list): Sequence of trajectory points.

    """

    def __init__(self, points):
        self.points = list(points)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        cls_name = type(self).__name__
        points_repr = _rlib.repr(self.points)
        return f'{cls_name}(points={points_repr})'

    def __str__(self):
        return self.__repr__()

    def to_txt(self, file):
        utl.write_trajectory_to_txt(file, self.points)


# TODO Give in the sampling algorithm one such an object and use it everywhere
class TrajectoryPermission(abc.ABC):
    """`TrajectoryPermission` defines an abstract parent class for the machine
    movement permissions.

    Notes:
        Any sub-class of `_PointTrajectory` must provide an implementation of

            - :meth:`permission_map`
    """
    # TODO refactor cond_point into components wherever it makes sense
    @abc.abstractmethod
    def permission_array(self, lattice, cond_point):
        """Generates a permission array based on a computational lattice and a
        condition point.

        Args:
            lattice (Lattice): Object defining the computational lattice.
            cond_point (array_like, shape=(n,)): Coordinate components.

        Returns:
            ndarray: Boolean array for permitted (True) and blocked (False)
                nodes.
        """

    @abc.abstractmethod
    def adjacency_graph(self, lattice):
        """Generates an adjacency graph based on a computational lattice.

        The entry (i, j) of the graph matrix is non-zero if there is a
        connection starting from node i and joining j. The linear indices i, j
        are ordered according to the order specified in `_settings`.

        Args:
            lattice (Lattice): Object defining the computational lattice.

        Returns:
            ndarray, shape=(lattice.nnodes,)*2: Boolean array representing
                adjacency graph.
        """


class TrajectoryPermissionFactory:
    """`TrajectoryPermissionFactory` produces instances of
    :class:`_PointTrajectory`.
    """

    @staticmethod
    def make(traj_type, **kwargs):
        """Creates `TrajectoryPermission` objects.

        Args:
            traj_type (str): Trajectory type.
            kwargs (dict): Type specific arguments.

        Returns:
            TrajectoryPermission: Trajectory permission object.
        """
        if traj_type == 'GantryDominant2D':
            return TrajectoryPermissionGantryDominant2D(**kwargs)
        else:
            raise ValueError(msg.err0000(traj_type))


class TrajectoryPermissionGantryDominant2D(TrajectoryPermission):
    """`TrajectoryPermissionGantryDominant2D` is the usual 2D gantry dominated
    movement restriction class.

    Args:
        ratio (float): Maximum allowed ratio between table and gantry angle
            rotation.
    """

    def __init__(self, ratio):
        self._ratio = ratio

    # TODO refactor cond_point into components wherever it makes sense
    def permission_array(self, lattice, cond_point):
        nnodes_dim = lattice.nnodes_dim
        if cond_point is None:
            return np.ones(nnodes_dim, dtype=bool)
        return self._permission_array_condition_point(lattice, cond_point)

    # TODO refactor cond_point into components wherever it makes sense
    def _permission_array_condition_point(self, lattice, cond_point):
        """Returns a two-dimensional permission map according to a conditioning
        point.
        """
        # Gantry/Table indices in self._nodes and self._condition_point
        dim_gantry  = settings.DIM_GANTRY
        dim_table   = settings.DIM_TABLE

        # Initialization
        nodes = lattice.nodes
        map_vals = np.zeros(lattice.nnodes_dim, dtype=bool)

        # Conical opening for the permitted area of the trajectory
        cntr = cond_point[dim_table]
        alpha = 2 * np.arctan(self._ratio)  # The opening is symmetric
        pts = nodes[dim_table]

        # Loop in gantry direction through the lattice
        for inode, node in enumerate(nodes[dim_gantry]):
            dist = abs(node - cond_point[dim_gantry])
            ind = utl.conical_opening_indicator(cntr, dist, alpha, pts)
            map_vals[inode, ind] = True

        # Correct dimension ordering if needed
        if dim_gantry > dim_table:
            map_vals = map_vals.transpose()

        return map_vals.ravel(order=_NP_ORDER)

    # TODO unit-test and implement `adjacency_graph` function
    def adjacency_graph(self, lattice):
        # Gantry/Table indices in self._nodes and self._condition_point
        nodes_gantry = lattice.nodes[settings.DIM_GANTRY]
        nodes_table = lattice.nodes[settings.DIM_TABLE]

        lin_ind_arr = self._lin_ind_arr(lattice.nnodes_dim)
        shape, dim = lin_ind_arr.shape, (len(nodes_gantry), len(nodes_table))
        if shape != dim:
            raise ValueError(msg.err2000(shape, dim))

        # Iteratively go through the nodes and generate adjacency graph
        alpha = 2 * np.arctan(self._ratio)  # The opening is symmetric
        pts = nodes_table
        graph = np.zeros((lattice.nnodes,) * 2, dtype=bool)
        for igan, node_g in enumerate(nodes_gantry[:-1]):
            # Conical opening for the permitted area of the trajectory
            dist = abs(node_g - nodes_gantry[igan+1])
            for itab, node_t in enumerate(nodes_table):
                cntr = node_t
                ind = utl.conical_opening_indicator(cntr, dist, alpha, pts)
                graph[lin_ind_arr[igan, itab], lin_ind_arr[igan+1, ind]] = True

        return graph

    def _lin_ind_arr(self, nnodes_dim):
        """Returns the two dimensional array of the linear indices where the
        lower dimension always considers the GANTRY dimension."""
        nnodes = int(np.prod(nnodes_dim))
        lin_ind_arr = np.arange(nnodes).reshape(nnodes_dim, order=_NP_ORDER)

        if settings.DIM_GANTRY > settings.DIM_TABLE:
            lin_ind_arr = lin_ind_arr.transpose()

        return lin_ind_arr
