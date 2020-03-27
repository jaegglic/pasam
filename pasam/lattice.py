# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements and the associated objects.

Classes
-------
    - :class:`Condition`: (abstract) Parent class for each condition.
    - :class:`ConditionFile`: Condition from a file.
    - :class:`ConditionPoint`: Condition from a condition point.

    - :class:`Lattice`: Lattice nodes in 1-, 2-, or 3-dimensions
    - :class:`LatticeMap`: Value map associated to a lattice

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
import numbers
# Third party requirements
import numpy as np
# Local imports
import pasam._messages as msg
from pasam._settings import NP_ORDER, RLIB_MAXLIST, DIM_GANTRY, DIM_TABLE
import pasam.utils as utl

# Constants and Variables
_rlib = reprlib.Repr()
_rlib.maxlist = RLIB_MAXLIST


# Condition Objects
class Condition(abc.ABC):
    """`Condition` defines an abstract parent class for restrictions in the
    path sampling.

    Notes:
        Any sub-class of `Condition` must provide an implementation of

            - :meth:`permission_map`
    """

    @abc.abstractmethod
    def permission_map(self, lattice, traj_perm):
        """Produces a permission map with permitted (``True``) and blocked
        (``False``) lattice nodes.

        Args:
            lattice (Lattice): Object defining the computational lattice.
            traj_perm (TrajectoryPermission): Defines a trajectory permission
               object for generating the permission map.

        Returns:
            LatticeMap: Lattice map issued from the condition.
        """


class ConditionFile(Condition):
    """`ConditionFile` defines a condition from a .txt file.

    Args:
        file (str or pathlib.Path): File or filename.

    Attributes:
        file (str or pathlib.Path): File or filename.
    """

    def __init__(self, file):
        self.file = file

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(file={self.file})'

    def __str__(self):
        return self.__repr__()

    # Definition of the abstract method in `Condition`
    def permission_map(self, lattice, traj_perm=None):
        _, _, vals = utl.readfile_latticemap(self.file)
        values = utl.ams_val_map_to_bool_map(vals)  # !! Values are inverted !!

        latticemap = LatticeMap(lattice, values)
        # TODO this is very dirty giving a default value for checking or not
        #  the suitability of the map is veeeeery bad
        if traj_perm:
            latticemap = traj_perm.permission_from_map(latticemap)
        return latticemap


class ConditionPoint(Condition):
    """`ConditionPoint` defines a condition point in a lattice grid.

    Args:
        components (array_like, shape=(n,)): Coordinate components.

    Attributes:
        components (ndarray, shape=(n,)): Coordinate components.
    """

    def __eq__(self, other):
        if isinstance(other, ConditionPoint):
            return np.all(self.components == other.components)
        elif isinstance(other, tuple) or isinstance(other, list):
            return np.all(self.components == np.asarray(other))
        return False

    def __init__(self, components):
        self.components = np.asarray(components)

    def __len__(self):
        return len(self.components)

    def __repr__(self):
        cls_name = type(self).__name__
        components_repr = _rlib.repr(self.components)
        return f'{cls_name}(components={components_repr})'

    def __str__(self):
        return self.__repr__()

    # Definition of the abstract method in `Condition`
    def permission_map(self, lattice, traj_perm):
        return traj_perm.permission_from_point(lattice, self.components)


# Lattice and LatticeMap Objects
class Lattice:
    """`Lattice` defines the computational lattice.

    A lattice is defined by a one-, two-, or three-dimensional regular grid of
    nodes. The nodes in one dimension is a set of strictly increasing values.

    Args:
        nodes (list of array_like): Tensor product nodes.

    Attributes:
        nodes (list of ndarray): Tensor product nodes.

    """

    def __eq__(self, other):
        if isinstance(other, Lattice) and self.nnodes_dim == other.nnodes_dim:
            nodes_eq = [np.all(n_sel == n_oth)
                        for n_sel, n_oth in zip(self.nodes, other.nodes)]
            return all(nodes_eq)
        return False

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != len(self.nodes):
            raise ValueError(msg.err1004(self, key))
        nodes = [n[k] for n, k in zip(self.nodes, key)]
        for inode, node in list(enumerate(nodes))[::-1]:
            if isinstance(node, numbers.Number):
                del nodes[inode]
        return self.__class__(nodes)

    def __init__(self, nodes):
        nodes = list(np.asarray(n) for n in nodes)
        if not all(utl.isincreasing(n, strict=True) for n in nodes):
            raise ValueError(msg.err1001(nodes))
        self.nodes = nodes

    def __repr__(self):
        cls_name = type(self).__name__
        nodes_repr = _rlib.repr(self.nodes)
        return f'{cls_name}(nodes={nodes_repr})'

    def __str__(self):
        return self.__repr__()

    def indices_from_point(self, components):
        """Returns the indices for the nearest node in each direction.

        Args:
            components (array_like, shape=(n,)): Components of point.

        Returns:
            tuple: Indices of closest node.
        """
        if len(components) != self.ndim:
            raise ValueError(msg.err1005(self.ndim, len(components)))

        ind = [np.argmin(np.abs(nodes - comp))
               for nodes, comp in zip(self.nodes, components)]
        return tuple(ind)

    @property
    def ndim(self):
        """The dimensionality of the lattice.

        Returns:
            int
        """
        return len(self.nodes)

    @property
    def nnodes_dim(self):
        """The total number of nodes in the lattice.

        Returns:
            tuple
        """
        return tuple(len(n) for n in self.nodes)

    @property
    def nnodes(self):
        """The total number of nodes in the lattice.

        Returns:
            int
        """
        return int(np.prod(self.nnodes_dim))


class LatticeMap:
    """`LatticeMap` defines a value map associated to a :class:`Lattice`.

    Args:
        lattice (Lattice or list of array_like): Object defining the
            computational lattice.
        map_vals (array_like, shape=(n,)): Map values associated to the lattice
            nodes.
        dtype (data-type, optional): The desired data-type for the map_values.
            If not given, then the type will be determined as the minimum
            requirement by `numpy`.

    Attributes:
        lattice (Lattice): Object defining the computational lattice.
        map_vals (ndarray, shape=(n,)): Map values associated to the lattice
            nodes.
    """

    # TODO refactor map_vals into values and keep all names as e.g. prior_map
    # then, one writes prior_map.values (not prior_map.map_values)
    def __add__(self, other):
        """Supports `LatticeMap` + `LatticeMap` as well as `LatticeMap` +
        `number.Number`"""
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.map_vals + other.map_vals)
            else:
                raise ValueError(msg.err1002('+'))

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.map_vals + other)

        return NotImplemented

    def __eq__(self, other):
        """Equality is based on `lattice` and `map_vals`."""
        if isinstance(other, LatticeMap) and self.lattice == other.lattice:
            return np.all(self.map_vals == other.map_vals)
        return False

    def __getitem__(self, key):
        """Uses the slicing of numpy (with according reshapes)"""
        lattice = self.lattice[key]
        nnodes_dim = self.lattice.nnodes_dim
        map_vals = self.map_vals.reshape(nnodes_dim, order=NP_ORDER)
        map_vals = map_vals[key].ravel(order=NP_ORDER)
        return self.__class__(lattice=lattice, map_vals=map_vals)

    def __init__(self, lattice, map_vals, dtype=None):
        if not isinstance(lattice, Lattice):
            lattice = Lattice(lattice)
        # TODO Refactor 'map_vals' to 'values'
        map_vals = np.asarray(map_vals, dtype=dtype).ravel(order=NP_ORDER)
        if lattice.nnodes != len(map_vals):
            raise ValueError(msg.err1003(lattice.nnodes, len(map_vals)))
        self.lattice = lattice
        self.map_vals = map_vals

    def __mul__(self, other):
        """Supports `LatticeMap` * `LatticeMap` as well as `LatticeMap` *
        `numbers.Number`.
        """
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.map_vals * other.map_vals)
            else:
                raise ValueError(msg.err1002('*'))

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.map_vals * other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        cls_name = type(self).__name__
        map_vals_repr = _rlib.repr(self.map_vals)
        # TODO Refactor 'map_vals' to 'values' (change it in __repr__)
        return f'{cls_name}(lattice={repr(self.lattice)}, ' \
               f'map_vals={map_vals_repr})'

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return self.__repr__()

    def indices_from_point(self, components):
        """Returns the indices for the nearest node in the lattice.

        Args:
            components (array_like, shape=(n,)): Components of point.

        Returns:
            tuple: Indices of closest node.
        """
        return self.lattice.indices_from_point(components)

    def slice(self, dim: int, ind: int):
        """Returns a slice where position `pos` is fixed in dimension `dim`.

        Args:
            dim (int): Static dimension for the slice.
            ind (int): Index of slicing position.

        Returns:
            LatticeMap: Slice of `self`.
        """
        slice_ = [slice(None) if i != dim else ind for i in range(self.ndim)]
        return self.__getitem__(tuple(slice_))

    @property
    def ndim(self):
        """The dimensionality of the lattice map.

        Returns:
            int
        """
        return self.lattice.ndim

    @classmethod
    def from_txt(cls, file, lattice=None):
        """Produces lattice map objects from .txt files.

        The structure of a latticemap text file is reported in the function
        :func:`pasam.utils.readfile_ams_latticemap`.

        Args:
            file (str or pathlib.Path): File or filename.
            lattice (Lattice, optional): Object defining the computational
                lattice.

        Returns:
            LatticeMap: Object defining a lattice map
        """
        _, nodes, map_vals = utl.readfile_latticemap(file)
        lattice_from_file = Lattice(nodes)

        if not lattice:
            lattice = lattice_from_file
        elif lattice != lattice_from_file:
            raise ValueError(msg.err1000(file, lattice))

        return cls(lattice, map_vals)


# Trajectory Related Objects
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
        Any sub-class of `TrajectoryPermission` must provide an implementation
        of

            - :meth:`permission_map`
            - :meth:`adjacency_graph`
    """

    @abc.abstractmethod
    def permission_from_map(self, restriction):
        """Extends a restriction map in order to respects trajectory
        permissions.

        Notes
            It is not uncommon that the output map and the input map are quite
            similar because this function tries to find the minimum of
            additional nodes that need to be blocked in order to respect the
            trajectory movement permission.

        Args:
            restriction (LatticeMap): Restriction map

        Returns:
            LatticeMap: Boolean map for permitted (True) and blocked (False)
                nodes.
        """

    @abc.abstractmethod
    def permission_from_point(self, lattice, components):
        """Generates a permission map based on a computational lattice and a
        condition point.

        Args:
            lattice (Lattice): Object defining the computational lattice.
            components (array_like, shape=(n,)): Coordinate components.

        Returns:
            LatticeMap: Boolean map for permitted (True) and blocked (False)
                nodes.
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
        # Gantry/Table nodes
        nodes_gantry = lattice.nodes[DIM_GANTRY]
        nodes_table = lattice.nodes[DIM_TABLE]

        # Get an array containing the linear indices
        lin_ind_arr = self._lin_ind_arr(lattice.nnodes_dim)
        shape, dim = lin_ind_arr.shape, (len(nodes_gantry), len(nodes_table))
        if shape != dim:
            raise ValueError(msg.err2000(shape, dim))

        # Iteratively go through the nodes and generate adjacency graph
        alpha = 2 * np.arctan(self._ratio)  # The opening is symmetric
        pts = nodes_table
        graph = np.zeros((lattice.nnodes,)*2, dtype=bool)
        for igan, node_g in enumerate(nodes_gantry[:-1]):

            # Conical opening for the permitted area of the trajectory
            dist = abs(node_g - nodes_gantry[igan+1])
            for itab, node_t in enumerate(nodes_table):
                cntr = node_t
                ind = utl.conical_opening_indicator(cntr, dist, alpha, pts)
                graph[lin_ind_arr[igan, itab], lin_ind_arr[igan+1, ind]] = True

        return graph

    def permission_from_map(self, restriction):
        graph = self.adjacency_graph(restriction.lattice)
        values_min = restriction.map_vals

        values_ext = self._perm_from_map(graph, values_min)
        return LatticeMap(restriction.lattice, values_ext)

    def permission_from_point(self, lattice, components):
        nnodes_dim = lattice.nnodes_dim
        if components is None:
            return np.ones(nnodes_dim, dtype=bool)
        else:
            values = self._perm_from_pt(lattice, components)
            return LatticeMap(lattice, values)

    def _lin_ind_arr(self, nnodes_dim):
        """Returns the two dimensional array of the linear indices where the
        lower dimension always considers the GANTRY dimension."""
        nnodes = int(np.prod(nnodes_dim))
        lin_ind_arr = np.arange(nnodes).reshape(nnodes_dim, order=NP_ORDER)

        if DIM_GANTRY > DIM_TABLE:
            lin_ind_arr = lin_ind_arr.transpose()

        return lin_ind_arr

    def _perm_from_map(self, graph, values):
        """Returns a permission array respecting graph connectivity and
        restrictions from a set of values."""
        values = np.array(values, dtype=bool, copy=True).ravel(order=NP_ORDER)
        graph = np.array(graph, copy=True)
        nnodes = len(values)

        # Initialize set of nodes to be checked
        inodes_control = set(np.where(np.logical_not(values))[0])

        # Boundary nodes
        no_outgoing = [i for i in range(nnodes) if not any(graph[i, :])]
        no_incoming = [i for i in range(nnodes) if not any(graph[:, i])]

        # Loop until there is no potential issue anymore
        while len(inodes_control) > 0:
            inode = inodes_control.pop()

            outgoing = np.where(graph[inode, :])[0]
            incoming = np.where(graph[:, inode])[0]

            # Remove node if there is no path passing through it
            outgoing_issue = len(outgoing) == 0 and inode not in no_outgoing
            incoming_issue = len(incoming) == 0 and inode not in no_incoming
            if not values[inode] or outgoing_issue or incoming_issue:
                # Remove node edges from graph
                graph[inode, :] = False
                graph[:, inode] = False

                # Add the modified nodes to the list to be checked
                inodes_control = inodes_control.union(outgoing)
                inodes_control = inodes_control.union(incoming)

                # Make it unpermitted
                values[inode] = False

        return values

    def _perm_from_pt(self, lattice, components):
        """Returns a permission array respecting a point conditioning.
        """
        # Initialization
        nodes = lattice.nodes
        map_vals = np.zeros(lattice.nnodes_dim, dtype=bool)

        # Conical opening for the permitted area of the trajectory
        cntr = components[DIM_TABLE]
        alpha = 2 * np.arctan(self._ratio)  # The opening is symmetric
        pts = nodes[DIM_TABLE]

        # Loop in gantry direction through the lattice
        for inode, node in enumerate(nodes[DIM_GANTRY]):
            dist = abs(node - components[DIM_GANTRY])
            ind = utl.conical_opening_indicator(cntr, dist, alpha, pts)
            map_vals[inode, ind] = True

        # Correct dimension ordering if needed
        if DIM_GANTRY > DIM_TABLE:
            map_vals = map_vals.transpose()

        return map_vals.ravel(order=NP_ORDER)
