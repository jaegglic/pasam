# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements and the associated objects.

Classes
-------
    - :class:`Condition`: (abstract) Parent class for each condition.
    - :class:`ConditionFile`: Condition from a file.
    - :class:`ConditionPoint`: Condition from a condition point.
    - :class:`Lattice`: Lattice nodes in 1-, 2-, or 3-dimensions
    - :class:`LatticeMap`: Value map associated to a lattice

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
import numbers
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


# Condition Objects
class Condition(abc.ABC):
    """`Condition` defines an abstract parent class for restrictions in the
    path sampling.

    Notes:
        Any sub-class of `Condition` must provide an implementation of

            - :meth:`permission_map`
    """

    @abc.abstractmethod
    def permission_map(self, lattice, **kwargs):
        """Produces a permission map with permitted (``True``) and blocked
        (``False``) lattice nodes.

        Args:
            lattice (Lattice): Object defining the computational lattice.
            kwargs (dict): Type specific arguments.

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
    def permission_map(self, lattice, **kwargs):
        map_vals = utl.permission_array_from_condition_file(self.file)
        return LatticeMap(lattice, map_vals)


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
    def permission_map(self, lattice, **kwargs):
        traj_perm = _TrajectoryPermissionFactory.make(**kwargs)
        map_vals = traj_perm.permission_array_from_cond_point(lattice, self.components)
        return LatticeMap(lattice, map_vals)


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
        for inode, node in enumerate(nodes):
            if isinstance(node, numbers.Number):
                nodes[inode] = [node]
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
        lattice (Lattice): Object defining the computational lattice.
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
        map_vals = self.map_vals.reshape(self.lattice.nnodes_dim, order=_NP_ORDER)
        map_vals = map_vals[key].ravel(order=_NP_ORDER)
        return self.__class__(lattice=lattice, map_vals=map_vals)

    def __init__(self, lattice, map_vals, dtype=None):
        map_vals = np.asarray(map_vals, dtype=dtype).ravel(order=_NP_ORDER)
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


# Private Classes
class _TrajectoryPermission(abc.ABC):
    """`_TrajectoryPermission` defines an abstract parent class for the machine
    movement permissions.

    Notes:
        Any sub-class of `_PointTrajectory` must provide an implementation of

            - :meth:`permission_map`
    """

    @abc.abstractmethod
    def permission_array_from_cond_point(self, lattice, cond_point):
        """Generates a permission array based on a computational lattice an a
        condition point.

        Args:
            lattice (Lattice, optional): Object defining the computational
                lattice.
            cond_point (array_like, shape=(n,)): Coordinate components.

        Returns:
            ndarray: Boolean array for permitted (True) and blocked (False)
                nodes.
        """


class _TrajectoryPermissionFactory:
    """`_TrajectoryPermissionFactory` produces instances of
    :class:`_PointTrajectory`.
    """

    @staticmethod
    def make(traj_type, **kwargs):
        """Creates `_TrajectoryPermission` objects.

        Args:
            traj_type (str): Trajectory type.
            kwargs (dict): Type specific arguments.

        Returns:
            _TrajectoryPermission: Trajectory permission object.
        """
        if traj_type == 'GantryDominant2D':
            return _TrajectoryPermissionGantryDominant2D(**kwargs)
        else:
            raise ValueError(msg.err0000(traj_type))


class _TrajectoryPermissionGantryDominant2D(_TrajectoryPermission):
    """`_TrajectoryPermissionGantryDominant2D` is the usual 2D gantry dominated
    movement restriction class.

    Args:
        ratio (float): Maximum allowed ratio between table and gantry angle
            rotation.
    """

    def __init__(self, ratio):
        self._ratio = ratio

    def permission_array_from_cond_point(self, lattice, cond_point):
        nnodes_dim = lattice.nnodes_dim
        if cond_point is None:
            return np.ones(nnodes_dim, dtype=bool)
        return self._permission_map_from_condition_point(lattice, cond_point)

    def _permission_map_from_condition_point(self, lattice, cond_point):
        """Returns a two-dimensional permission map according to a conditioning
        point.
        """
        # Gantry/Table indices in self._nodes and self._condition_point
        dim_gantry  = settings.DIM_GANTRY
        dim_table   = settings.DIM_TABLE

        # Initialization
        nodes = lattice.nodes
        map_vals = np.zeros(lattice.nnodes_dim, dtype=bool)

        # Loop in gantry direction through the lattice
        for inode, node in enumerate(nodes[dim_gantry]):
            v_range = abs(node - cond_point[dim_gantry]) * self._ratio
            v_min = cond_point[dim_table] - v_range
            v_max = cond_point[dim_table] + v_range

            ind_true = np.logical_and(nodes[dim_table] >= v_min,
                                      nodes[dim_table] <= v_max)
            map_vals[inode, ind_true] = True
        return map_vals.ravel(order=_NP_ORDER)


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [4, 5, 6, 7, 8, 9, 10.5]
    z = [-1, 0, 1]

    lattice_2D = Lattice([x, y])
    print('\n2D lattice:')
    print(repr(lattice_2D))

    lattice_3D = Lattice([x, y, z])
    print('\n3D lattice:')
    print(repr(lattice_3D))

    map_vals = np.ones(lattice_2D.nnodes)
    latticemap_2D = LatticeMap(lattice_2D, map_vals)
    print('\n2D latticemap:')
    print(repr(latticemap_2D))

    map_vals = np.ones(lattice_3D.nnodes)
    latticemap_3D = LatticeMap(lattice_3D, map_vals)
    print('\n3D latticemap:')
    print(repr(latticemap_3D))

    from pasam._paths import PATH_TESTFILES
    file = PATH_TESTFILES + 'latticemap2d_float.txt'
    latticemap_from_txt = LatticeMap.from_txt(file)
    print('\nlatticemap from .txt')
    print(repr(latticemap_from_txt))
