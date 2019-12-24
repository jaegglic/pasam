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
import pasam.utils as utl

# Constants and Variables
_RLIB_MAXLIST = 3
_rlib = reprlib.Repr()
_rlib.maxlist = _RLIB_MAXLIST
_NP_ORDER = 'F'


# Condition Objects
class Condition(abc.ABC):
    """`Condition` defines an abstract parent class for restrictions in the
    path sampling.

    Notes:
        Any sub-class of `Condition` must provide an implementation of

            - :meth:`permission_map`
    """

    @abc.abstractmethod
    def permission_map(self, lattice):
        """Produces a permission map with permitted (``True``) and blocked
        (``False``) lattice nodes.

        Args:
            lattice (Lattice): Object defining a lattice.

        Returns:
            LatticeMap: Lattice map issued from the condition.
        """


class ConditionFile(Condition):
    """`ConditionFile` defines a condition from a .txt file.

    Args:
        file (str or pathlib.Path): File or filename.
    """

    def __init__(self, file):
        self._file = file

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(file={self._file})'

    def __str__(self):
        return self.__repr__()

    # Definition of the abstract method in `Condition`
    def permission_map(self, lattice):
        map_vals = utl.permission_map_from_condition_file(self._file)
        return LatticeMap(lattice, map_vals)


class ConditionPoint(Condition):
    """`ConditionPoint` defines a condition point in a lattice grid.

    Args:
        components (array_like, shape=(n,)): Coordinate components.
    """

    def __eq__(self, other):
        if isinstance(other, ConditionPoint):
            return np.all(self._components == other._components)
        elif isinstance(other, tuple) or isinstance(other, list):
            return np.all(self._components == np.asarray(other))
        return False

    def __init__(self, components):
        self._components = np.asarray(components)

    def __len__(self):
        return len(self._components)

    def __repr__(self):
        cls_name = type(self).__name__
        components_repr = _rlib.repr(self._components)
        return f'{cls_name}(components={components_repr})'

    def __str__(self):
        return self.__repr__()

    # Definition of the abstract method in `Condition`
    def permission_map(self, lattice):
        point = self._components
        nodes = lattice.nodes
        map_vals = utl.permission_map_from_condition_point(point, nodes)
        return LatticeMap(lattice, map_vals)


# Lattice and LatticeMap Objects
class Lattice:
    """`Lattice` defines the computational lattice.

    A lattice is defined by a one-, two-, or three-dimensional regular grid of
    nodes. The nodes in one dimension is a set of strictly increasing values.

    Args:
        nodes (list): Tensor product nodes.

    Attributes:
        nodes (list): Tensor product nodes.

    """

    def __eq__(self, other):
        if isinstance(other, Lattice) and self.nnodes_dim == other.nnodes_dim:
            nodes_eq = [np.all(n_sel == n_oth)
                        for n_sel, n_oth in zip(self.nodes, other.nodes)]
            return all(nodes_eq)
        return False

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
        lattice (Lattice): Object defining a lattice.
        map_vals (array_like, shape=(n,)): Map values associated to the lattice
            nodes.
        dtype (data-type, optional): The desired data-type for the map_values.
            If not given, then the type will be determined as the minimum
            requirement by `numpy`.

    Attributes:
        lattice (Lattice): Object defining a lattice
        map_vals (ndarray, shape=(n,)): Map values associated to the lattice
            nodes
    """

    def __add__(self, other):
        """Supports `LatticeMap` + `LatticeMap` as well as `LatticeMap` +
        `number.Number`"""
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.map_vals + other.map_vals)
            else:
                raise ValueError(msg.err1002)

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.map_vals + other)

        return NotImplemented

    def __eq__(self, other):
        """Equality is based on `lattice` and `map_vals`."""
        if isinstance(other, LatticeMap) and self.lattice == other.lattice:
            return np.all(self.map_vals == other.map_vals)
        return False

    def __init__(self, lattice, map_vals, dtype=None):
        map_vals = np.asarray(map_vals, dtype=dtype).ravel(order=_NP_ORDER)
        if lattice.nnodes != len(map_vals):
            raise ValueError(msg.err1003(lattice.nnodes, len(map_vals)))

        self.lattice = lattice
        self.map_vals = map_vals

    def __mul__(self, other):
        """Supports multiplication by ``numbers.Number``."""
        if isinstance(other, numbers.Number):
            map_vals = self.map_vals * other
            return LatticeMap(self.lattice, map_vals)
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
            lattice (Lattice, optional): Object defining a lattice.

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
