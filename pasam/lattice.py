# -*- coding: utf-8 -*-
"""This module defines the lattice grid elements and the associated value maps.
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
from pasam._settings import NP_ORDER, RLIB_MAXLIST
import pasam.utils as utl

# Constants
_rlib = reprlib.Repr()
_rlib.maxlist = RLIB_MAXLIST


# Lattice and LatticeMap Objects
class Lattice:
    """`Lattice` defines the computational lattice.

    A lattice is defined by a one-, two-, or three-dimensional regular grid of
    strictly increasing node values. The grid can then be considered as a
    tensor product of all nodes, as for example::

        Lattice(nodes=[[0, 2, 4, 6], [-1, 0, 1]])

            +1  +-------+-------+-------+
                |       |       |       |
             0  +-------+-------+-------+
                |       |       |       |
            -1  +-------+-------+-------+
                0       2       4       6

    Args:
        nodes (list of array_like): Tensor product nodes.

    Attributes:
        nodes (list of ndarray): Tensor product nodes.

    """

    def __eq__(self, other):
        """Compares if all nodes are the same in both instances."""
        if isinstance(other, Lattice) and self.nnodes_dim == other.nnodes_dim:
            equal = [np.all(n_sel == n_oth)
                     for n_sel, n_oth in zip(self.nodes, other.nodes)]
            return all(equal)
        return False

    def __getitem__(self, key):
        """Defines the slicing of a lattice."""
        if not isinstance(key, tuple) or len(key) != len(self.nodes):
            raise ValueError(msg.err1004(self, key))
        nodes = [n[k] for n, k in zip(self.nodes, key)]

        # Reduce the singleton dimension
        for inode, node in list(enumerate(nodes))[::-1]:
            if isinstance(node, numbers.Number):
                del nodes[inode]
        return self.__class__(nodes)

    def __init__(self, nodes):
        # List of strictly increasing ndarrays
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

    A lattice map is the combination of a lattice and a value map associated to
    in a one-, two-, or three-dimensional regular grid of nodes:
    The grid can then be considered as a tensor product of all nodes, as for
    example::

        nodes = [[0, 2, 4, 6], [-1, 0, 1]]
        values = [
            0.1, 0,2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ]
        LatticeMap(nodes, values)

            +1 0.9-----1.0-----1.1-----1.2
                |       |       |       |
             0 0.5-----0.6-----0.7-----0.8
                |       |       |       |
            -1 0.1-----0.2-----0.3-----0.4
                0       2       4       6

    The index order is Fortran-like such that it first iterates along the
    lowest dimension.

    Args:
        lattice (Lattice or list of array_like): Object defining the
            computational lattice.
        values (array_like, shape=(n,)): Map values associated to the lattice
            nodes.
        dtype (data-type, optional): The desired data-type for the map_values.
            If not given, then the type will be determined as the minimum
            requirement by `numpy`.

    Attributes:
        lattice (Lattice): Object defining the computational lattice.
        values (ndarray, shape=(n,)): Map values associated to the lattice
            nodes.
    """

    def __add__(self, other):
        """Supports `LatticeMap` + `LatticeMap` as well as `LatticeMap` +
        `number.Number`"""
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.values + other.values)
            else:
                raise ValueError(msg.err1002('+'))

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.values + other)

        return NotImplemented

    def __eq__(self, other):
        """Equality is based on `lattice` and `values`."""
        if isinstance(other, LatticeMap) and self.lattice == other.lattice:
            return np.all(self.values == other.values)
        return False

    def __getitem__(self, key):
        """Uses the slicing of numpy for the values."""
        lattice = self.lattice[key]
        nnodes_dim = self.lattice.nnodes_dim
        values = self.values.reshape(nnodes_dim, order=NP_ORDER)
        values = values[key].ravel(order=NP_ORDER)
        return self.__class__(lattice=lattice, values=values)

    def __init__(self, lattice, values, dtype=None):
        # Support both:
        # - LatticeMap(lattice, values)
        # - LatticeMap(nodes, values)
        if not isinstance(lattice, Lattice):
            lattice = Lattice(lattice)

        # Transform values to a one-dimensional ndarray
        values = np.asarray(values, dtype=dtype).ravel(order=NP_ORDER)
        if lattice.nnodes != len(values):
            raise ValueError(msg.err1003(lattice.nnodes, len(values)))
        self.lattice = lattice
        self.values = values

    def __mul__(self, other):
        """Supports `LatticeMap` * `LatticeMap` as well as `LatticeMap` *
        `numbers.Number`.
        """
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.values * other.values)
            else:
                raise ValueError(msg.err1002('*'))

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.values * other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        cls_name = type(self).__name__
        values_repr = _rlib.repr(self.values)
        return f'{cls_name}(lattice={repr(self.lattice)}, ' \
               f'values={values_repr})'

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
    def from_txt(cls, file):
        """Produces lattice map objects from .txt files.

        The structure of a latticemap text file is reported in the function
        :meth:`readfile_latticemap`.

        Args:
            file (str or pathlib.Path): File or filename.

        Returns:
            LatticeMap: Lattice map from file.
        """
        return readfile_latticemap(file)


# Methods
def readfile_latticemap(file) -> LatticeMap:
    """Reads a lattice map text file.

    The structure of the lattice map file is as follows::

            -----------------------------------------
            |  <nnodes_dim>                         |
            |  <nodes_x>                            |
            |  <nodes_y>                            |
            |  (<nodes_z>)                          |
            |  values(x=0,...,n-1; y=0, (z=0))      |
            |  values(x=0,...,n-1; y=1, (z=0))      |
            |  ...                                  |
            |  values(x=0,...,n-1; y=m-1, (z=0))    |
            |  values(x=0,...,n-1; y=0, (z=1))      |
            |  ...                                  |
            |  values(x=0,...,n-1; y=0, (z=r-1))    |
            -----------------------------------------

    In the case of two-dimensional maps, the quantities in parentheses are
    omitted.

    Args:
        file (str or pathlib.Path): File or filename.

    Returns:
        LatticeMap: Lattice map from file.
    """
    nodes, values = readfile_nodes_values(file)
    return LatticeMap(nodes, values)


def readfile_nodes_values(file):
    """Similar as :meth:`readfile_latticemap` but directly returns the nodes
    and the values.
    """
    lines = utl.readlines_(file, remove_blank_lines=True)

    # Number of nodes per dimension (defined in lines[0])
    nnodes_dim = utl.findall_num_in_str(lines[0])
    ndim = len(nnodes_dim)

    # Definition of the lattice (defined in lines[1:ndim+1])
    lines_nodes, lines_values = lines[1:ndim + 1], lines[ndim + 1:]
    nodes = [utl.findall_num_in_str(line) for line in lines_nodes]

    # Definition of the values (defined in lines[ndim+1:])
    values = [utl.findall_num_in_str(line) for line in lines_values]

    # Flatten the list of values
    values = np.asarray([val for vals in values for val in vals])

    return nodes, values
