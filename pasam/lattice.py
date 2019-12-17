# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements in 2D and 3D.

Classes
-------
    - :class:`Lattice`: (`abstract`) Parent type definition for the lattice
      factory :class:`LatticeFactory`
    - :class:`LatticeFactory`: Factory of lattices
    - :class:`Lattice2D`: Lattice in two dimensions
    - :class:`Lattice3D`: Lattice in three dimensions
    - :class:`LatticeMap`: (`abstract`) Parent type definition for the
      latticemap factory :class:`LatticeMapFactory`
    - :class:`LatticeMapFactory`: Factory of latticemaps
    - :class:`LatticeMap2D`: Latticemap in two dimensions
    - :class:`LatticeMap3D`: Latticemap in three dimensions

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
import pasam.utils as utl

# Constants and Variables
_RLIB_MAXLIST = 3
_rlib = reprlib.Repr()
_rlib.maxlist = _RLIB_MAXLIST


class Lattice:
    """`Lattice` defines the computational lattice.

    A lattice is defined by a two- or three-dimensional regular grid of nodes.

    Args:
        nodes (list): Tensor product nodes.

    Attributes:
        nodes (list): Tensor product nodes.

    """

    def __init__(self, nodes):
        self.nodes = list(np.asarray(n).ravel() for n in nodes)

    def __eq__(self, other):
        if isinstance(other, Lattice) and self.nnodes_dim == other.nnodes_dim:
            nodes_eq = [np.all(n_sel == n_oth)
                        for n_sel, n_oth in zip(self.nodes, other.nodes)]
            return all(nodes_eq)
        return False

    def __repr__(self):
        cls_name = type(self).__name__
        nodes_repr = _rlib.repr(self.nodes)
        return f'{cls_name}(nodes={nodes_repr})'

    def __str__(self):
        return self.__repr__()

    @property
    def ndim(self):
        """The number of dimensions for the lattice.

        Returns:
            int: Dimensionality of the lattice.
        """
        return len(self.nodes)

    @property
    def nnodes_dim(self):
        """The total number of nodes in the lattice.

        Returns:
            tuple: Total number of nodes in each dimension.
        """
        return tuple(len(n) for n in self.nodes)

    @property
    def nnodes(self):
        """The total number of nodes in the lattice.

        Returns:
            int: Total number of nodes.
        """
        return int(np.prod(self.nnodes_dim))


class LatticeMap:
    """`LatticeMap` defines an abstract parent class for any map of real values
    that is associated to a lattice.

    Args:
        lattice (Lattice): Object defining a lattice
        map_vals (array_like, shape=(n,)): Map values associated to the lattice
            nodes

    Attributes:
        lattice (Lattice): Object defining a lattice
        map_vals (ndarray, shape=(n,)): Map values associated to the lattice
            nodes
    """

    def __add__(self, other):
        """Supports ``LatticeMap + LatticeMap`` as well as ``LatticeMap +
        number.Number``"""
        if isinstance(other, LatticeMap):
            if self.lattice is other.lattice or self.lattice == other.lattice:
                return LatticeMap(self.lattice, self.map_vals + other.map_vals)
            raise ValueError('unsupported operation + for different Lattice objects')

        elif isinstance(other, numbers.Number):
            return LatticeMap(self.lattice, self.map_vals + other)

        return NotImplemented

    def __eq__(self, other):
        """Equality is based on `lattice` and `map_vals`."""
        if isinstance(other, LatticeMap) and self.lattice == other.lattice:
            return np.all(self.map_vals == other.map_vals)
        return False

    def __init__(self, lattice, map_vals):
        map_vals_flat = np.asarray(map_vals).ravel()
        if lattice.nnodes != len(map_vals_flat):
            raise ValueError(f'Uncomparable lattice (nnodes = '
                             f'{lattice.nnodes}) with map values '
                             f'(nval={len(map_vals_flat)})')

        self.lattice = lattice
        self.map_vals = map_vals_flat

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
        """The number of dimensions for the lattic map.

        Returns:
            int: Dimensionality of the lattice map.
        """
        return self.lattice.ndim

    @classmethod
    def from_txt(cls, file, lattice=None):
        """Produces two and three dimensional lattice map objects from .txt
        files.

        The structure of the txt file is as follows::

            ----------------------------------------
            | <nnode_dim>                          |
            | <nodes_x>                            |
            | <nodes_y>                            |
            | (<nodes_z>)                          |
            | map_vals(x=0,...,n-1; y=0, (z=0))    |
            | map_vals(x=0,...,n-1; y=1, (z=0))    |
            | ...                                  |
            | map_vals(x=0,...,n-1; y=m-1, (z=0))  |
            | map_vals(x=0,...,n-1; y=0, (z=1))    |
            | ...                                  |
            | map_vals(x=0,...,n-1; y=0, (z=r-1))  |
            ----------------------------------------

        In the case of two-dimensional maps, the quantities in parentheses are
        omitted.

        Args:
            file (str or pathlib.Path): File or filename.
            lattice (Lattice, optional): Object defining a lattice.

        Returns:
            LatticeMap: Object defining a lattice map
        """
        _REM_BLANK = True
        lines = utl.readlines_(file, remove_blank_lines=_REM_BLANK)

        # Number of nodes per dimenstion (defined in lines[0])
        nnodes_dim = utl.findall_num_in_str(lines[0])
        ndim = len(nnodes_dim)

        # Definition of the lattice (defined in lines[1:ndim+1])
        lines_nodes, lines_map_vals = lines[1:ndim+1], lines[ndim+1:]
        if not lattice:
            nodes = [utl.findall_num_in_str(line) for line in lines_nodes]
            lattice = Lattice(nodes)

        # Definition of the map_vals (defined in lines[ndim+1:])
        map_vals = [utl.findall_num_in_str(line) for line in lines_map_vals]

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
