# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements in 2D and 3D.

Classes
-------
    - :class:`Lattice`: (`abstract`) Parent type definition for the lattice
      factory :class:`LatticeFactory`
    - :class:`LatticeFactory`: Factory of lattices
    - :class:`Lattice2D`: Lattice in two dimensions (x, y)
    - :class:`Lattice3D`: Lattice in three dimensions (x, y, z)

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
# Third party requirements
import numpy as np
# Local imports

# Constants and Variables
_RLIB_MAXLIST = 3
_rlib = reprlib.Repr()
_rlib.maxlist = _RLIB_MAXLIST


class Lattice(abc.ABC):
    """`Lattice` defines an abstract parent class for the lattice factory.

    A lattice is defined by a two- or three-dimensional regular grid of nodes.

    Note:
        Each (explicit) subclass of `Lattice` must provide implementation(s)
        of:

        - :meth:`ndim`
    """

    def __init__(self, nodes):
        self._nodes = tuple(np.asarray(n).ravel() for n in nodes)

    def __eq__(self, other):
        if isinstance(other, Lattice) and self.nnodes_dim == other.nnodes_dim:
            nodes_eq = [np.all(n_sel == n_oth)
                        for n_sel, n_oth in zip(self._nodes, other._nodes)]
            return all(nodes_eq)
        return False

    def __repr__(self):
        cls_name = type(self).__name__
        nodes_repr = _rlib.repr(self._nodes)
        return f'{cls_name}(nodes={nodes_repr})'

    def __str__(self):
        return self.__repr__()

    @property
    @abc.abstractmethod
    def ndim(self):
        """The number of dimensions for the lattice.

        Returns:
            int: Dimensionality of the lattice.
        """

    @property
    def nnodes_dim(self):
        """The total number of nodes in the lattice.

        Returns:
            tuple: Total number of nodes in each dimension.
        """
        return tuple(len(n) for n in self._nodes)

    @property
    def nnodes(self):
        """The total number of nodes in the lattice.

        Returns:
            int: Total number of nodes.
        """
        return int(np.prod(self.nnodes_dim))


# TODO make LatticeFactory abstract
class LatticeFactory:
    """`LatticeFactory` produces two and three dimensional lattice objects.
    """

    @staticmethod
    def make_lattice(nodes):
        """Produces two and three dimensional lattice objects.

        If only `x` and `y` are provided, it will produce a two dimensional
        lattice, while in the case where `z` is defined, it returns a three
        dimensional lattice.

        Args:
            nodes (tuple): Tensor product nodes ((x, y) or (x, y, z) of
                array_like x, y, and z)

        Returns:
            Lattice: Object defining a lattice
        """
        if len(nodes) == 2:
            return Lattice2D(nodes)
        elif len(nodes) == 3:
            return Lattice3D(nodes)
        else:
            raise ValueError(f'Length of tensor product nodes must either be 2'
                             f' or 3 (the actual length is {len(nodes)})')


class Lattice2D(Lattice):
    """`Lattice2D` defines a two-dimensional lattice.

    This class inherits from the abstract class :class:`Lattice` and is used
    for any regular, two-dimensional tensor product lattice.

    Args:
        nodes (tuple): Tensor product nodes (x, y)

    Examples:
        >>> import pasam as ps
        >>> nodes = ([1, 2], [8, 9, 10.5])
        >>> lfact = ps.LatticeFactory()
        >>> lfact.make_lattice(nodes)
        Lattice2D(nodes=(array([1, 2]), array([ 8. ,  9. , 10.5])))
    """
    _NDIM = 2

    def __init__(self, nodes):
        x, y = nodes
        self._x = np.asarray(x).ravel()
        self._y = np.asarray(y).ravel()

        super().__init__((self._x, self._y))

    @property
    def ndim(self):
        """ Dimensionality of the `Lattice2D`.
        """
        return self._NDIM


class Lattice3D(Lattice):
    """`Lattice3D` defines a three-dimensional lattice.

    This class inherits from the abstract class :class:`Lattice` and is used
    for any associated to a regular lattice of three-dimensional nodes.

    Args:
        nodes (tuple): Tensor product nodes (x, y, z)

    Examples:
        >>> import pasam as ps
        >>> nodes = ([1, 2], [8, 9, 10.5], [-1, 0, 1])
        >>> lfact = ps.LatticeFactory()
        >>> lfact.make_lattice(nodes)
        Lattice3D(nodes=(array([1, 2]), array([ 8. ,  9. , 10.5]), array([-1,  0,  1])))
    """
    _NDIM = 3

    def __init__(self, nodes):
        x, y, z = nodes
        self._x = np.asarray(x).ravel()
        self._y = np.asarray(y).ravel()
        self._z = np.asarray(z).ravel()

        super().__init__((self._x, self._y, self._z))

    @property
    def ndim(self):
        """ Dimensionality of the `Lattice3D`.
        """
        return self._NDIM


class LatticeMap(abc.ABC):
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

    def __init__(self, lattice, map_vals):
        if lattice.nnodes != len(map_vals):
            raise ValueError(f'Uncomparable lattice (nnodes = '
                             f'{lattice.nnodes}) with map values '
                             f'(nval={len(map_vals)})')

        self.lattice = lattice
        self.map_vals = np.asarray(map_vals).ravel()

    def __eq__(self, other):
        if isinstance(other, LatticeMap) and self.lattice == other.lattice:
            return np.all(self.map_vals == other.map_vals)
        return False

    def __repr__(self):
        cls_name = type(self).__name__
        map_vals_repr = _rlib.repr(self.map_vals)
        return f'{cls_name}(lattice={repr(self.lattice)}, ' \
               f'map_vals={map_vals_repr})'

    def __str__(self):
        return self.__repr__()

    @property
    def ndim(self):
        """The number of dimensions for the lattic map.

        Returns:
            int: Dimensionality of the lattice map.
        """
        return self.lattice.ndim


class LatticeMapFactory:
    """`LatticeMapFactory` produces two and three dimensional lattice map
    objects.
    """

    @staticmethod
    def make_latticemap(lattice, map_vals):
        """ Produces two and tree dimensional lattice map objects.

        Args:
            lattice (Lattice): Object defining a lattice
            map_vals (array_like, shape=(n,)): Map values associated to the
                lattice nodes

        Returns:
            LatticeMap: Object defining a lattice map
        """
        ndim = lattice.ndim
        if ndim == 2:
            return LatticeMap2D(lattice, map_vals)
        elif ndim == 3:
            return LatticeMap3D(lattice, map_vals)
        else:
            raise ValueError(f'LatticeMap can not be generated for a lattice'
                             f'of dimentions {ndim}')


class LatticeMap2D(LatticeMap):
    """`LatticeMap2D` defines a two-dimensional lattice map.

    `LatticeMap2D` inhertis form :class:`LatticeMap` where the class behaviour
    is documented in detail.
    """
    # TODO: Add doctests for LatticeMap2D

    def __init__(self, lattice, map_vals):
        super().__init__(lattice, map_vals)


class LatticeMap3D(LatticeMap):
    """`LatticeMap*D` defines a three-dimensional lattice map.

    `LatticeMap3D` inhertis form :class:`LatticeMap` where the class behaviour
    is documented in detail.
    """
    # TODO: Add doctests for LatticeMap3D

    def __init__(self, lattice, map_vals):
        super().__init__(lattice, map_vals)


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [4, 5, 6, 7, 8, 9, 10.5]
    z = [-1, 0, 1]

    lattice_factory = LatticeFactory()
    latticemap_factory = LatticeMapFactory()

    lattice_2D = lattice_factory.make_lattice((x, y))
    print('\n2D lattice:')
    print(repr(lattice_2D))

    lattice_3D = lattice_factory.make_lattice((x, y, z))
    print('\n3D lattice:')
    print(repr(lattice_3D))

    map_vals = np.random.randn(lattice_2D.nnodes)
    latticemap_2D = latticemap_factory.make_latticemap(lattice_2D, map_vals)
    print('\n2D latticemap:')
    print(repr(latticemap_2D))

    map_vals = np.random.randn(lattice_3D.nnodes)
    latticemap_3D = latticemap_factory.make_latticemap(lattice_3D, map_vals)
    print('\n3D latticemap:')
    print(repr(latticemap_3D))
