# -*- coding: utf-8 -*-
"""  Definitions of the lattice grid elements in 2D and 3D.

Classes
-------
    - :class:`LatticeMap`: (`abstract`) Parent type definition for the
      lattice map factory :class:`LatticeMapFactory`
    - :class:`LatticeMapFactory`: Factory of lattice maps
    - :class:`LatticeMap2D`: Lattice map in two dimensions (x, y)
    - :class:`LatticeMap3D`: Lattice map in three dimensions (x, y, z)

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
# import warnings
# Third party requirements
import numpy as np
# Local imports

# Constants and Variables
_RLIB_MAXLIST = 3
_rlib = reprlib.Repr()
_rlib.maxlist = _RLIB_MAXLIST


class LatticeMap(abc.ABC):
    """`LatticeMap` defines an abstract parent class for the lattice map
    factory.

    A lattice map is defined by a two- or three-dimensional regular grid of
    nodes. Each node is associated to a real value (the map).

    Note:
        Each (explicit) subclass of `LatticeMap` must provide implementation(s)
        of:

        - :meth:`ndim`
    """

    def __repr__(self):
        return f'{self.ndim}'

    def __str__(self):
        return self.__repr__()

    @property
    @abc.abstractmethod
    def ndim(self):
        """ The number of dimensions for the lattice.

        Returns:
            int: Dimensionality of the lattice.
        """


# TODO make LatticeMapFactory abstract
class LatticeMapFactory:
    """`LatticeMapFactory` produces a lattice map object.

    This class defines the factory used to construct lattice map objects. If
    only `x` and `y` are passed as input arguments, it will produce a two
    dimensional lattice map, while in the case where `z` is given, it returns
    a three dimensional lattice map.

    Args:
        x (array_like): Tensor product nodes in x-direction
        y (array_like): Tensor product nodes in x-direction
        z (array_like, optional): Tensor product nodes in x-direction
    """
    pass


class LatticeMap2D(LatticeMap):
    """`LatticeMap2D` defines a two-dimensional lattice map.

    This type inherits from the abstract class :class:`LatticeMap` and is used
    for any map associated to a regular lattice of two-dimensional nodes.

    Args:
        x (array_like): Tensor product nodes in x-direction
        y (array_like): Tensor product nodes in x-direction
        z (array_like, optional): Tensor product nodes in x-direction
    """
    _NDIM = 2

    def __init__(self, x, y):
        self._x = np.asarray(x).ravel()
        self._y = np.asarray(y).ravel()

    def __repr__(self):
        cls_name = type(self).__name__

        x_repr = _rlib.repr(self._x)
        y_repr = _rlib.repr(self._y)

        return f'{cls_name}(x={x_repr}, y={y_repr})'

    @property
    def ndim(self):
        """ Dimensionality of the `LatticeMap2D` (=2).
        """
        return self._NDIM


class LatticeMap3D(LatticeMap):
    """`LatticeMap3D` defines a three-dimensional lattice map.

    This class inherits from the abstract class :class:`LatticeMap` and is used
    for any map associated to a regular lattice of three-dimensional nodes.
    """
    _NDIM = 3

    @property
    def ndim(self):
        """ Dimensionality of the `LatticeMap3D` (=3).
        """
        return self._NDIM


if __name__ == '__main__':
    x = [1, 2, 3, 4.5, 5, 8]
    y = [-1.5, -1, 0, 5.76]
    # z = [-100, 50, 199998.2]

    lattice_2D = LatticeMap2D(x, y)

    print(f'__repr__ of 2D lattice:  {lattice_2D}')
    print(f'__str__ of 2D lattice:   {str(lattice_2D)}')

