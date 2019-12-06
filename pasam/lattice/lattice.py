# -*- coding: utf-8 -*-
"""  Definitions of the lattice grid elements in 2D and 3D.

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
# import warnings
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


# TODO make LatticeFactory abstract
class LatticeFactory:
    """`LatticeFactory` produces a lattice object.

    This class defines the factory used to construct lattice objects. If only
    `x` and `y` are provided, it will produce a two dimensional lattice, while
    in the case where `z` is defined, it returns a three dimensional lattice.

    Args:
        x (array_like): Tensor product nodes in x-direction
        y (array_like): Tensor product nodes in x-direction
        z (array_like, optional): Tensor product nodes in x-direction
    """
    pass


class Lattice2D(Lattice):
    """`Lattice2D` defines a two-dimensional lattice.

    This class inherits from the abstract class :class:`Lattice` and is used
    for any regular, two-dimensional tensor product lattice.

    Args:
        x (array_like): Tensor product nodes in x-direction
        y (array_like): Tensor product nodes in x-direction
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
        """ Dimensionality of the `Lattice2D`.
        """
        return self._NDIM


class Lattice3D(Lattice):
    """`Lattice3D` defines a three-dimensional lattice.

    This class inherits from the abstract class :class:`Lattice` and is used
    for any associated to a regular lattice of three-dimensional nodes.
    """
    _NDIM = 3

    @property
    def ndim(self):
        """ Dimensionality of the `Lattice3D`.
        """
        return self._NDIM


if __name__ == '__main__':
    x = [1, 2, 3, 4.5, 5, 8]
    y = [-1.5, -1, 0, 5.76]
    # z = [-100, 50, 199998.2]

    lattice_2D = Lattice2D(x, y)

    print(f'__repr__ of 2D lattice:  {lattice_2D}')
    print(f'__str__ of 2D lattice:   {str(lattice_2D)}')

