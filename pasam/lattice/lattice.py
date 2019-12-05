# -*- coding: utf-8 -*-
"""  Definitions of the lattice grid elements in 2D and 3D.

Classes
-------
    - :class:`LatticeMap`: (`abstract`) Parent type definition for the
      lattice map factory :class:`LatticeMapFactory`
    - class:`LatticeMapFactory`: Factory of lattice maps
    - class:`LatticeMap2D`: Lattice map in TWO dimensions (x, y)
    - class:`LatticeMap3D`: Lattice map in THREE dimensions (x, y, z)

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
# import reprlib
# import warnings
# Third party requirements
# Local imports


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

    @property
    @abc.abstractmethod
    def ndim(self):
        """ The number of dimensions for the lattice.

        Returns:
            int: Dimensionality of the lattice.
        """


# class LatticeMapFactory():
#     pass


class LatticeMap2D(LatticeMap):
    """`LatticeMap2D` defines a two-dimensional lattice map.

    This type inherits from the abstract class :class:`LatticeMap` and is used
    for any map associated to a regular lattice of two-dimensional nodes.
    """
    _NDIM = 2

    @property
    def ndim(self):
        """ Dimensionality of the `LatticeMap2D` (=2).
        """
        return self._NDIM


class LatticeMap3D(LatticeMap):
    """`LatticeMap3D` defines a three-dimensional lattice map.

    This type inherits from the abstract class :class:`LatticeMap` and is used
    for any map associated to a regular lattice of two-dimensional nodes.
    """
    _NDIM = 3

    @property
    def ndim(self):
        """ Dimensionality of the `LatticeMap3D` (=3).
        """
        return self._NDIM
