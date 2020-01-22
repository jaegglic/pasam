# -*- coding: utf-8 -*-
"""Definitions of the lattice grid elements and the associated objects.

Classes
-------
    - :class:`Trajectory`: Definition of dynamic trajectories.

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
import reprlib
# Third party requirements
# Local imports
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


if __name__ == '__main__':
    points = [(1, 2), (3, 4), (5, 6)]

    traj = Trajectory(points)
    print('\n2D trajectory:')
    print(str(traj))
