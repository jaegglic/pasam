# -*- coding: utf-8 -*-
"""Definitions of the trajectory sampling algorithms.

Classes
-------
    - :class:`Sampler`: (abstract) Parent class for each sampling algorithm.
    - :class:`SamplerFactory`: Factory for the different samplers.
    - :class:`SamplerGantryDominant2D`: Gantry dominant 2D trajectory sampler.

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
# Third party requirements
# Local imports
import pasam._settings as settings
import pasam._messages as msg

# Constants

# Constants and Variables
_NP_ORDER = settings.NP_ORDER
_rlib = reprlib.Repr()
_rlib.maxlist = settings.RLIB_MAXLIST


class Sampler(abc.ABC):
    """Parent class for all sampling algorithms.

    Args:
        prior_map (LatticeMap): Prior sampling map for trajectory points.

    """

    def __init__(self, prior_map):
        self._prior_map = prior_map

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(prior_map={self._prior_map})'

    def __str__(self):
        return self.__repr__()

    @abc.abstractmethod
    def __call__(self, conditions):
        """Start the sampling algorithm.

        Args:
            conditions (list of Condition): Set of conditions.

        Returns:
            Trajectory: Sampled trajectory
        """


class SamplerFactory:
    """Factory for creating the sampling algorithm objects.
    """

    @staticmethod
    def make(type_, **kwargs):
        """Creates :class:`Sampler` objects.

        Args:
            type_ (str): Trajectory sampler type.
            kwargs (dict): Type specific arguments.

        Returns:
            Sampler: Sampling algorithm object.
        """

        if type_ == 'GantryDominant2D':
            return SamplerGantryDominant2D(**kwargs)
        else:
            raise ValueError(msg.err3000(type_))


class SamplerGantryDominant2D(Sampler):
    """`SamplerGantryDominant2D` is the usual 2D gantry dominated trajectory
    movement sampler class.
    """

    def __init__(self, prior_map):
        super().__init__(prior_map)

    def __call__(self, conditions):
        """Start the sampling algorithm.

        Args:
            conditions (list of Condition): Set of conditions.

        Returns:
            Trajectory: Sampled trajectory
        """

        # _gantry_ind = 0       # maybe define it in the settings!!!
        # cnd_pts = [cnd for cnd in conditions if isinstance(cond, ConditionPoint)]
        # gantry_cnd_ind = [self.prior_map.point_to_index(cnd.point)[_gantry_ind]]
        # smpl_ind = [ind for ind in gantry_all_ind if not ind in gantry_cnd_ind]

        pass