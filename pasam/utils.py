# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :func:`condmap_from_file`: Condition map from txt file.
    - :func:`condmap_from_point`: Condition map from conditioning point.
    - :func:`findall_num_in_str`: Extracts all numbers from a string.
    - :func:`readlines_`: Reads txt file (possibility to remove empty lines).
    - :func:`readfile_latticemap`: Reads a latticemap file.
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
import re
from pathlib import Path
# Third party requirements
import numpy as np
# Local imports


# `Public` methods
def permission_map_from_file(file):
    """Reads a permission map from a given .txt file.

    Args:
        file (str or pathlib.Path): File or filename.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.

    """
    _, _, vals = readfile_latticemap(file)
    map_vals = _ams_map_to_bool_map(vals)
    return map_vals


def condmap_from_point(components, nodes):
    """Generates a condition map from a conditioning point.

    Args:
        components (array_like, shape=(n,)): Coordinates of point.
        nodes (list): Tensor product nodes.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.
    """
    # TODO: add test for condmap_from_point
    map_vals = _ams_point_to_bool(components, nodes)
    return map_vals


def findall_num_in_str(s):
    """Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (``float`` or ``int``)
    """
    pat = r'-?[0-9]+\.?[0-9]*'
    nums = re.findall(pat, s)
    return [_str2num(n) for n in nums]


def readlines_(file, remove_blank_lines=False, hint=-1):
    """Reading txt file (similar to builtin ``readlines``).

    In addition to the standard implementation of ``readlines`` for
    ``_io.TextIOWrapper``, this version provides the possibility to remove
    empty text lines.

    Args:
        file (str or pathlib.Path): File or filename.
        remove_blank_lines (bool, optional): Remove blank lines.
        hint (int, optional): Limit of the cumulative size (in bytes/
            characters) of all lines to be read.

    Returns:
        list: All non empty lines of text file
    """
    if isinstance(file, Path):
        file = str(file)

    with open(file, 'r') as txtfile:
        lines = txtfile.readlines()

    if remove_blank_lines:
        lines = [line for line in lines if not _is_blank(line)]
    return lines


def readfile_latticemap(file):
    """Reads a latticemap file.

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

    Returns:
        nnodes_dim (tuple): The number of nodes per dimension
        nodes (list): The nodes given in the file.
        map_vals (ndarray, shape=(n,)): Map values given in the file.
    """
    _REM_BLANK = True
    lines = readlines_(file, remove_blank_lines=_REM_BLANK)

    # Number of nodes per dimenstion (defined in lines[0])
    nnodes_dim = findall_num_in_str(lines[0])
    ndim = len(nnodes_dim)

    # Definition of the lattice (defined in lines[1:ndim+1])
    lines_nodes, lines_map_vals = lines[1:ndim + 1], lines[ndim + 1:]
    nodes = [findall_num_in_str(line) for line in lines_nodes]

    # Definition of the map_vals (defined in lines[ndim+1:])
    map_vals = [findall_num_in_str(line) for line in lines_map_vals]

    return nnodes_dim, nodes, map_vals


# `Private` Methods
def _ams_point_to_bool(components, nodes):
    """Generates a condition map from a conditioning point.

    Args:
        components (array_like, shape=(n,)): Coordinates of point.
        nodes (list): Tensor product nodes.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.
    """
    # TODO: add tests for _ams_point_to_bool
    # TODO: refactor _ams_point_to_bool
    # TODO: make `specs` an argument
    # TODO: maybe rename it
    # Movement specifications
    specs = {
        'type':                         'GantryDominant',
        # Table rotation per gantry rotation:
        #   1: 3 neighbors (+- 2 table degrees per 2 gantry degrees)
        #   2: 5 neighbors (+- 4 table degrees per 2 gantry degrees)
        'table_per_gantry_rot':         2,
    }

    factory = _TrajectoryRegionFactory.make_trajectoryregion
    pnt_traj = factory(components, nodes, specs)

    return pnt_traj.make_condmap()


def _ams_map_to_bool_map(vals):
    """Inverts `0` to `True` and `1` to `False`.

    By default, the AMS generates files where the permitted nodes have
    values `0` and the blocked nodes have value `1`.
    """
    # TODO: add tests for _ams_vals_to_bool
    # TODO: refactor _ams_vals_to_bool
    # TODO: maybe rename it
    _TRUE_VALS  = (-0.1, 0.1)
    _FALSE_VALS = ( 0.9, 1.1)

    vals = np.asarray(vals).ravel()
    map_vals = np.asarray(vals, dtype=bool)

    ind_true = np.logical_and(vals > _TRUE_VALS[0], vals < _TRUE_VALS[1])
    ind_false = np.logical_and(vals > _FALSE_VALS[0], vals < _FALSE_VALS[1])

    # Check whether all values have been covered
    if np.sum(np.logical_xor(ind_true, ind_false)) != len(map_vals):
        ind_not_unique = np.logical_not( np.logical_xor(ind_true, ind_false) )
        ind = np.where(ind_not_unique)[0]
        raise ValueError(f'Values(s) {vals[ind]} are not uniquely identified')

    map_vals[ind_true] = True
    map_vals[ind_false] = False
    return map_vals


def _is_blank(s):
    """Check whether a string only contains whitespace characters.
    """
    return bool(re.match(r'^\s+$', s))


def _str2num(s):
    """Generates ``int`` or ``float`` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)


# `Private` Classes
# TODO: Review and refactor all trajectories classes (evtl. rename them)
# TODO: Test all trajectory classes
# TODO: Define _PointTrajectoryGantryDominant3D
class _TrajectoryRegion(abc.ABC):
    """`_TrajectoryRegion` defines an abstract parent class for the machine
    movement restrictions by a point in the space.

    Notes:
        Any sub-class of `_PointTrajectory` must provide an implementation of

            - :meth:`make_condmap`
    """
    
    @abc.abstractmethod
    def make_condmap(self):
        """Generates a condition map from a conditioning point.

        Returns:
            ndarray: Boolean array for permitted (True) and blocked (False)
                nodes.
        """


class _TrajectoryRegionFactory:
    """`_TrajectoryRegionFactory` produces instances of ``_PointTrajectory``
    according to the specifications.
    """

    @staticmethod
    def make_trajectoryregion(components, nodes, specs):
        _dim = len(nodes)
        if _dim == 2 and specs['type'] == 'GantryDominant':
            return _TrajectoryRegionGantryDominant2D(components, nodes, specs)
        else:
            msg = f'No sub-class implementation for ' \
                  f'dim={_dim} and type="{specs["type"]}"'
            raise ValueError(msg)


class _TrajectoryRegionGantryDominant2D(_TrajectoryRegion):
    """`_TrajectoryRegionGantryDominant2D` is the usual 2D gantry dominated
    movement restriction class.

    Args:
        components (array_like, shape=(n,)): Coordinates of point.
        nodes (list): Tensor product nodes.
        specs (dict): Specifications for the different trajectory regions.
    """
    # TODO: maybe rename components into restriction
    # TODO: if doing so, we also need to do it in `_ams_point_to_bool`
    # TODO: maybe make components (restriction) default equal to None

    def __init__(self, components, nodes, specs):
        self._components = components
        self._nodes = nodes
        self._specs = specs

    def make_condmap(self):
        _igantry = 0
        _itable = 1

        nnodes_dim = tuple(len(n) for n in self._nodes)
        if self._components is None:
            return np.ones(nnodes_dim, dtype=bool)
        map_vals = np.zeros(nnodes_dim, dtype=bool)

        gantry_nodes = np.asarray(self._nodes[_igantry]).ravel()
        table_nodes = np.asarray(self._nodes[_itable]).ravel()

        gantry_comp = self._components[_igantry]
        table_comp = self._components[_itable]

        tpg_rot = self._specs['table_per_gantry_rot']
        for inode, node in enumerate(gantry_nodes):
            table_range = abs(node - gantry_comp)*tpg_rot
            v_min, v_max = table_comp - table_range, table_comp + table_range

            itable_min = np.argmin(np.abs(table_nodes - v_min))
            itable_max = np.argmin(np.abs(table_nodes - v_max))

            map_vals[inode, itable_min:itable_max+1] = True

        return map_vals


