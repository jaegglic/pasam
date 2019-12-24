# -*- coding: utf-8 -*-
"""Definitions of some package tools.

Generic methods
---------------
    - :func:`findall_num_in_str`: Extracts all numbers from a string.
    - :func:`isincreasing`: Checks if a sequence of values increases.
    - :func:`permission_map_from_file`: Permission map from txt file.
    - :func:`permission_map_from_point`: Permission map from conditioning point.
    - :func:`readfile_latticemap`: Reads a latticemap file.
    - :func:`readlines_`: Reads txt file (possibility to remove empty lines).
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
import pasam._messages as msg
import pasam._settings as settings

# Constants
_NP_ORDER = 'F'


# `Public` methods
def findall_num_in_str(s):
    """Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (`float` or `int`)
    """
    pat = r'-?[0-9]+\.?[0-9]*'
    nums = re.findall(pat, s)
    return [_str2num(n) for n in nums]


def isincreasing(vals, strictly=True):
    """Checks if a set of values is increasing.

    Args:
        vals (array_like, shape=(n,)): Set of values
        strictly (bool, optional): Strictly (`strictly=True`) or simply
            (`strictly=False`) increasing.

    Returns:
        bool
    """
    vals = np.asarray(vals).ravel(order=_NP_ORDER)
    if strictly:
        return np.all(vals[:-1] < vals[1:])
    else:
        return np.all(vals[:-1] <= vals[1:])


def permission_map_from_condition_file(file):
    """Reads a permission map from a given .txt file.

    Args:
        file (str or pathlib.Path): File or filename.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.

    """
    _, _, vals = readfile_latticemap(file)

    # !!! Values are inverted in the ams map !!!
    map_vals = _ams_val_map_to_bool_map(vals)

    return map_vals


def permission_map_from_condition_point(point, nodes):
    """Generates a permission map from a conditioning point.

    Args:
        point (array_like, shape=(n,)): Coordinates of the condition point.
        nodes (list): Tensor product nodes.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.
    """
    map_vals = _ams_condition_point_to_bool_map(point, nodes)
    return map_vals


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
        list(str): All non empty lines of text file
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

    The structure of the latticemap file is as follows::

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

    # Flatten the list of values
    map_vals = np.asarray([val for vals in map_vals for val in vals])

    return nnodes_dim, nodes, map_vals


# `Private` Methods
def _ams_condition_point_to_bool_map(point, nodes, specs=None):
    """Generates a permission map from a conditioning point.

    Args:
        point (array_like, shape=(n,)): Coordinates of the condition point.
        nodes (list): Tensor product nodes.
        specs (dict, optional): Specifications for the trajectory permission.
            Fields are:

            - 'type': Type of trajectory permission (`str`).
            - 'ndim': Number of dimensions (`int`).
            - ... (type related specifications)

    Returns:
        ndarray, shape=(n,): Boolean array for permitted (True) and blocked
            (False) nodes.
    """
    if not specs:
        specs = settings.AMS_TRAJ_PERM_SPECS

    # Specifications for the trajectory restriction by a point
    specs['condition_point'] = point
    specs['nodes'] = nodes

    factory = _TrajectoryPermissionFactory
    traj_perm = factory.make(specs)

    return traj_perm.permission_map()


def _ams_val_map_to_bool_map(vals):
    """Inverts `0` to `True` and `1` to `False`.

    By default, the AMS generates files where the permitted nodes have
    values `0` and the blocked nodes have value `1`.
    """
    # Value intervals for permitted (True) and blocked (False) nodes
    INTERVAL_TRUE  = (-0.1, 0.1)
    INTERVAL_FALSE = ( 0.9, 1.1)

    vals = np.asarray(vals).ravel(order=_NP_ORDER)
    map_vals = np.asarray(vals, dtype=bool)

    ind_true = np.logical_and(vals > INTERVAL_TRUE[0], vals < INTERVAL_TRUE[1])
    ind_false = np.logical_and(vals > INTERVAL_FALSE[0], vals < INTERVAL_FALSE[1])

    # Check whether all values have been covered
    if np.sum(np.logical_xor(ind_true, ind_false)) != len(map_vals):
        ind_not_unique = np.logical_not( np.logical_xor(ind_true, ind_false) )
        ind = np.where(ind_not_unique)[0]
        raise ValueError(f'Values(s) {vals[ind]} are not uniquely identified')

    # Tag nodes according to the permission
    map_vals[ind_true] = True
    map_vals[ind_false] = False

    return map_vals


def _is_blank(s):
    """Check whether a string only contains whitespace characters.
    """
    return bool(re.match(r'^\s+$', s))


def _str2num(s):
    """Generates `int` or `float` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)


# `Private` Classes
class _TrajectoryPermission(abc.ABC):
    """`_TrajectoryPermission` defines an abstract parent class for the machine
    movement restrictions by a point in the space.

    Notes:
        Any sub-class of `_PointTrajectory` must provide an implementation of

            - :meth:`permission_map`
    """
    
    @abc.abstractmethod
    def permission_map(self):
        """Generates a condition map from a conditioning point.

        Returns:
            ndarray: Boolean array for permitted (True) and blocked (False)
                nodes.
        """


class _TrajectoryPermissionFactory:
    """`_TrajectoryPermissionFactory` produces instances of
    :class:`_PointTrajectory` according to the specifications.
    """

    @staticmethod
    def make(specs):
        """Creates `_TrajectoryPermission` objects.

        Args:
            specs (dict): Specifications for the trajectory permission. Fields
                are:

                - 'type': Type of trajectory permission (`str`).
                - 'ndim': Number of dimensions (`int`).
                - ... (type related specifications)

        Returns:
            _TrajectoryPermission: Trajectory permission object.
        """
        type_ = specs['type']
        ndim = specs['ndim']
        if ndim == 2 and type_ == 'GantryDominant':
            return _TrajectoryPermissionGantryDominant2D(specs)
        else:
            raise ValueError(msg.err0000(ndim, type_))


class _TrajectoryPermissionGantryDominant2D(_TrajectoryPermission):
    """`_TrajectoryPermissionGantryDominant2D` is the usual 2D gantry dominated
    movement restriction class.

    Args:
        specs (dict): Specifications for the trajectory permission. Fields are:

        - 'nodes': Tensor product nodes (`list`).
        - 'condition_point': Condition point (`tuple` or `None`)
        - 'ratio_table_gantry_rotation': Maximum allowed ratio between table
            and gantry angle rotation.
    """

    def __init__(self, specs):
        self._nodes = specs['nodes']
        self._condition_point = specs['condition_point']
        self._ratio = specs['ratio_table_gantry_rotation']

    def permission_map(self):
        nnodes_dim = tuple(len(n) for n in self._nodes)
        if self._condition_point is None:
            return np.ones(nnodes_dim, dtype=bool)

        return self._permission_map_from_condition_point()

    def _permission_map_from_condition_point(self):
        """Returns a two-dimensional permission map according to a conditioning
        point.
        """
        # Gantry/Table indices in self._nodes and self._condition_point
        IND_GANTRY = 0
        IND_TABLE = 1

        # Initialization
        nodes = [np.asarray(n) for n in self._nodes]
        condition_point = self._condition_point
        ratio = self._ratio

        # Loop in gantry direction through the lattice
        nnodes_dim = tuple(len(n) for n in self._nodes)
        map_vals = np.zeros(nnodes_dim, dtype=bool)
        for inode, node in enumerate(nodes[IND_GANTRY]):
            v_range = abs(node - condition_point[IND_GANTRY]) * ratio
            v_min = condition_point[IND_TABLE] - v_range
            v_max = condition_point[IND_TABLE] + v_range

            ind_true = np.logical_and(nodes[IND_TABLE] >= v_min,
                                      nodes[IND_TABLE] <= v_max)
            map_vals[inode, ind_true] = True
        return map_vals.ravel(order=_NP_ORDER)


class _TrajectoryPermissionGantryDominant3D(_TrajectoryPermission):
    # TODO: Define _TrajectoryPermissionGantryDominant3D
    # TODO: Add tests for _TrajectoryPermissionGantryDominant3D
    pass
