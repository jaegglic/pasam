# -*- coding: utf-8 -*-
"""Definitions of some package tools.

Generic methods
---------------
    - :func:`findall_num_in_str`: Extracts all numbers from a string.
    - :func:`isincreasing`: Checks if a sequence of values increases.
    - :func:`permission_map_from_condition_file`: Permission map from file.
    - :func:`permission_map_from_condition_point`: Permission map from point.
    - :func:`readfile_latticemap`: Reads a latticemap file.
    - :func:`readlines_`: Reads txt file (possibility to remove empty lines).
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
import re
from pathlib import Path
# Third party requirements
import numpy as np
# Local imports
import pasam._messages as msg
import pasam._settings as settings

# Constants
_NP_ORDER = settings.NP_ORDER


# Public methods
def findall_num_in_str(s):
    """Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (`float` or `int`)
    """
    re_num = r'-?[0-9]+\.?[0-9]*'
    nums = re.findall(re_num, s)
    return [_str2num(n) for n in nums]


def isincreasing(vals, strict=True):
    """Checks if a set of values is increasing.

    Args:
        vals (array_like, shape=(n,)): Set of values
        strict (bool, optional): Strictly (`strict=True`) or simply
            (`strict=False`) increasing.

    Returns:
        bool
    """
    vals = np.asarray(vals).ravel(order=_NP_ORDER)
    if strict:
        return np.all(vals[:-1] < vals[1:])
    else:
        return np.all(vals[:-1] <= vals[1:])


def permission_array_from_condition_file(file):
    """Reads a permission map from a given .txt file.

    Args:
        file (str or pathlib.Path): File or filename.

    Returns:
        ndarray: Boolean array for permitted (True) and blocked (False) nodes.

    """
    _, _, vals = readfile_latticemap(file)

    # !!! Values are inverted in the ams map !!!
    # There, `0` means permitted and `1` blocked
    map_vals = _ams_val_map_to_bool_map(vals)

    return map_vals


def readlines_(file, remove_blank_lines=False):
    """Reading txt file (similar to builtin ``readlines``).

    In addition to the standard implementation of ``readlines`` for
    ``_io.TextIOWrapper``, this version provides the possibility to remove
    empty text lines.

    Args:
        file (str or pathlib.Path): File or filename.
        remove_blank_lines (bool, optional): Remove blank lines.

    Returns:
        list(str): All non empty lines of text file
    """
    if isinstance(file, Path):
        file = str(file)

    with open(file, 'r') as txtfile:
        lines = txtfile.readlines()

    if remove_blank_lines:
        lines = [line for line in lines if not _isblank(line)]
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
    lines = readlines_(file, remove_blank_lines=True)

    # Number of nodes per dimension (defined in lines[0])
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


def write_trajectory_to_txt(fname, points):
    """Writes the trajectory to a text file.

    Args:
        fname (file, str, or pathlib.Path): File, filename, or generator to
            write.
        points (list): Sequence of trajectory points.

    Returns:
        None
    """
    _ams_write_trajectory_to_txt(fname, points)


# Private Methods
def _ams_val_map_to_bool_map(vals):
    """Assigns `0` to ``True`` and `1` to ``False``.

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
        raise ValueError(msg.err0001(vals[ind]))

    # Tag nodes according to the permission
    map_vals[ind_true] = True
    map_vals[ind_false] = False

    return map_vals


def _ams_write_trajectory_to_txt(fname, points):
    """Write a trajectory to a txt file according to the AMS guidelines.
    """
    with open(fname, 'w+') as tfile:
        tfile.write(f'{len(points)}\n')
        for pt in points:
            tfile.write('\t'.join([f'{p}' for p in pt]) + '\n')


def _isblank(s):
    """Check whether a string only contains whitespace characters.
    """
    re_blank = r'^\s+$'
    return bool(re.match(re_blank, s))


def _str2num(s):
    """Generates `int` or `float` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)
