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
import re
from pathlib import Path
# Third party requirements
import numpy as np
# Local imports
import pasam._messages as msg
from pasam._settings import NP_ORDER, NP_ATOL, NP_RTOL


# Public methods
def ams_val_map_to_bool_map(vals):
    """Assigns `0` to ``True`` and `1` to ``False``.

    By default, the AMS generates files where the permitted nodes have
    values `0` and the blocked nodes have value `1`.
    """
    # Value intervals for permitted (True) and blocked (False) nodes
    INTERVAL_TRUE  = (-0.1, 0.1)
    INTERVAL_FALSE = ( 0.9, 1.1)

    vals = np.asarray(vals).ravel(order=NP_ORDER)
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


def cartesian_product(*args, order='F'):
    """Cartesian product for a set of containers.

    Args:
        args (array_like of array_like):
        order (str {'C', 'F'}): 'F' for Fortran and 'C' for C-type ordering.

    Returns:
        list of tuples

    Notes:
        This function is equivalent to the function `itertools.product` but in
        addiction let's the user choose the ordering (Fortran or C).

        For an input with two lists::

            args = [[a, b, c], [e, f]]

        it produces a list of tuples such that::

            [(a, e), (b, e), (c, e), (a, f), (b, f), (c, f)] (order='F')
            [(a, e), (a, f), (b, e), (b, f), (c, e), (c, f)] (order='C').
    """
    pools = map(list, args)
    result = [[]]

    if order == 'F':
        for pool in pools:
            result = [x + [y] for y in pool for x in result]
    elif order == 'C':
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
    else:
        raise ValueError(msg.err0002(order))
    return [tuple(prod) for prod in result]


# TODO unit test utl.conical_opening_indicator
def is_within_conical_opening(center, distance, ratio, points):
    """Indicates whether the points are within (True) or outside (False) of the
    conical opening.

    Notes:

        An example of a conical (symmetric) opening::

                      `center`
                         *                  -----
                        | \                     |
                       |   \                    |
                      |     \               `distance`
                     |       \                  |
                    |         \                 |
                   |-----------\            -----

                   |------------|           `distance * ratio`

             *---------*----*--------*      `points`

        where `distance` is the height of the triangle, the opening is the
        product `distance * ratio` and `points` is a set of real values. For
        the given example above, two out of four points are lying within the
        conical (symmetric) opening so that the function would return
        `[False, True, True, False]`.

    Args:
        center (float): Center of the cone.
        distance (float): Distance from the center.
        ratio (float): Ratio of the opening by distance.
        points (ndarray, shape=(n,)): Set of real values.

    Returns:
        array-like, shape=(n,): Indicator for points inside (True) and outside
            (False) of the conical opening.
    """
    # The opening is supposed to by symmetric around the center
    v_range = distance * ratio / 2
    v_min = center - v_range
    v_max = center + v_range

    # Accept numerical inaccuracies; the below method is equivalent to
    # ind = np.logical_or(ind, np.isclose(points, v_min, atol=NP_ATOL, rtol=NP_RTOL))
    # ind = np.logical_or(ind, np.isclose(points, v_max, atol=NP_ATOL, rtol=NP_RTOL))
    v_min -= NP_ATOL + NP_RTOL*abs(v_min)
    v_max += NP_ATOL + NP_RTOL*abs(v_max)

    # Make standard less_equal and bigger_equal test
    points = np.asarray(points)
    ind = np.logical_and(points >= v_min, points <= v_max)

    return ind


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
    vals = np.asarray(vals).ravel(order=NP_ORDER)
    if strict:
        return np.all(vals[:-1] < vals[1:])
    else:
        return np.all(vals[:-1] <= vals[1:])


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


# TODO change `utl.readfile_latticemap` as follows
#   - remove the return of the nnodes_dim (this is in nodes)
#   - make it possible to return the latticemap or the nodes + values
def readfile_latticemap(file):
    """Reads a latticemap file.

    The structure of the latticemap file is as follows::

            ----------------------------------------
            | <nnodes_dim>                         |
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
    # TODO rename map_vals into values
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
