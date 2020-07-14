# -*- coding: utf-8 -*-
"""This module defines some generic package tools.
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
    values `0` and the blocked nodes have value `1` but for the module it is
    more convenient to use `False` for blocked and `True` for permitted nodes.
    """
    # Value intervals for permitted (True) and blocked (False) nodes
    INTVL_TRUE  = (-0.1, 0.1)
    INTVL_FALSE = ( 0.9, 1.1)

    # Indices for reverse values
    vals = np.asarray(vals).ravel(order=NP_ORDER)
    ind_true = np.logical_and(vals > INTVL_TRUE[0], vals < INTVL_TRUE[1])
    ind_false = np.logical_and(vals > INTVL_FALSE[0], vals < INTVL_FALSE[1])

    # Assign new values according to the intervals
    values = np.asarray(vals, dtype=bool)
    values[ind_true] = True
    values[ind_false] = False

    # Check whether all values have been covered
    if np.sum(np.logical_xor(ind_true, ind_false)) != len(values):
        ind_not_unique = np.logical_not(np.logical_xor(ind_true, ind_false))
        ind = np.where(ind_not_unique)[0]
        raise ValueError(msg.err0001(vals[ind]))

    return values


def cartesian_product(*args, order=NP_ORDER):
    """Cartesian product for a set of containers.

    Args:
        args (array_like of array_like): Input arguments to be composed.
        order (str {'C', 'F'}): 'F' for Fortran and 'C' for C-type ordering.

    Returns:
        list of tuples

    Notes:
        This function is equivalent to the function `itertools.product` but in
        addiction the ordering can be chosen ('F' or 'C').

        For an input with two lists::

            args = [[a, b, c], [e, f]]

        it produces a list of tuples such that::

            order='F':
            [(a, e), (b, e), (c, e),
             (a, f), (b, f), (c, f)],

            order='C':
            [(a, e), (a, f),
             (b, e), (b, f),
             (c, e), (c, f)].
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


def within_conical_opening(center, distance, ratio, points):
    """Indicates whether the points are within (``True``) or outside
    (``False``) of a conical opening.

    An example of a conical (symmetric) opening with distributed points::

                      center
                         *                 ---
                        / \                 |
                       /   \                |
                      /     \            distance
                     /       \              |
                    /         \             |
                   /___________\           ---

                  |-------------|        distance * ratio

             *---------*----*--------*   points
             |         |    |        |
          [False,    True, True,   False]    return

    where `distance` is the height of the triangle, the opening is the
    product `distance * ratio` and `points` is a set of real values. For
    the given example above, two out of four points are lying within the
    conical (symmetric) opening.

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

    # Incorporate inaccuracies due to machine precision
    v_min -= NP_ATOL + NP_RTOL*abs(v_min)
    v_max += NP_ATOL + NP_RTOL*abs(v_max)
    points = np.asarray(points)
    ind = np.logical_and(points >= v_min, points <= v_max)

    # NOTE The above method is equivalent to:
    # points = np.asarray(points)
    # ind = np.logical_and(points >= v_min, points <= v_max)
    # ind = np.logical_or(ind, np.isclose(points, v_min,
    #                                     atol=NP_ATOL, rtol=NP_RTOL))
    # ind = np.logical_or(ind, np.isclose(points, v_max,
    #                                     atol=NP_ATOL, rtol=NP_RTOL))

    return ind


def findall_num_in_str(s):
    """Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (`float` or `int`)
    """
    # Looks for patterns similar to -56.37, 0.24, or 189
    pat_num = r'-?[0-9]+\.?[0-9]*'
    nums = re.findall(pat_num, s)
    return [_str2num(n) for n in nums]


def isincreasing(values, strict=True):
    """Checks if a set of values is increasing.

    Args:
        values (array_like, shape=(n,)): Set of values
        strict (bool, optional): Strictly (`strict=True`) or simply
            (`strict=False`) increasing.

    Returns:
        bool
    """
    values = np.asarray(values).ravel(order=NP_ORDER)
    if strict:
        return np.all(values[:-1] < values[1:])
    else:
        return np.all(values[:-1] <= values[1:])


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


# Private Methods
def _isblank(s):
    """Check whether a string only contains whitespace characters.
    """
    pat_blank = r'^\s+$'
    return bool(re.match(pat_blank, s))


def _str2num(s):
    """Generates `int` or `float` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)
