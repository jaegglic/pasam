# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :func:`findall_num_in_str`: Extracts all numbers from a string.
    - :func:`readlines_`: Reads txt file.
"""

# Standard library
import re
from pathlib import Path
# Third party requirements
# Local imports


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

