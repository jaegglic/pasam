# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :func:`findall_num_in_str`: Extracts all numbers from a string
    - :func:`nonempty_lines_from_txt`: Returns all non-empty lines from .txt
"""

# Standard library
import re
from pathlib import Path
# Third party requirements
# Local imports


def findall_num_in_str(s):
    """ Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (``float`` or ``int``)
    """
    pat = r'-?[0-9]+\.?[0-9]*'
    nums = re.findall(pat, s)
    return [_str2num(n) for n in nums]


def readlines_(file, remove_blank_lines=False, hint=-1):
    """ Provides all non-empty files of a text file.

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


def _is_blank(s):
    """Check whether a string only contains whitespace characters.
    """
    return bool(re.match(r'^\s+$', s))


def _str2num(s):
    """ Generates ``int`` or ``float`` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)

