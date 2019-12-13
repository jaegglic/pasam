# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :fun:`findall_num_in_str`: Extracts all numbers from a string
    - :fun:`nonempty_lines_from_txt`: Returns all non-empty lines from .txt
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


def read_nempty_lines(file):
    """ Provides all non-empty files of a text file.

    Args:
        file (str or pathlib.Path): File or filename.

    Returns:
        list: All non empty lines of text file
    """
    if isinstance(file, Path):
        file = str(file)

    with open(file, 'r') as txtfile:
        lines = txtfile.readlines()
    return [line for line in lines if not _is_blank(line)]


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

