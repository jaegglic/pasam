# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :fun:`findall_num_in_str`: Extracts all numbers from a string
"""

# Standard library
import re
# Third party requirements
# Local imports


def findall_num_in_str(s):
    """ Extracts all numbers in a string.

    Args:
        s (str): Input string containing numbers

    Returns:
        list: List of numbers (``float`` or ``int``)
    """
    pat = r'[0-9]+\.?[0-9]*'
    nums = re.findall(pat, s)
    return [_str2num(n) for n in nums]


def _str2num(s):
    """ Generates ``int`` or ``float`` from a string.
    """
    try:
        return int(s)
    except ValueError:
        return float(s)

