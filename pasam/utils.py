# -*- coding: utf-8 -*-
"""Definitions of useful tools.

Generic methods
---------------
    - :fun:`is_pathlib_path`: Check whether an object-type is pathlib.Path
"""

# Standard library
import abc
from pathlib import Path
# Third party requirements
# Local imports


def is_pathlib_path(file):
    """
    Check whether file is a pathlib.Path object.
    """
    return isinstance(file, Path)
