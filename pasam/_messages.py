# -*- coding: utf-8 -*-
"""Collection of error and warning messages.
"""

# -------------------------------- 0XXX utils ---------------------------------
# Warnings


# Errors
def err0000(type_):
    return f"ERR0000 " \
           f"No :class:`_TrajectoryPermission` implementation for " \
           f"trajectory type='{type_}'"


def err0001(vals):
    return f"ERR0001 " \
           f"Values {vals} are not uniquely identified to be either `True` " \
           f"or `False`"


def err0002(order):
    return f"ERR0002 " \
           f"Unknown cartesian product ordering '{order}'"


# ------------------------------- 1XXX lattice --------------------------------
# Warnings


# Errors
def err1001(nodes):
    return f"ERR1001 " \
           f"Nodes {nodes} are not strictly increasing"


def err1002(op):
    return f"ERR1002 " \
           f"Unsupported operation `{op}` for different lattice objects"


def err1003(nnodes, nval):
    return f"ERR1003 " \
           f"Mismatching lattice (nnodes = {nnodes}) and " \
           f"map values (nval={nval})"


def err1004(obj, key):
    return f"ERR1004 " \
           f"Key '{key}' is not an appropriate slicing object " \
           f"for '{obj}'."


def err1005(ndim, ncomp):
    return f"ERR1005 " \
           f"Point dimension (={ncomp}) does not match lattice dimension " \
           f"(={ndim})."


# ------------------------------ 2XXX trajectory ------------------------------
# Warnings


# Errors


# ------------------------------ 3XXX sampling --------------------------------
# Warnings
warn3000 = "WARN3000 The sum of the sliced map values is equal to 0."


# Errors
def err3000(lat1, lat2):
    return f'ERR3000 {lat1} and {lat2} do not match.'


def err3001(file, lattice):
    return f"ERR3001 " \
           f"Inconsistent lattice in file '{file}' comparing to {lattice}"


def err3002(shape, dim):
    return f'ERR3002 ' \
           f'Shape of linear indices array {shape} ' \
           f'does not match dimensions {dim}.'


err3003 = f'ERR3003 ' \
          f'The ratio is too small; consider adapting the ratio or ' \
          f'spacing of the grid nodes (c.f. docstring of `_check_ratio`).'
