# -*- coding: utf-8 -*-
"""Collection of error and warning messages.
"""

# ------------------------------- 0XXX utils.py -------------------------------
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


# ------------------------------ 1XXX lattice.py ------------------------------
# Warnings


# Errors
def err1000(file, lattice):
    return f"ERR1000 " \
           f"Inconsistent lattice in file '{file}' comparing to {lattice}"


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


# ------------------------------ 2XXX pathgen.py ------------------------------
# Warnings


# Errors


# ----------------------------- 3XXX sampling.py ------------------------------
# Warnings
warn3000 = "WARNING 3000 The sum of the sliced map values is equal to 0."


# Errors
def err3000(type_):
    return f"ERR3000 " \
           f"No :class:`Sampler` implementation for " \
           f"trajectory type='{type_}'"


if __name__ == '__main__':
    print(err0000('GantryDominant'))
    print(err1001([1, 2, 4, 3]))
