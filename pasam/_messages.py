# -*- coding: utf-8 -*-
"""Collection of error and warning messages.
"""

# ------------------------------- 0XXX utils.py -------------------------------
# Warnings


# Errors
def err0000(ndim, type_):
    return f"ERR0000 " \
           f"No :class:`_TrajectoryPermission` implementation for " \
           f"dim={ndim} and type='{type_}'"


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


err1002 = f"ERR1002 " \
          f"Unsupported operation `+` for different :class:`Lattice` objects"


def err1003(nnodes, nval):
    return f"ERR1003 " \
           f"Mismatching lattice (nnodes = {nnodes}) and " \
           f"map values (nval={nval})"


def err1004(obj, key):
    return f"ERR1004 " \
           f"Key '{key}' is not an appropriate slicing object " \
           f"for '{obj}'."


if __name__ == '__main__':
    print(err0000(2, 'GantryDominant'))
    print(err1001([1, 2, 4, 3]))
