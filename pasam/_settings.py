# -*- coding: utf-8 -*-
"""Problem specific settings file.
"""

# Dimensional indices of physical problem
DIM_GANTRY   = 0
DIM_TABLE    = 1
DIM_COLLI    = 2

# __repr__ handling of long arrays
RLIB_MAXLIST = 3

# numpy settings
NP_SEED = 46784316
NP_ORDER = 'F'      # 'Fortran' reordering style (otherwise unit-tests fail)
NP_ATOL = 1e-8      # Absolute tolerance as in `np.isclose`
NP_RTOL = 1e-5      # Relative tolerance as in `np.isclose`
