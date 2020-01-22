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
NP_SEED = 458967
NP_ORDER = 'F'      # 'Fortran' reordering style (otherwise unit-tests fail)

# Dynamic Trajectory specifications
AMS_TRAJ_SPECS = {
    # Rotation / Permission type
    'type': 'GantryDominant2D',

    # Max ratio between table and gantry rotation angle:
    #   1.0: 3 neighbors (+- 2 table degrees per 2 gantry degrees)
    #   2.0: 5 neighbors (+- 4 table degrees per 2 gantry degrees)
    'ratio_table_gantry_rotation': 2.0,
}
