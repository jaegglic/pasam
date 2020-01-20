# -*- coding: utf-8 -*-
"""Problem specific settings file.
"""

# __repr__ handling of long arrays
RLIB_MAXLIST = 3

# 'Fortran' reordering style (DON'T CHANGE IT --> Unit-Tests will fail)
NP_ORDER = 'F'

# Dynamic Trajectory specifications
AMS_TRAJ_SPECS = {
    # Rotation / Permission type
    'type': 'GantryDominant',

    # Dimension
    'ndim': 2,

    # Max ratio between table and gantry rotation angle:
    #   1.0: 3 neighbors (+- 2 table degrees per 2 gantry degrees)
    #   2.0: 5 neighbors (+- 4 table degrees per 2 gantry degrees)
    'ratio_table_gantry_rotation': 2.0,
}
