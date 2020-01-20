# -*- coding: utf-8 -*-
"""Problem specific settings file.
"""

NP_ORDER = 'F'         # 'F' for 'Fortran' and 'C' for 'C++' reordering style

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