# -*- coding: utf-8 -*-
""" Os-independent path definitons.
"""

# Standard library
import os
import platform
# Third party requirements
# Local imports

# Define file separator
os_name = platform.system()
if os_name == 'Windows':
    file_sep = '\\'
elif os_name == 'Linux':
    file_sep = '/'
elif os_name == 'Darwin':
    file_sep = ':'
else:
    msg = f'Unknown file separator for operating system "{os_name}"'
    raise ValueError(msg)

# Path for the project's root directory
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + file_sep

# Specific path definitions
PATH_PASAM      = f'{PATH_ROOT}pasam{file_sep}'
PATH_TESTS      = f'{PATH_ROOT}tests{file_sep}'
PATH_TESTFILES  = f'{PATH_TESTS}testfiles{file_sep}'
PATH_EXAMPLES   = f'{PATH_ROOT}examples{file_sep}'
