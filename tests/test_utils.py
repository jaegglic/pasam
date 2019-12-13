#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the utilities in `pasam.utils.py`.
"""

# -------------------------------------------------------------------------
#   Authors: Stefanie Marti and Christoph Jaeggli
#   Institute: Insel Data Science Center, Insel Gruppe AG
#
#   Copyright (c) 2020 Stefanie Marti, Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Standard library
import unittest
# Third party requirements
# Local imports
import pasam.utils as utl


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_utils_str2num(self):
        self.assertEqual(utl._str2num('5'), 5)
        self.assertTrue(isinstance(utl._str2num('5'), int))
        self.assertEqual(utl._str2num('5.5'), 5.5)
        self.assertTrue(isinstance(utl._str2num('5.5'), float))
        with self.assertRaises(ValueError):
            utl._str2num('adf')

    def test_utils_findall_num_in_str(self):
        s = '\n\t  \n\n .sadf  \t 1.1  asdf/*-+  \t  23. 555.8   2\t3.478\n   '
        numbers = utl.findall_num_in_str(s)
        self.assertEqual(numbers, [1.1, 23, 555.8, 2, 3.478])


if __name__ == '__main__':
    unittest.main()
