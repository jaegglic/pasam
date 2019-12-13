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
from pasam._paths import PATH_TESTFILES


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_utils__str2num(self):
        self.assertEqual(utl._str2num('5'), 5)
        self.assertTrue(isinstance(utl._str2num('5'), int))
        self.assertEqual(utl._str2num('5.5'), 5.5)
        self.assertTrue(isinstance(utl._str2num('5.5'), float))
        with self.assertRaises(ValueError):
            utl._str2num('adf')

    def test_utils_findall_num_in_str(self):
        s = '\n\t  \n\n .sadf  \t -1.1  asdf/*-+  \t  23. 555.8   2\t3.478\n  '
        numbers = utl.findall_num_in_str(s)
        self.assertEqual(numbers, [-1.1, 23, 555.8, 2, 3.478])

    def test_utils__is_blank(self):
        self.assertTrue(utl._is_blank('    '))
        self.assertTrue(utl._is_blank('  \t  '))
        self.assertTrue(utl._is_blank('  \t \n'))
        self.assertTrue(utl._is_blank('  \t \n    \t   '))
        self.assertFalse(utl._is_blank('  9    '))
        self.assertFalse(utl._is_blank('  asdfasf    '))
        self.assertFalse(utl._is_blank('  \t \n    .'))
        self.assertFalse(utl._is_blank('  \t \n    * '))
        self.assertFalse(utl._is_blank('  \t \n    § '))

    def test_utils_read_nempty_lines(self):
        lines_true = [
            'I   like\n',
            'working\t\t\tUnit\n',
            '   Tests!!\n',
            'Hopefully \t\t\tthis \n',
            'one\n',
            '      *\n',
            '  will\n',
            '\t'*22 + 'pass!\n',
            '  8\n',
            '  ^\n',
        ]
        file = PATH_TESTFILES + 'nonempty_lines.txt'
        lines_test = utl.read_nempty_lines(file)
        self.assertEqual(len(lines_test), len(lines_true))
        for ltest, ltrue in zip(lines_test, lines_true):
            self.assertEqual(ltest, ltrue)


if __name__ == '__main__':
    unittest.main()
