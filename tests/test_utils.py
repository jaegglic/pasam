#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNIT TEST FILE

Unit tests for the utilities in `pasam.utils.py`.
"""

# -------------------------------------------------------------------------
#   Authors: Stefanie Marti and Christoph Jaeggli
#   Institute: Insel Data Science Center, Insel Gruppe AG
#
#   MIT License
#   Copyright (c) 2020 Stefanie Marti, Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Standard library
import unittest
# Third party requirements
import numpy as np
# Local imports
from pasam._paths import PATH_TESTFILES
import pasam.utils as utl


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
        self.assertTrue(utl._isblank('    '))
        self.assertTrue(utl._isblank('  \t  '))
        self.assertTrue(utl._isblank('  \t \n'))
        self.assertTrue(utl._isblank('  \t \n    \t   '))
        self.assertFalse(utl._isblank('  9    '))
        self.assertFalse(utl._isblank('  asdfasf    '))
        self.assertFalse(utl._isblank('  \t \n    .'))
        self.assertFalse(utl._isblank('  \t \n    * '))
        self.assertFalse(utl._isblank('  \t \n    § '))

    def test_utils_readlines_(self):
        lines_all_true = [
            'I   like\n',
            '                                 \n',
            'working\t\t\tUnit\n',
            '  \t           \t\n',
            '   Tests!!\n',
            '\t\t\n',
            'Hopefully \t\t\tthis \n',
            'one\n',
            '\t\n',
            '      *\n',
            '\t\t\t   \n',
            '  will\n',
            '\t'*22 + 'pass!\n',
            '\n',
            '\n',
            '  8\n',
            '  ^\n',
            '\t\t\t\t\t    \t \t\n',
            '\n',
        ]
        lines_nempty_true = [
            'I   like\n',
            'working\t\t\tUnit\n',
            '   Tests!!\n',
            'Hopefully \t\t\tthis \n',
            'one\n',
            '      *\n',
            '  will\n',
            '\t' * 22 + 'pass!\n',
            '  8\n',
            '  ^\n',
        ]
        file = PATH_TESTFILES + 'nonempty_lines.txt'
        lines_all_test = utl.readlines_(file, remove_blank_lines=False)
        lines_nempty_test = utl.readlines_(file, remove_blank_lines=True)

        self.assertEqual(len(lines_all_test), len(lines_all_true))
        self.assertEqual(len(lines_nempty_test), len(lines_nempty_true))
        for ltest, ltrue in zip(lines_all_test, lines_all_true):
            self.assertEqual(ltest, ltrue)
        for ltest, ltrue in zip(lines_nempty_test, lines_nempty_true):
            self.assertEqual(ltest, ltrue)

    def test_utils_readfile_latticemap2D(self):
        nnodes_dim_true = [5, 6]
        nodes_true = [[-1.5, 1.5, 5, 8, 9], [1, 2, 3, 4, 5, 6]]
        map_vals_true = np.asarray([
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]).ravel(order='C')

        file = PATH_TESTFILES + 'latticemap2d_simple.txt'
        nnodes_dim_test, nodes_test, map_vals_test = utl.readfile_latticemap(file)

        self.assertEqual(nnodes_dim_true, nnodes_dim_test)
        self.assertEqual(nodes_true, nodes_test)
        self.assertTrue(np.all(map_vals_true == map_vals_test))

    def test_utils_readfile_latticemap3D(self):
        nnodes_dim_true = [2, 3, 2]
        nodes_true = [[-1.5, 1.5], [5, 8, 9], [-2, 3]]
        map_vals_true = np.asarray([
            [0.5, 0.5],
            [0.8, 0.8],
            [0.1, 0.1],
            [0.6, 0.6],
            [0.9, 0.9],
            [0.2, 0.2],
        ]).ravel(order='C')

        file = PATH_TESTFILES + 'latticemap3d_simple.txt'
        nnodes_dim_test, nodes_test, map_vals_test = utl.readfile_latticemap(file)

        self.assertEqual(nnodes_dim_true, nnodes_dim_test)
        self.assertEqual(nodes_true, nodes_test)
        self.assertTrue(np.all(map_vals_true == map_vals_test))

    def test_utils_isincreasing(self):
        self.assertTrue(utl.isincreasing([1]))
        self.assertTrue(utl.isincreasing([1, 2, 3, 4, 5]))
        self.assertTrue(utl.isincreasing([-0.110001, -0.11000001, -0.11]))
        self.assertTrue(utl.isincreasing([[1, 3], [2, 4]]))
        self.assertTrue(utl.isincreasing([1, 2, 3, 4, 4], strict=False))

        self.assertFalse(utl.isincreasing([-0.11, -0.1100001, -0.110001]))
        self.assertFalse(utl.isincreasing([1, 2, 3, 4, 4]))

    def test_utils_tensor_dot_lists_of_int(self):
        lists = [[1., 2, 3]]
        tensor_dot_test = utl.cartesian_product(*lists, order='F')
        tensor_dot_true = [
            (1,), (2,), (3,),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        tensor_dot_test = utl.cartesian_product(*lists, order='C')
        tensor_dot_true = [
            (1,), (2,), (3,),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        lists = [[1, 2, 3], [4, 5]]
        tensor_dot_test = utl.cartesian_product(*lists, order='F')
        tensor_dot_true = [
            (1, 4), (2, 4), (3, 4),
            (1, 5), (2, 5), (3, 5),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        tensor_dot_test = utl.cartesian_product(*lists, order='C')
        tensor_dot_true = [
            (1, 4), (1, 5),
            (2, 4), (2, 5),
            (3, 4), (3, 5),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        lists = [[1, 2, 3], [4, 5], [6, 7, 8]]
        tensor_dot_test = utl.cartesian_product(*lists, order='F')
        tensor_dot_true = [
            (1, 4, 6), (2, 4, 6), (3, 4, 6),
            (1, 5, 6), (2, 5, 6), (3, 5, 6),
            (1, 4, 7), (2, 4, 7), (3, 4, 7),
            (1, 5, 7), (2, 5, 7), (3, 5, 7),
            (1, 4, 8), (2, 4, 8), (3, 4, 8),
            (1, 5, 8), (2, 5, 8), (3, 5, 8),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        tensor_dot_test = utl.cartesian_product(*lists, order='C')
        tensor_dot_true = [
            (1, 4, 6), (1, 4, 7), (1, 4, 8),
            (1, 5, 6), (1, 5, 7), (1, 5, 8),
            (2, 4, 6), (2, 4, 7), (2, 4, 8),
            (2, 5, 6), (2, 5, 7), (2, 5, 8),
            (3, 4, 6), (3, 4, 7), (3, 4, 8),
            (3, 5, 6), (3, 5, 7), (3, 5, 8),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        lists = [1, 2, 3]
        with self.assertRaises(TypeError):
            utl.cartesian_product(*lists, order='F')

        lists = [np.array([1, 2, 3])]
        tensor_dot_test = utl.cartesian_product(*lists)
        tensor_dot_true = [
            (1,), (2,), (3,),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        lists = [np.array([1, 2, 3]), np.array([4, 5])]
        tensor_dot_test = utl.cartesian_product(*lists)
        tensor_dot_true = [
            (1, 4), (2, 4), (3, 4),
            (1, 5), (2, 5), (3, 5),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

        lists = np.array([(1, 2, 3), (4, 5), (6, 7, 8)])
        tensor_dot_test = utl.cartesian_product(*lists)
        tensor_dot_true = [
            (1, 4, 6), (2, 4, 6), (3, 4, 6),
            (1, 5, 6), (2, 5, 6), (3, 5, 6),
            (1, 4, 7), (2, 4, 7), (3, 4, 7),
            (1, 5, 7), (2, 5, 7), (3, 5, 7),
            (1, 4, 8), (2, 4, 8), (3, 4, 8),
            (1, 5, 8), (2, 5, 8), (3, 5, 8),
        ]
        self.assertEqual(tensor_dot_true, tensor_dot_test)

    def test_utils_conical_opening_indicator(self):
        cntr = -1.5
        dist = 2.75
        fact = 1
        alpha = np.pi / 2
        points = [
            -4.25 * 1.000001,
            -4.25 * 1.001,
            cntr-fact*dist,
            cntr,
            cntr+fact*dist,
            1.25 * 1.001,
            1.25 * 1.000001,
        ]
        ind_true = np.array([True, False, True, True, True, False, True])

        ind_test = utl.conical_opening_indicator(cntr, dist, alpha, points)
        self.assertTrue(all(ind_true == ind_test))

        cntr = 10
        dist = 5
        fact = 0.5
        alpha = 2*np.arctan(fact)
        points = [
            7.5 * 0.999999,
            7.5 * 0.999,
            cntr-fact*dist,
            cntr,
            cntr+fact*dist,
            12.5 * 1.001,
            12.5 * 1.000001,
        ]
        ind_true = np.array([True, False, True, True, True, False, True])

        ind_test = utl.conical_opening_indicator(cntr, dist, alpha, points)
        self.assertTrue(all(ind_true == ind_test))


if __name__ == '__main__':
    unittest.main()
