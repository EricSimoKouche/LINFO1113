#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import student

import sys
for forbidden_module in [ 'sympy' ]:
    if forbidden_module in sys.modules.keys():
        raise Exception('You are not allowed to import module "%s"' % forbidden_module)

import math
import unittest

from grading_toolbox import grade, grade_feedback

xs_1 = numpy.array([ 2.36, 2.37, 2.38, 2.39, 2.40 ], dtype=float)
ys_1 = numpy.array([ 0.85866, 0.86289, 0.86710, 0.87129, 0.87680 ], dtype=float)

xs_2 = numpy.array([ 0.0, 0.1, 0.2, 0.3, 0.4 ], dtype=float)
ys_2 = numpy.array([ 0, 0.078348, 0.138910, 0.192916, 0.244981 ], dtype=float)


class Tests(unittest.TestCase):
    @grade(1)
    def test_simple(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: Finite difference approximation (1/2)"
        self.assertAlmostEqual(0.42, student.first_derivative_by_first_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(-0.2, student.second_derivative_by_first_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.423, student.first_derivative_by_first_forward_difference(2.36, xs_1, ys_1))
        self.assertAlmostEqual(0.551, student.first_derivative_by_first_backward_difference(2.40, xs_1, ys_1))
        self.assertAlmostEqual(0.424, student.first_derivative_by_second_forward_difference(2.36, xs_1, ys_1))
        self.assertAlmostEqual(0.617, student.first_derivative_by_second_backward_difference(2.40, xs_1, ys_1))

    @grade(1)
    def test_complete(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: Finite difference approximation (2/2)"
        self.assertAlmostEqual(0.4219999999999904, student.first_derivative_by_first_central_difference(2.37, xs_1, ys_1))
        self.assertAlmostEqual(0.41999999999998844, student.first_derivative_by_first_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.4849999999999909, student.first_derivative_by_first_central_difference(2.39, xs_1, ys_1))

        self.assertAlmostEqual(-0.20000000000130103, student.second_derivative_by_first_central_difference(2.37, xs_1, ys_1))
        self.assertAlmostEqual(-0.19999999999908058, student.second_derivative_by_first_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(13.19999999999927, student.second_derivative_by_first_central_difference(2.39, xs_1, ys_1))

        self.assertAlmostEqual(0.42299999999999693, student.first_derivative_by_first_forward_difference(2.36, xs_1, ys_1))
        self.assertAlmostEqual(0.4209999999999839, student.first_derivative_by_first_forward_difference(2.37, xs_1, ys_1))
        self.assertAlmostEqual(0.41899999999999304, student.first_derivative_by_first_forward_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.5509999999999887, student.first_derivative_by_first_forward_difference(2.39, xs_1, ys_1))

        self.assertAlmostEqual(0.42299999999999693, student.first_derivative_by_first_backward_difference(2.37, xs_1, ys_1))
        self.assertAlmostEqual(0.4209999999999839, student.first_derivative_by_first_backward_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.41899999999999304, student.first_derivative_by_first_backward_difference(2.39, xs_1, ys_1))
        self.assertAlmostEqual(0.5509999999999887, student.first_derivative_by_first_backward_difference(2.40, xs_1, ys_1))

        self.assertAlmostEqual(0.4240000000000035, student.first_derivative_by_second_forward_difference(2.36, xs_1, ys_1))
        self.assertAlmostEqual(0.42199999999998483, student.first_derivative_by_second_forward_difference(2.37, xs_1, ys_1))
        self.assertAlmostEqual(0.35299999999998405, student.first_derivative_by_second_forward_difference(2.38, xs_1, ys_1))

        self.assertAlmostEqual(0.41999999999998844, student.first_derivative_by_second_backward_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.4179999999999976, student.first_derivative_by_second_backward_difference(2.39, xs_1, ys_1))
        self.assertAlmostEqual(0.6169999999999867, student.first_derivative_by_second_backward_difference(2.40, xs_1, ys_1))

        # New tests in 2025-2026
        self.assertAlmostEqual(0.69455, student.first_derivative_by_first_central_difference(0.1, xs_2, ys_2))
        self.assertAlmostEqual(0.57284, student.first_derivative_by_first_central_difference(0.2, xs_2, ys_2))
        self.assertAlmostEqual(0.5303549999999999, student.first_derivative_by_first_central_difference(0.3, xs_2, ys_2))

        self.assertAlmostEqual(-1.7785999999999993, student.second_derivative_by_first_central_difference(0.1, xs_2, ys_2))
        self.assertAlmostEqual(-0.6556000000000005, student.second_derivative_by_first_central_difference(0.2, xs_2, ys_2))
        self.assertAlmostEqual(-0.19410000000000258, student.second_derivative_by_first_central_difference(0.3, xs_2, ys_2))

        self.assertAlmostEqual(0.78348, student.first_derivative_by_first_forward_difference(0, xs_2, ys_2))
        self.assertAlmostEqual(0.60562, student.first_derivative_by_first_forward_difference(0.1, xs_2, ys_2))
        self.assertAlmostEqual(0.54006, student.first_derivative_by_first_forward_difference(0.2, xs_2, ys_2))
        self.assertAlmostEqual(0.52065, student.first_derivative_by_first_forward_difference(0.3, xs_2, ys_2))

        self.assertAlmostEqual(0.78348, student.first_derivative_by_first_backward_difference(0.1, xs_2, ys_2))
        self.assertAlmostEqual(0.60562, student.first_derivative_by_first_backward_difference(0.2, xs_2, ys_2))
        self.assertAlmostEqual(0.54006, student.first_derivative_by_first_backward_difference(0.3, xs_2, ys_2))
        self.assertAlmostEqual(0.52065, student.first_derivative_by_first_backward_difference(0.4, xs_2, ys_2))

        self.assertAlmostEqual(0.8724099999999999, student.first_derivative_by_second_forward_difference(0, xs_2, ys_2))
        self.assertAlmostEqual(0.6383999999999999, student.first_derivative_by_second_forward_difference(0.1, xs_2, ys_2))
        self.assertAlmostEqual(0.5497649999999998, student.first_derivative_by_second_forward_difference(0.2, xs_2, ys_2))

        self.assertAlmostEqual(0.5166900000000002, student.first_derivative_by_second_backward_difference(0.2, xs_2, ys_2))
        self.assertAlmostEqual(0.50728, student.first_derivative_by_second_backward_difference(0.3, xs_2, ys_2))
        self.assertAlmostEqual(0.5109449999999999, student.first_derivative_by_second_backward_difference(0.4, xs_2, ys_2))

    @grade(1)
    def test_second_derivative_by_second_central_difference(self):
        # In INGInious 2024-2025, this corresponds to "Question 3: Richardson's extrapolation"
        self.assertAlmostEqual(-1.3166666666669657, student.second_derivative_by_second_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(-0.600475000000003, student.second_derivative_by_second_central_difference(0.2, xs_2, ys_2))

    @grade(1)
    def test_first_derivative_by_third_central_difference(self):
        # In INGInious 2024-2025, this corresponds to "Question 4: Accuracy"
        self.assertAlmostEqual(0.40883333333332683, student.first_derivative_by_third_central_difference(2.38, xs_1, ys_1))
        self.assertAlmostEqual(0.5596358333333332, student.first_derivative_by_third_central_difference(0.2, xs_2, ys_2))

    @grade(1)
    def test_derivatives_using_parabola(self):
        # In INGInious 2024-2025, this corresponds to "Question 5: Unequal steps"
        d = student.derivatives_using_parabola(1, 0.97, 0.85040, 1, 0.84147, 1.05, 0.82612)
        self.assertAlmostEqual(-0.301166666666669, d[0])
        self.assertAlmostEqual(-0.2333333333331903, d[1])

        d = student.derivatives_using_parabola(1.03, 0.97, 0.85040, 1, 0.84147, 1.05, 0.82612)
        self.assertAlmostEqual(-0.3081666666666647, d[0])
        self.assertAlmostEqual(-0.2333333333331903, d[1])


if __name__ == '__main__':
    unittest.main()
