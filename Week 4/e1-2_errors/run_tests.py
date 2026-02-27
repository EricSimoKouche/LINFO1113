#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import student

import sys
for forbidden_module in [ 'numpy', 'sympy', 'ctypes' ]:
    if forbidden_module in sys.modules.keys():
        raise Exception('You are not allowed to import module "%s"' % forbidden_module)

import math
import numpy
import unittest

from grading_toolbox import grade, grade_feedback

class Tests(unittest.TestCase):
    @grade(1)
    def test_fixed_point_scaling(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: Relative error of fixed-point representation"
        self.assertEqual(7, student.fixed_point_scaling(0.1, 0.05))

    @grade(1)
    def test_maclaurin_sin(self):
        self.assertAlmostEqual(0, student.maclaurin_sin(0.7, 0))
        self.assertAlmostEqual(0.7, student.maclaurin_sin(0.7, 1))
        self.assertAlmostEqual(0.6428333333333333, student.maclaurin_sin(0.7, 2))
        self.assertAlmostEqual(0.6442339166666666, student.maclaurin_sin(0.7, 3))
        self.assertAlmostEqual(0.6442175765277778, student.maclaurin_sin(0.7, 4))

        self.assertAlmostEqual(0, student.maclaurin_sin(-0.3, 0))
        self.assertAlmostEqual(-0.3, student.maclaurin_sin(-0.3, 1))
        self.assertAlmostEqual(-0.2955, student.maclaurin_sin(-0.3, 2))
        self.assertAlmostEqual(-0.29552025, student.maclaurin_sin(-0.3, 3))
        self.assertAlmostEqual(-0.2955202066071428, student.maclaurin_sin(-0.3, 4))

    @grade(1)
    def test_maclaurin_exp(self):
        self.assertAlmostEqual(0, student.maclaurin_exp(0.7, 0))
        self.assertAlmostEqual(1, student.maclaurin_exp(0.7, 1))
        self.assertAlmostEqual(1.7, student.maclaurin_exp(0.7, 2))
        self.assertAlmostEqual(1.945, student.maclaurin_exp(0.7, 3))
        self.assertAlmostEqual(2.002166666666667, student.maclaurin_exp(0.7, 4))
        self.assertAlmostEqual(2.012170833333333, student.maclaurin_exp(0.7, 5))

        self.assertAlmostEqual(0, student.maclaurin_exp(-1.3, 0))
        self.assertAlmostEqual(1, student.maclaurin_exp(-1.3, 1))
        self.assertAlmostEqual(-0.3, student.maclaurin_exp(-1.3, 2))
        self.assertAlmostEqual(0.545, student.maclaurin_exp(-1.3, 3))
        self.assertAlmostEqual(0.1788333333333333, student.maclaurin_exp(-1.3, 4))
        self.assertAlmostEqual(0.2978375, student.maclaurin_exp(-1.3, 5))

    @grade(1)
    def test_maclaurin_ln(self):
        self.assertAlmostEqual(0, student.maclaurin_ln(0.7, 0))
        self.assertAlmostEqual(0.7, student.maclaurin_ln(0.7, 1))
        self.assertAlmostEqual(0.455, student.maclaurin_ln(0.7, 2))
        self.assertAlmostEqual(0.5693333333333332, student.maclaurin_ln(0.7, 3))
        self.assertAlmostEqual(0.5093083333333333, student.maclaurin_ln(0.7, 4))
        self.assertAlmostEqual(0.5429223333333333, student.maclaurin_ln(0.7, 5))
        self.assertAlmostEqual(0.5233141666666666, student.maclaurin_ln(0.7, 6))

        self.assertAlmostEqual(0, student.maclaurin_ln(1.3, 0))
        self.assertAlmostEqual(1.3, student.maclaurin_ln(1.3, 1))
        self.assertAlmostEqual(0.455, student.maclaurin_ln(1.3, 2))
        self.assertAlmostEqual(1.187333333333333, student.maclaurin_ln(1.3, 3))
        self.assertAlmostEqual(0.4733083333333332, student.maclaurin_ln(1.3, 4))
        self.assertAlmostEqual(1.215894333333333, student.maclaurin_ln(1.3, 5))
        self.assertAlmostEqual(0.4114261666666666, student.maclaurin_ln(1.3, 6))

    @grade(1)
    def test_optimize_series_truncation(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: Truncation errors for series"
        points = numpy.arange(-numpy.pi, numpy.pi, step=0.01).tolist()
        self.assertEqual(11, student.optimize_series_truncation(math.sin, student.maclaurin_sin, points, 1e-10))

        points = numpy.arange(-1, 1, step=0.01).tolist()
        self.assertEqual(14, student.optimize_series_truncation(math.exp, student.maclaurin_exp, points, 1e-10))

        # "math.log()" returns the natural log (i.e., the "ln(x)" function)
        points = numpy.arange(0, 0.9, step=0.1).tolist()
        self.assertEqual(80, student.optimize_series_truncation(lambda x: math.log(1 + x), student.maclaurin_ln, points, 1e-10))

        # NB: Using the line below leads to the result "1495", which was the expected value in INGInious 2024-2025
        # points = numpy.arange(0, 0.9, step=0.1).tolist() + [ 0.99 ]

    @grade(1)
    def test_compute_exact_range(self):
        # In INGInious 2024-2025, this corresponds to a subpart of "Question 3: Absolute and relative errors (1/4)"
        e = student.compute_exact_range(0.937, 3)
        self.assertAlmostEqual(0.9365, e[0])
        self.assertAlmostEqual(0.9375, e[1])

        # In INGInious 2024-2025, this corresponds to a subpart of "Question 3: Absolute and relative errors (3/4)"
        e = student.compute_exact_range(0.999, 3)
        self.assertAlmostEqual(0.9985, e[0])
        self.assertAlmostEqual(0.9995, e[1])

        e = student.compute_exact_range(-0.237, 6)
        self.assertAlmostEqual(-0.2370005, e[0])
        self.assertAlmostEqual(-0.2369995, e[1])

    @grade(2)
    def test_compute_errors_upper_bound(self):
        # In INGInious 2024-2025, this corresponds to "Question 3: Absolute and relative errors (1/4)"
        e = student.compute_errors_upper_bound(lambda x: x, lambda x: 1, 0.9365, 0.0005)  # Identity function
        self.assertAlmostEqual(0.0005, e[0], places=12)
        self.assertAlmostEqual(0.053390282968499736, e[1], places=7)
        e = student.compute_errors_upper_bound(lambda x: x, lambda x: 1, 0.9375, 0.0005)
        self.assertAlmostEqual(0.0005, e[0], places=12)
        self.assertAlmostEqual(0.05333333333333334, e[1], places=7)

        # In INGInious 2024-2025, this corresponds to "Question 4: Absolute and relative errors (2/4)"
        e = student.compute_errors_upper_bound(lambda x: math.sqrt(1-x), lambda x: -1/(2*math.sqrt(1-x)), 0.9365, 0.0005)
        self.assertAlmostEqual(0.0009920947376656814, e[0], places=12)
        self.assertAlmostEqual(0.39370078740157477, e[1], places=7)
        e = student.compute_errors_upper_bound(lambda x: math.sqrt(1-x), lambda x: -1/(2*math.sqrt(1-x)), 0.9375, 0.0005)
        self.assertAlmostEqual(0.001, e[0], places=12)
        self.assertAlmostEqual(0.4, e[1], places=7)

        # In INGInious 2024-2025, this corresponds to "Question 5: Absolute and relative errors (3/4)"
        e = student.compute_errors_upper_bound(lambda x: math.sqrt(1-x), lambda x: -1/(2*math.sqrt(1-x)), 0.9985, 0.0005)
        self.assertAlmostEqual(0.006454972243679145, e[0], places=12)
        self.assertAlmostEqual(16.666666666667272, e[1], places=7)
        e = student.compute_errors_upper_bound(lambda x: math.sqrt(1-x), lambda x: -1/(2*math.sqrt(1-x)), 0.9995, 0.0005)
        self.assertAlmostEqual(0.011180339887498322, e[0], places=12)
        self.assertAlmostEqual(49.9999999999944, e[1], places=7)

    @grade(2)
    def test_bound_absolute_error_propagation(self):
        # In INGInious 2024-2025, this corresponds to "Question 8: Bounds on error propagation (1/4)"
        self.assertAlmostEqual(25.46492573835946, student.bound_absolute_error_propagation(
            0.3722155379767289, 0.16755531475417695, 0.8475555988229495,
            0.27575971443165814, 0.7232577540143574, 0.7634552494572096))
        self.assertAlmostEqual(9.372734303395738, student.bound_absolute_error_propagation(
            0.19853090814255114, 0.1314940222801284, 0.626009899016065,
            0.03859593033235187, 0.5941479165349165, 0.11293239379734477))
        self.assertAlmostEqual(0.7427831276375904, student.bound_absolute_error_propagation(
            0.06975515572012037, 0.6905368106321245, 0.7295514359464482,
            0.6768203017581806, 0.7597055183033526, 0.09406204204782742))
        self.assertAlmostEqual(0.5953403488632976, student.bound_absolute_error_propagation(
            0.5236169040282306, 0.24355592719442065, 0.09172551400730389,
            0.2942336727566225, 0.8430683795836534, 0.3803830373900636))
        self.assertAlmostEqual(1.3181026679356553, student.bound_absolute_error_propagation(
            0.7540361114611339, 0.755556532950095, 0.5556628846408851,
            0.6463729875887189, 0.16750770137574889, 0.5674418646932039))

if __name__ == '__main__':
    unittest.main()
