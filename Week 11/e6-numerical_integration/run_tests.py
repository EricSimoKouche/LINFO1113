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

count_eval = 0

def f_sys(x):
    global count_eval
    count_eval += 1
    return numpy.log(3 + numpy.tan(x))

def f_square(x):
    global count_eval
    count_eval += 1
    return x * x

def f_log_tan(x):
    global count_eval
    count_eval += 1
    return numpy.log(1 + numpy.tan(x))

def generate_sampler(xs, ys):
    assert(len(xs.shape) == 1)
    assert(len(ys.shape) == 1)
    assert(xs.shape[0] == ys.shape[0])

    def f(x):
        for i in range(xs.shape[0]):
            if math.isclose(xs[i], x):
                return ys[i]
        raise Exception('The function is not available at this value of x: %f' % x)

    return f


class Tests(unittest.TestCase):
    @grade(1)
    def test_one_panel(self):
        global count_eval

        count_eval = 0
        self.assertAlmostEqual(0.9425907465484042, student.panel_left_hand_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(1, count_eval)

        count_eval = 0
        self.assertAlmostEqual(1.22742248126379, student.panel_right_hand_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(1, count_eval)

        count_eval = 0
        self.assertAlmostEqual(1.0722005035723887, student.panel_midpoint_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(1, count_eval)

        count_eval = 0
        self.assertAlmostEqual(1.085006613906097, student.panel_trapezoidal_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(1.0764692070169584, student.panel_simpson_1_3_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(3, count_eval)

        count_eval = 0
        self.assertAlmostEqual(1.0763339704230435, student.panel_simpson_3_8_rule(f_sys, -0.1, numpy.pi / 4))
        self.assertEqual(4, count_eval)

    @grade(1)
    def test_composite_trapezoidal(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: Trapezoidal rule (1/3)"
        global count_eval

        count_eval = 0
        self.assertAlmostEqual(0.9758205594788009, student.composite_trapezoidal(f_sys, 0, numpy.pi / 4, 1))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.9683835556085423, student.composite_trapezoidal(f_sys, 0, numpy.pi / 4, 5))
        self.assertEqual(6, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.9681280994034811, student.composite_trapezoidal(f_sys, 0, numpy.pi / 4, 10))
        self.assertEqual(11, count_eval)

        count_eval = 0
        self.assertAlmostEqual(32.5, student.composite_trapezoidal(f_square, -2, 3, 1))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(16.875, student.composite_trapezoidal(f_square, -2, 3, 2))
        self.assertEqual(3, count_eval)

        count_eval = 0
        self.assertAlmostEqual(12.96875, student.composite_trapezoidal(f_square, -2, 3, 4))
        self.assertEqual(5, count_eval)

        count_eval = 0
        self.assertAlmostEqual(11.9921875, student.composite_trapezoidal(f_square, -2, 3, 8))
        self.assertEqual(9, count_eval)

        count_eval = 0
        self.assertAlmostEqual(11.748046875, student.composite_trapezoidal(f_square, -2, 3, 16))
        self.assertEqual(17, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.27219826128795027, student.composite_trapezoidal(f_log_tan, 0, numpy.pi / 4, 1))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.27219826128795027, student.composite_trapezoidal(f_log_tan, 0, numpy.pi / 4, 2))
        self.assertEqual(3, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.composite_trapezoidal(f_log_tan, 0, numpy.pi / 4, 4))
        self.assertEqual(5, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.composite_trapezoidal(f_log_tan, 0, numpy.pi / 4, 8))
        self.assertEqual(9, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.composite_trapezoidal(f_log_tan, 0, numpy.pi / 4, 16))
        self.assertEqual(17, count_eval)

    @grade(1)
    def test_recursive_trapezoidal(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: Trapezoidal rule (2/3)"
        # and "Question 3: Trapezoidal rule (3/3)"
        global count_eval

        # The total number of evaluations of "f(x)" must be equal between
        # "student.composite_trapezoidal()" and "student.recursive_trapezoidal()"
        count_eval = 0
        i1 = student.recursive_trapezoidal_step(f_square, -2, 3, None, 1)
        self.assertAlmostEqual(32.5, i1)
        self.assertEqual(2, count_eval)
        i2 = student.recursive_trapezoidal_step(f_square, -2, 3, i1, 2)
        self.assertAlmostEqual(16.875, i2)
        self.assertEqual(3, count_eval)
        i3 = student.recursive_trapezoidal_step(f_square, -2, 3, i2, 3)
        self.assertAlmostEqual(12.96875, i3)
        self.assertEqual(5, count_eval)
        i4 = student.recursive_trapezoidal_step(f_square, -2, 3, i3, 4)
        self.assertAlmostEqual(11.9921875, i4)
        self.assertEqual(9, count_eval)
        i5 = student.recursive_trapezoidal_step(f_square, -2, 3, i4, 5)
        self.assertAlmostEqual(11.748046875, i5)
        self.assertEqual(17, count_eval)

        count_eval = 0
        self.assertAlmostEqual(32.5, student.recursive_trapezoidal(f_square, -2, 3, 1))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(16.875, student.recursive_trapezoidal(f_square, -2, 3, 2))
        self.assertEqual(3, count_eval)

        count_eval = 0
        self.assertAlmostEqual(12.96875, student.recursive_trapezoidal(f_square, -2, 3, 3))
        self.assertEqual(5, count_eval)

        count_eval = 0
        self.assertAlmostEqual(11.9921875, student.recursive_trapezoidal(f_square, -2, 3, 4))
        self.assertEqual(9, count_eval)

        count_eval = 0
        self.assertAlmostEqual(11.748046875, student.recursive_trapezoidal(f_square, -2, 3, 5))
        self.assertEqual(17, count_eval)

        count_eval = 0
        i1 = student.recursive_trapezoidal_step(f_log_tan, 0, numpy.pi / 4, None, 1)
        self.assertAlmostEqual(0.27219826128795027, i1)
        self.assertEqual(2, count_eval)
        i2 = student.recursive_trapezoidal_step(f_log_tan, 0, numpy.pi / 4, i1, 2)
        self.assertAlmostEqual(0.27219826128795027, i2)
        self.assertEqual(3, count_eval)
        i3 = student.recursive_trapezoidal_step(f_log_tan, 0, numpy.pi / 4, i2, 3)
        self.assertAlmostEqual(0.2721982612879502, i3)
        self.assertEqual(5, count_eval)
        i4 = student.recursive_trapezoidal_step(f_log_tan, 0, numpy.pi / 4, i3, 4)
        self.assertAlmostEqual(0.2721982612879502, i4)
        self.assertEqual(9, count_eval)
        i5 = student.recursive_trapezoidal_step(f_log_tan, 0, numpy.pi / 4, i4, 5)
        self.assertAlmostEqual(0.2721982612879502, i5)
        self.assertEqual(17, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.27219826128795027, student.recursive_trapezoidal(f_log_tan, 0, numpy.pi / 4, 1))
        self.assertEqual(2, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.27219826128795027, student.recursive_trapezoidal(f_log_tan, 0, numpy.pi / 4, 2))
        self.assertEqual(3, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.recursive_trapezoidal(f_log_tan, 0, numpy.pi / 4, 3))
        self.assertEqual(5, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.recursive_trapezoidal(f_log_tan, 0, numpy.pi / 4, 4))
        self.assertEqual(9, count_eval)

        count_eval = 0
        self.assertAlmostEqual(0.2721982612879502, student.recursive_trapezoidal(f_log_tan, 0, numpy.pi / 4, 5))
        self.assertEqual(17, count_eval)

    @grade(1)
    def test_composite_simpson_1_3(self):
        # In INGInious 2024-2025, this corresponds to "Question 4: Simpson 1/3 rule (1/2)"
        # and "Question 5: Simpson 1/3 rule (2/2)"

        f = lambda x: math.cos(2 / math.cos(x))
        self.assertAlmostEqual(-1.1196854554190074, student.composite_simpson_1_3(f, -1, 1, 1))
        self.assertAlmostEqual(-1.2884087548681498, student.composite_simpson_1_3(f, -1, 1, 2))
        self.assertAlmostEqual(-1.355011028796083, student.composite_simpson_1_3(f, -1, 1, 4))

        f = lambda x: math.cos(2 * (4 / math.cos(x)))
        self.assertAlmostEqual(-1.105367035316216, student.composite_simpson_1_3(f, -1, 1, 3))

    @grade(1)
    def test_romberg_initialize_table(self):
        r = student.romberg_initialize_table(f_sys, -0.1, numpy.pi / 4, 5)
        self.assertTrue(numpy.allclose(r, numpy.array([
            [ student.composite_trapezoidal(f_sys, -0.1, numpy.pi / 4, 1), 0, 0, 0, 0 ],
            [ student.composite_trapezoidal(f_sys, -0.1, numpy.pi / 4, 2), 0, 0, 0, 0 ],
            [ student.composite_trapezoidal(f_sys, -0.1, numpy.pi / 4, 4), 0, 0, 0, 0 ],
            [ student.composite_trapezoidal(f_sys, -0.1, numpy.pi / 4, 8), 0, 0, 0, 0 ],
            [ student.composite_trapezoidal(f_sys, -0.1, numpy.pi / 4, 16), 0, 0, 0, 0] ], dtype=float)))
        self.assertAlmostEqual(1.08500661, r[0,0])
        self.assertAlmostEqual(1.07860356, r[1,0])
        self.assertAlmostEqual(1.07682858, r[2,0])
        self.assertAlmostEqual(1.07636982, r[3,0])
        self.assertAlmostEqual(1.07625408, r[4,0])

        r = student.romberg_initialize_table(f_square, -2, 3, 4)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 32.5, 0, 0, 0 ],
                                                        [ 16.875, 0, 0, 0 ],
                                                        [ 12.96875, 0, 0, 0 ],
                                                        [ 11.9921875, 0, 0, 0] ], dtype=float)))

        r = student.romberg_initialize_table(f_log_tan, 0, numpy.pi / 4, 4)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 0.2721982612879502, 0, 0, 0 ],
                                                        [ 0.2721982612879502, 0, 0, 0 ],
                                                        [ 0.2721982612879502, 0, 0, 0 ],
                                                        [ 0.2721982612879502, 0, 0, 0] ], dtype=float)))

        f = lambda x: math.cos(2 / math.cos(x))
        r = student.romberg_initialize_table(f, -1, 1, 4)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ -1.69446902, 0, 0, 0 ],
                                                        [ -1.26338135, 0, 0, 0 ],
                                                        [ -1.2821519, 0, 0, 0 ],
                                                        [ -1.33679625, 0, 0, 0] ], dtype=float)))

        xs = numpy.array([ 0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi ], dtype=float)
        ys = numpy.array([ 1, 0.3431, 0.25, 0.3431, 1 ], dtype=float)
        r = student.romberg_initialize_table(generate_sampler(xs, ys), 0, math.pi, 3)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 3.14159265, 0, 0 ],
                                                        [ 1.96349541, 0, 0 ],
                                                        [ 1.52068792, 0, 0 ] ], dtype=float)))


    @grade(1)
    def test_romberg_fill_table(self):
        # This example comes from the slides
        r = numpy.array([ [ 0, 0, 0, 0 ],
                          [ 16, 0, 0, 0 ],
                          [ 30, 0, 0, 0 ],
                          [ 39, 0, 0, 0] ], dtype=float)
        r = student.romberg_fill_table(r, 2)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 0, 0, 0, 0 ],
                                                        [ 16, 21.33333333, 0, 0 ],
                                                        [ 30, 34.66666667, 35.55555556, 0 ],
                                                        [ 39, 42, 42.48888889, 42.5989418 ] ])))

        r = numpy.array([ [ 0, 0, 0, 0 ],
                          [ 16, 0, 0, 0 ],
                          [ 30, 0, 0, 0 ],
                          [ 39, 0, 0, 0] ], dtype=float)
        r = student.romberg_fill_table(r, 3.5)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 0, 0, 0, 0 ],
                                                        [ 16, 17.42222222, 0, 0 ],
                                                        [ 30, 31.24444444, 31.33717214, 0 ],
                                                        [ 39, 39.8, 39.85739576, 39.86203321 ] ])))

        r = numpy.array([ [ -1.69446902, 0, 0, 0 ],
                          [ -1.26338135, 0, 0, 0 ],
                          [ -1.2821519, 0, 0, 0 ],
                          [ -1.33679625, 0, 0, 0] ], dtype=float)
        r = student.romberg_fill_table(r, 2)
        self.assertTrue(numpy.allclose(r, numpy.array([ [-1.69446902, 0, 0, 0 ],
                                                        [-1.26338135, -1.11968546, 0, 0 ],
                                                        [-1.2821519, -1.28840875, -1.29965697, 0 ],
                                                        [-1.33679625, -1.35501103, -1.35945118, -1.36040029] ])))

        # Remark: The first (resp. second) column of the Romberg table
        # corresponds to composite trapezoidal (resp. composite Simpson's 1/3)
        f = lambda x: math.cos(2 / math.cos(x))
        self.assertAlmostEqual(r[0,0], student.composite_trapezoidal(f, -1, 1, 1))
        self.assertAlmostEqual(r[1,0], student.composite_trapezoidal(f, -1, 1, 2))
        self.assertAlmostEqual(r[2,0], student.composite_trapezoidal(f, -1, 1, 4))
        self.assertAlmostEqual(r[3,0], student.composite_trapezoidal(f, -1, 1, 8))
        self.assertAlmostEqual(r[1,1], student.composite_simpson_1_3(f, -1, 1, 1))
        self.assertAlmostEqual(r[2,1], student.composite_simpson_1_3(f, -1, 1, 2))
        self.assertAlmostEqual(r[3,1], student.composite_simpson_1_3(f, -1, 1, 4))

        r = numpy.array([ [ 3.14159265, 0, 0 ],
                          [ 1.96349541, 0, 0 ],
                          [ 1.52068792, 0, 0 ] ], dtype=float)
        r = student.romberg_fill_table(r, 2)
        self.assertTrue(numpy.allclose(r, numpy.array([ [ 3.14159265, 0, 0 ],
                                                        [ 1.96349541, 1.57079633, 0 ],
                                                        [ 1.52068792, 1.37308542, 1.3599047 ] ])))

        # Remark: The first (resp. second) column of the Romberg table
        # corresponds to composite trapezoidal (resp. composite Simpson's 1/3)
        xs = numpy.array([ 0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi ], dtype=float)
        ys = numpy.array([ 1, 0.3431, 0.25, 0.3431, 1 ], dtype=float)
        f = generate_sampler(xs, ys)
        self.assertAlmostEqual(r[0,0], student.composite_trapezoidal(f, 0, math.pi, 1))
        self.assertAlmostEqual(r[1,0], student.composite_trapezoidal(f, 0, math.pi, 2))
        self.assertAlmostEqual(r[2,0], student.composite_trapezoidal(f, 0, math.pi, 4))
        self.assertAlmostEqual(r[1,1], student.composite_simpson_1_3(f, 0, math.pi, 1))
        self.assertAlmostEqual(r[2,1], student.composite_simpson_1_3(f, 0, math.pi, 2))

    @grade(1)
    def test_romberg(self):
        # In INGInious 2024-2025, this corresponds to "Question 6:
        # Romberg (1/2)" and "Question 7: Romberg (2/2)"

        def f(x):
            global count_eval
            count_eval += 1
            return math.cos(2 / math.cos(x))

        global count_eval
        count_eval = 0
        r = student.romberg(f, -1, 1, 4)
        self.assertEqual(9, count_eval)  # If you get "19" here, this is because you use the composite trapezoidal rule, not the recursive rule
        self.assertAlmostEqual(-1.3604002947652445, r)

        # In INGInious 2024-2025, this corresponds to "Question 8: Precision"
        xs = numpy.array([ 0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi ], dtype=float)
        ys = numpy.array([ 1, 0.3431, 0.25, 0.3431, 1 ], dtype=float)
        r = student.romberg(generate_sampler(xs, ys), 0, math.pi, 3)
        self.assertAlmostEqual(1.3599047026179178, r)

        f = lambda x: math.cos(2 * (4 / math.cos(x)))
        r = student.romberg(f, -1, 1, 6)
        self.assertAlmostEqual(-0.5883155864674628, r)


if __name__ == '__main__':
    unittest.main()
