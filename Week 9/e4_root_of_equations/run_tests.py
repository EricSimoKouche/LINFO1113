#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy.testing as nt
import unittest

import student
from grading_toolbox import grade

class Tests(unittest.TestCase):
    @grade(1)
    def test_incremental_root_search(self):
        # f(x) = x
        f = lambda x: x
        dx = 0.01

        # There's no native assertTupleAlmostEqual method in unittest, so we use numpy instead.
        # It does the same thing as basic unittest functions, so do not be afraid :)
        nt.assert_almost_equal(student.incremental_root_search(f, -1, 1, dx), (-0.01, 0))
        self.assertIsNone(student.incremental_root_search(f, 1, 2, dx)) # No root exist in this interval, it should return None

        # f(x) = cos(x)
        f = lambda x: math.cos(x)
        nt.assert_almost_equal(student.incremental_root_search(f, 0, 3, dx), (1.57, 1.58))

        # f(x) = x^3 - 1.2x^2 - 8.19x + 13.23
        # This function has a double root at x = 2.1 (it "touches" zero but doesn't cross the x-axis)
        # You can use tools like Desmos or Wolfram Alpha to visualize it!
        # An incremental root search can't detect those, so it should return None
        f = lambda x: x ** 3 - 1.2 * x ** 2 - 8.19 * x + 13.23

        self.assertIsNone(student.incremental_root_search(f, 2, 3, dx))

        # Corresponds to E4.2 in 2024-2025
        # f(x) = x^4 + 2x^3 - 7x^2 + 3
        f = lambda x: x ** 4 + 2 * x ** 3 - 7 * x ** 2 + 3

        nt.assert_almost_equal(student.incremental_root_search(f, -4, 3, dx), (-3.8, -3.79))
        nt.assert_almost_equal(student.incremental_root_search(f, -1, 0, dx), (-0.62, -0.61))
        nt.assert_almost_equal(student.incremental_root_search(f, 0, 1, dx), (0.79, 0.8))
        nt.assert_almost_equal(student.incremental_root_search(f, 1, 2, dx), (1.61, 1.62))

    @grade(1)
    def test_bisection_root_search(self):
        # f(x) = x
        f = lambda x: x
        tol = 1e-8

        self.assertAlmostEqual(0.0, student.bisection_root_search(f, -1, 1.5, tol))

        # f(x) = cos(x)
        f = lambda x: math.cos(x)
        self.assertAlmostEqual(math.pi / 2, student.bisection_root_search(f, 0, 3, tol))

        # Corresponds to E4.4 in 2024-2025
        # f(x) = x^4 + 2x^3 - 7x^2 + 3
        f = lambda x: x ** 4 + 2 * x ** 3 - 7 * x ** 2 + 3

        self.assertAlmostEqual(-3.791287847235798, student.bisection_root_search(f, -4, -3, tol))
        self.assertAlmostEqual(-0.6180339887498948, student.bisection_root_search(f, -1, 0, tol))
        self.assertAlmostEqual(0.7912878474779201, student.bisection_root_search(f, 0, 1, tol))
        self.assertAlmostEqual(1.6180339887498948, student.bisection_root_search(f, 1, 2, tol))

    @grade(1)
    def test_newton_raphson_root_search(self):
        # f(x) = x
        f = lambda x: x
        df = lambda x: 1
        tol = 1e-8

        self.assertAlmostEqual(0.0, student.newton_raphson_root_search(f, df, -1, 1.5, tol))

        # f(x) = cos(x)
        f = lambda x: math.cos(x)
        df = lambda x: -math.sin(x)
        self.assertAlmostEqual(math.pi / 2, student.newton_raphson_root_search(f, df, 0, 3, tol))

        # Corresponds to E4.7 in 2024-2025
        # f(x) = x^3 - 1.2x^2 - 8.19x + 13.23
        # This is the famous double root function. Newton-Raphson should find it without any issue!
        f = lambda x: x ** 3 - 1.2 * x ** 2 - 8.19 * x + 13.23
        df = lambda x: 3 * x ** 2 - 2.4 * x - 8.19

        self.assertAlmostEqual(-3, student.newton_raphson_root_search(f, df, -4, -3, tol))
        self.assertAlmostEqual(2.1, student.newton_raphson_root_search(f, df, 2, 3, tol))

        # Corresponds to E4.9 in 2024-2025
        # f(x) = x^3 - 1.2x^2 - 8x + 15
        f = lambda x: x ** 3 - 1.2 * x ** 2 - 8 * x + 15
        df = lambda x: 3 * x ** 2 - 2.4 * x - 8

        # Newton-Raphson should NOT converge on this function!
        self.assertIsNone(student.newton_raphson_root_search(f, df, -4, 3, tol))

    @grade(1)
    def test_improved_newton_raphson_root_search(self):
        # f(x) = x
        f = lambda x: x
        df = lambda x: 1
        tol = 1e-8

        self.assertAlmostEqual(0.0, student.improved_newton_raphson_root_search(f, df, -1, 1.5, tol))

        # f(x) = cos(x)
        f = lambda x: math.cos(x)
        df = lambda x: -math.sin(x)

        self.assertAlmostEqual(math.pi / 2, student.improved_newton_raphson_root_search(f, df, 0, 3, tol))

        # Corresponds to E4.9 in 2024-2025
        # f(x) = x^3 - 1.2x^2 - 8x + 15
        f = lambda x: x ** 3 - 1.2 * x ** 2 - 8 * x + 15
        df = lambda x: 3 * x ** 2 - 2.4 * x - 8

        # This function should now converge correctly
        self.assertAlmostEqual(-3.045009359872523, student.improved_newton_raphson_root_search(f, df, -4, 3, tol))

if __name__ == "__main__":
    unittest.main()
