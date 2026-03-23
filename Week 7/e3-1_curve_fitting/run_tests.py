#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as nt
import unittest

import student
from student import Point
from grading_toolbox import grade


class Tests(unittest.TestCase):
    @grade(1)
    def test_interpolator_base_property(self):
        # An interpolation MUST pass exactly through all the given data points
        points = [
            Point(0, 1),
            Point(np.pi / 2, 0),
            Point(np.pi, -1),
            Point(3 * np.pi / 2, 0)
        ]

        lagrange = student.Lagrange(points)
        newton = student.Newton(points)
        newton.fit()

        for p in points:
            self.assertAlmostEqual(p.y, lagrange.evaluate(p.x))
            self.assertAlmostEqual(p.y, newton.evaluate(p.x))

    @grade(2)
    def test_interpolator_evaluation(self):
        # Because of Lagrange's interpolation theorem (cf. slide 11), both methods
        # MUST yield the exact same polynomials and therefore the same evaluations.
        # We can test both simultaneously!
        test_cases = [
            {
                "points": [Point(-1, 6), Point(0, 1), Point(2, 3)],
                "checks": [(1, 0), (3, 10)]  # (x_to_evaluate, expected_y)
            },
            # Corresponds to E3.1.1 in 2024-2025
            {
                "points": [Point(-1.2, -5.76), Point(0.3, -5.61), Point(1.1, -3.69)],
                "checks": [(0, -6), (1, -4), (2, 0)]
            },
            {
                "points": [Point(-2, 47), Point(-1, -3), Point(0, -5), Point(1, -1), Point(2, 15)],
                "checks": [(3, 97), (-3, 235), (1.5, 3.25)]
            }
        ]

        for case in test_cases:
            points = case["points"]
            
            lagrange = student.Lagrange(points)
            newton = student.Newton(points)
            newton.fit()

            for x_eval, expected_y in case["checks"]:
                self.assertAlmostEqual(expected_y, lagrange.evaluate(x_eval))
                self.assertAlmostEqual(expected_y, newton.evaluate(x_eval))

    @grade(1)
    def test_newton_coeffs(self):
        points = [
            Point(0, 3),
            Point(1, 6),
            Point(2, 11)
        ]

        newton = student.Newton(points)
        newton.fit()

        nt.assert_almost_equal(newton.coeffs, (3.0, 3.0, 1.0))
    
        # Corresponds to E3.1.5 in 2024-2025
        points = [
            Point(-2, -1),
            Point(1, 2),
            Point(4, 59),
            Point(-1, 4),
            Point(3, 24),
            Point(-4, -53)
        ]

        newton = student.Newton(points)
        newton.fit()

        nt.assert_almost_equal(newton.coeffs, (-1, 1, 3, 1, 0, 0))

    @grade(1)
    def test_runge_phenomenon(self):
        # This test will allow you to experience the Runge
        # phenomenon first-hand!

        # This is the Runge function, as seen in the slides (cf. slide 34)
        f = lambda x: 1 / (25 * x ** 2 + 1)

        # np.linspace(a, b, n) generates n equally-spaced points between a and b
        x_eval = np.linspace(-1, 1, 1000)

        # We first test with a low-degree interpolation (n = 5 points, degree 4)
        x_train = np.linspace(-1, 1, 5)
        points = [student.Point(x, f(x)) for x in x_train]

        lagrange = student.Lagrange(points)
        newton = student.Newton(points)
        newton.fit()

        error_lagrange = lagrange.compute_max_error(f, x_eval)
        error_newton = newton.compute_max_error(f, x_eval)
        
        self.assertAlmostEqual(0.4383497951290457, error_lagrange) # Notice how the error is relatively contained
        self.assertAlmostEqual(0.4383497951290457, error_newton) # Same for Newton's interpolation

        # We then test with a high-degree interpolation (n = 15 points, degree 14)
        # More points = better approximation, right?
        x_train = np.linspace(-1, 1, 15)
        points = [student.Point(x, f(x)) for x in x_train]

        lagrange = student.Lagrange(points)
        newton = student.Newton(points)
        newton.fit()

        error_lagrange = lagrange.compute_max_error(f, x_eval)
        error_newton = newton.compute_max_error(f, x_eval)

        self.assertAlmostEqual(7.192324287742117, error_lagrange) # The error exploded!
        self.assertAlmostEqual(7.192324287742641, error_newton) # Same thing here!

        # Moral of the story: more points is not always better! This is known as overfitting.
        # You'll hear this word plenty during future classes, especially in machine learning :)
    
if __name__ == "__main__":
    unittest.main()
