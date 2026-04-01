#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as nt
import unittest

import student
from grading_toolbox import grade

class Tests(unittest.TestCase):
    @grade(1)
    def test_linear_regression(self):
        # The regression should be the function f(x) = x
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]

        lin = student.LinearRegression(x, y)
        lin.fit()

        self.assertAlmostEqual(lin.a0, 0)
        self.assertAlmostEqual(lin.a1, 1)
        self.assertAlmostEqual(lin.predict(2), 2)
        self.assertAlmostEqual(lin.predict(1113), 1113)

        # Corresponds to E3.3.1 in 2024-2025
        x = [1198, 1715, 2530, 2014, 2136, 1492, 1652, 1168, 1492, 1602, 1192, 2045]
        y = [11.90, 6.80, 5.53, 6.38, 5.53, 8.50, 7.65, 13.60, 9.78, 8.93, 11.90, 6.38]

        lin = student.LinearRegression(x, y)
        lin.fit()

        self.assertAlmostEqual(lin.a0, 18.409940514092238)
        self.assertAlmostEqual(lin.a1, -0.00583313333510115)
        self.assertAlmostEqual(lin.compute_rmse(2), 1.1641716342135628)
        self.assertAlmostEqual(lin.predict(2000), 6.743673843889939)

    @grade(1)
    def test_polynomial_regression(self):
        # f(x) = 1 + 2x + 3x^2
        x = [0, 1, 2, 3, 4, 5]
        y = [1, 6, 17, 34, 57, 86]

        poly = student.PolynomialRegression(x, y, degree=2)
        poly.fit()

        nt.assert_array_almost_equal(poly.coeffs, [3, 2, 1])
        self.assertAlmostEqual(poly.compute_rmse(3), 0)
        self.assertAlmostEqual(poly.predict(10), 321)

        # f(x) = x^3 - 2x^2 + x - 1
        x = [-2, -1, 0, 1, 2]
        y = [-19, -5, -1, -1, 1]

        regression = student.PolynomialRegression(x, y, degree=3)
        regression.fit()

        nt.assert_array_almost_equal(regression.coeffs, [1, -2, 1, -1])
        self.assertAlmostEqual(regression.compute_rmse(4), 0)
        self.assertAlmostEqual(regression.predict(0), -1)

        # Corresponds to E3.3.3 in 2024-2025
        x = [1, 2.5, 3.5, 4, 1.1, 1.8, 2.2, 3.7]
        y = [6.008, 15.722, 27.130, 33.772, 5.257, 9.549, 11.098, 28.828]

        poly = student.PolynomialRegression(x, y, 2)
        poly.fit()

        nt.assert_almost_equal(poly.coeffs, (2.1081182, -1.0688961, 4.4056738))
        self.assertAlmostEqual(poly.compute_rmse(3), 0.81292796105407)
        self.assertAlmostEqual(poly.predict(2), 10.700354369212983)

    @grade(1)
    def test_polynomial_regression_equivalent(self):
        # A polynomial of degree 1 should give the same result as linear regression
        x = [1, 2, 3, 4, 5]
        y = [2.1, 3.9, 6.2, 7.8, 10.1]

        lin = student.LinearRegression(x, y)
        lin.fit()

        poly = student.PolynomialRegression(x, y, degree=1)
        poly.fit()

        self.assertAlmostEqual(lin.a0, poly.coeffs[1])
        self.assertAlmostEqual(lin.a1, poly.coeffs[0])
        self.assertAlmostEqual(lin.predict(3.5), poly.predict(3.5))
        poly.plot()
        lin.plot()

    @grade(1)
    def test_multivariate_regression(self):
        # f(x1, x2) = 5 + x1 - x2
        X = [
            [1, 1],
            [2, 1],
            [1, 2],
            [3, 3],
            [0, 0]
        ]
        y = [5, 6, 4, 5, 5]

        multi = student.MultivariateRegression(X, y)
        multi.fit()
        predictions = multi.predict([[10, 5], [0, 0]])

        nt.assert_array_almost_equal(multi.coeffs, [5, 1, -1])
        nt.assert_array_almost_equal(predictions, [10, 5])

        # f(x1, x2, x3) = 4 - 2 * x1 + 0.5 * x2 + 3 * x3
        X = [
            [1, 2, 3],
            [2, 0, 1],
            [0, 1, 2],
            [3, 3, 0],
            [1, 1, 1],
            [4, 2, 2],
        ]
        y = [12, 3, 10.5, -0.5, 5.5, 3]

        multi = student.MultivariateRegression(X, y)
        multi.fit()
        predictions = multi.predict([[0, 0, 0], [1, 1, 1], [2, 4, 3]])

        nt.assert_array_almost_equal(multi.coeffs, [4, -2, 0.5, 3])
        self.assertAlmostEqual(multi.compute_rmse(4), 0)
        nt.assert_array_almost_equal(predictions, [4, 5.5, 11])

        # Corresponds to E3.3.4 in 2024-2025
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [2, 1],
            [2, 2]
        ])
        y = [1.46, 2.09, 0.78, 0.42, 0.66, 1.05]

        multi = student.MultivariateRegression(X, y)
        multi.fit()

        nt.assert_array_almost_equal(multi.coeffs, [1.536047, -0.631163, 0.415465])
        self.assertAlmostEqual(multi.compute_rmse(3), 0.14809252293359443)

if __name__ == "__main__":
    unittest.main()
