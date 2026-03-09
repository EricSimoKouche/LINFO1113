#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

def ForbiddenFunction(*args):
    raise Exception('You are supposed to implement the algorithms by yourself :-)')

# Forbidden functions must be done before "import student"
numpy.linalg.solve = ForbiddenFunction
numpy.linalg.inv = ForbiddenFunction


import student

import sys
for forbidden_module in [ 'sympy' ]:
    if forbidden_module in sys.modules.keys():
        raise Exception('You are not allowed to import module "%s"' % forbidden_module)

import math
import unittest

from grading_toolbox import grade, grade_feedback

class Tests(unittest.TestCase):
    @grade(1)
    def test_gauss_elimination_without_pivoting(self):
        a = numpy.array([ [8, -4, 2 ],
                          [-4, 8, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        b = numpy.array([ [ 11  ],
                          [ -16 ],
                          [ 17  ] ], dtype=float)
        (c, d) = student.gauss_elimination_without_pivoting(a, b)
        self.assertTrue(numpy.allclose(c, numpy.array([ [ 8, -4, 2 ],
                                                        [ 0, 6, 3 ],
                                                        [ 0, 0, 9 ]], dtype=float)))
        self.assertTrue(numpy.allclose(d, numpy.array([ [ 11 ],
                                                        [ -10.5 ],
                                                        [ 9 ]], dtype=float)))
        
    @grade(1)
    def test_solve_upper_triangular(self):
        u = numpy.array([ [ 8, -4, 2 ],
                          [ 0, 6, 3 ],
                          [ 0, 0, 9 ]], dtype=float)
        b = numpy.array([ [ 11 ],
                          [ -10.5 ],
                          [ 9 ]], dtype=float)
        x = student.solve_upper_triangular(u, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 0 ],
                                                        [ -2.25 ],
                                                        [ 1 ]], dtype=float)))
        
    @grade(1)
    def test_gauss_solve_without_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: Gauss elimination"
        a = numpy.array([ [8, -4, 2 ],
                          [-4, 8, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        b = numpy.array([ [ 11  ],
                          [ -16 ],
                          [ 17  ] ], dtype=float)
        x = student.gauss_solve_without_pivoting(a, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 0 ],
                                                        [ -2.25 ],
                                                        [ 1 ]], dtype=float)))

    @grade(1)
    def test_gauss_jordan_elimination_without_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: Gauss-Jordan elimination (1/2)"
        # and "Question 3: Gauss-Jordan elimination (2/2)"
        a = numpy.array([ [8, -4, 2 ],
                          [-4, 8, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        b = numpy.array([ [ 11  ],
                          [ -16 ],
                          [ 17  ] ], dtype=float)
        (c, x) = student.gauss_jordan_elimination_without_pivoting(a, b)
        self.assertTrue(numpy.allclose(c, numpy.array([ [1, 0, 0],
                                                        [0, 1, 0],
                                                        [0, 0, 1]], dtype=float)))
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 0 ],
                                                        [ -2.25 ],
                                                        [ 1 ]], dtype=float)))

    @grade(1)
    def test_gauss_jordan_multiple_solve_without_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 4: Multiple vectors"
        a = numpy.array([ [8, -4, 2 ],
                          [-4, 8, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        b1 = numpy.array([ [ 11  ],
                           [ -16 ],
                           [ 17  ] ], dtype=float)
        b2 = numpy.array([ [ 11  ],
                           [ -20 ],
                           [ 17  ] ], dtype=float)
        b3 = numpy.array([ [ 11  ],
                           [ 1 ],
                           [ 1  ] ], dtype=float)
        b = numpy.concatenate([ b1, b2, b3 ], 1)
        x = student.gauss_jordan_multiple_solve_without_pivoting(a, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 0, -0.2222222222222221, 1.833333333333333 ],
                                                        [ -2.25, -2.805555555555555, 1 ],
                                                        [ 1, 0.7777777777777778, 0.1666666666666667 ]], dtype=float)))

    @grade(1)
    def test_invert_matrix(self):
        # In INGInious 2024-2025, this corresponds to "Question 5: Matrix inversion"
        a = numpy.array([ [8, -4, 2 ],
                          [-4, 8, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        inv = student.invert_matrix(a)
        self.assertTrue(numpy.allclose(inv, numpy.array([ [ 0.1666666666666667, 0.05555555555555555, -0.05555555555555555 ],
                                                          [ 0.08333333333333334, 0.1388888888888889, -0.05555555555555555 ],
                                                          [ 0, 0.05555555555555555, 0.1111111111111111 ]], dtype=float)))

    @grade(1)
    def test_gauss_solve_partial_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 6: Partial pivoting"
        a = numpy.array([ [ 0, -4, 2 ],
                          [ -4, 0, 2 ],
                          [ 2, -4, 8 ]], dtype=float)
        b = numpy.array([ [ 1 ],
                          [ 1 ],
                          [ 1 ] ], dtype=float)
        (c, d, x) = student.gauss_solve_partial_pivoting(a, b)
        self.assertTrue(numpy.allclose(c, numpy.array([ [ -4, 0, 2 ],
                                                        [ 0, -4, 2 ],
                                                        [ 0, 0, 7 ]], dtype=float)))
        self.assertTrue(numpy.allclose(d, numpy.array([ [ 1 ],
                                                        [ 1 ],
                                                        [ 0.5 ]], dtype=float)))
        self.assertTrue(numpy.allclose(x, numpy.array([ [ -0.2142857142857143 ],
                                                        [ -0.2142857142857143 ],
                                                        [ 0.07142857142857142 ]], dtype=float)))

if __name__ == '__main__':
    unittest.main()
