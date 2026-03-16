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
    def test_lu_decomposition_without_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: LU decomposition (2/3)"
        a = numpy.array([ [ 3, -3, 3 ],
                          [ -3, 5, 1 ],
                          [ 3, 1, 5 ]], dtype=float)
        (l, u) = student.lu_decomposition_without_pivoting(a)
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0 ],
                                                        [ -1, 1, 0 ],
                                                        [ 1, 2, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 3, -3, 3 ],
                                                        [ 0, 2, 4 ],
                                                        [ 0, 0, -6 ]], dtype=float)))

    @grade(1)
    def test_solve_triangular_systems(self):
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

        l = numpy.array([ [ 2, 0, 0 ],
                          [ -1, 3, 0 ],
                          [ 4, 2, 1 ]], dtype=float)
        b = numpy.array([ [ 4 ],
                          [ 5 ],
                          [ 6 ]], dtype=float)
        y = student.solve_lower_triangular(l, b)
        self.assertTrue(numpy.allclose(y, numpy.array([ [ 2 ],
                                                        [ 7/3 ],
                                                        [ -20/3 ]], dtype=float)))

    @grade(1)
    def test_lu_solve_without_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: LU decomposition (1/3)"
        # and "Question 3: LU decomposition (3/3)"
        a = numpy.array([ [ 3, -3, 3 ],
                          [ -3, 5, 1 ],
                          [ 3, 1, 5 ]], dtype=float)
        b = numpy.array([ [ 9 ],
                          [ -7 ],
                          [ 12 ]], dtype=float)
        (l, u) = student.lu_decomposition_without_pivoting(a)
        x = student.lu_solve_without_pivoting(l, u, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 3.5 ],
                                                        [ 2/3 ],
                                                        [ 1/6 ]], dtype=float)))

    @grade(1)
    def test_cholesky_decomposition(self):
        # In INGInious 2024-2025, this corresponds to "Question 5: Cholesky decomposition (2/3)"
        a = numpy.array([ [ 2, -1, 0 ],
                          [ -1, 2, -1 ],
                          [ 0, -1, 2 ]], dtype=float)
        l = student.cholesky_decomposition(a)
        self.assertTrue(numpy.allclose(l, numpy.array([ [ math.sqrt(2), 0, 0 ],
                                                        [ -1/math.sqrt(2), math.sqrt(3/2), 0 ],
                                                        [ 0, -math.sqrt(2/3), math.sqrt(4/3) ]], dtype=float)))

    @grade(1)
    def test_cholesky_solve(self):
        # In INGInious 2024-2025, this corresponds to "Question 4: Cholesky decomposition (1/3)"
        # and "Question 6: Cholesky decomposition (3/3)"
        a = numpy.array([ [ 2, -1, 0 ],
                          [ -1, 2, -1 ],
                          [ 0, -1, 2 ]], dtype=float)
        b = numpy.array([ [ 3 ],
                          [ -1 ],
                          [ 4 ]], dtype=float)
        x = student.cholesky_solve(a, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 2.75 ],
                                                        [ 2.5 ],
                                                        [ 3.25 ]], dtype=float)))

    @grade(1)
    def test_lu_decomposition_partial_pivoting(self):
        # Example from the slides
        a = numpy.array([ [ 0, 1, 2 ],
                          [ 1, 2, 1 ],
                          [ 2, 7, 8 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        self.assertTrue(numpy.allclose(p, numpy.array([ [ 0, 0, 1 ],
                                                        [ 0, 1, 0 ],
                                                        [ 1, 0, 0 ]], dtype=float)))
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0 ],
                                                        [ 0.5, 1, 0 ],
                                                        [ 0, -2/3, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 2, 7, 8 ],
                                                        [ 0, -1.5, -3 ],
                                                        [ 0, 0, 0 ]], dtype=float)))

        # In INGInious 2024-2025, this corresponds to "Question 7: Row pivoting"
        a = numpy.array([ [ 3, -3, 3 ],
                          [ -3, 5, 1 ],
                          [ 3, 1, 5 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        self.assertTrue(numpy.allclose(p, numpy.array([ [ 1, 0, 0 ],
                                                        [ 0, 0, 1 ],
                                                        [ 0, 1, 0 ]], dtype=float)))
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0 ],
                                                        [ 1, 1, 0 ],
                                                        [ -1, 0.5, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 3, -3, 3 ],
                                                        [ 0, 4, 2 ],
                                                        [ 0, 0, 3 ]], dtype=float)))

        a = numpy.array([ [ 0.6, -0.4, 1 ],
                          [ -0.3, 0.2, 0.5 ],
                          [ 0.6, -1, 0.5 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        self.assertTrue(numpy.allclose(p, numpy.array([ [ 1, 0, 0 ],
                                                        [ 0, 0, 1 ],
                                                        [ 0, 1, 0 ]], dtype=float)))
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0 ],
                                                        [ 1, 1, 0 ],
                                                        [ -0.5, 0, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 0.6, -0.4, 1 ],
                                                        [ 0, -0.6, -0.5 ],
                                                        [ 0, 0, 1 ]], dtype=float)))

        # Example from: https://ece.uwaterloo.ca/~ne112/Lecture_materials/pdfs/10.1%20The%20PLU%20decomposition.pdf
        a = numpy.array([ [ -1.5, 2.1, 6.4 ],
                          [ 2, 3.9, 3.1 ],
                          [ 5, 3, 2 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        self.assertTrue(numpy.allclose(p, numpy.array([ [ 0, 1, 0 ],
                                                        [ 0, 0, 1 ],
                                                        [ 1, 0, 0 ]], dtype=float)))
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0 ],
                                                        [ -0.3, 1, 0 ],
                                                        [ 0.4, 0.9, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 5, 3, 2 ],
                                                        [ 0, 3, 7 ],
                                                        [ 0, 0, -4 ]], dtype=float)))

        # Example from: https://ece.uwaterloo.ca/~ne112/Lecture_materials/pdfs/10.1%20The%20PLU%20decomposition.pdf
        a = numpy.array([ [ 3.2, -5, 5.9, 4.1 ],
                          [ -.4, 5.5, 2.2, -2.1 ],
                          [ -1.6, 6.4, 0.2, -3 ],
                          [ 4, -1, 2, 0 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        self.assertTrue(numpy.allclose(p, numpy.array([ [ 0, 0, 1, 0 ],
                                                        [ 0, 0, 0, 1 ],
                                                        [ 0, 1, 0, 0 ],
                                                        [ 1, 0, 0, 0 ]], dtype=float)))
        self.assertTrue(numpy.allclose(l, numpy.array([ [ 1, 0, 0, 0 ],
                                                        [ -0.4, 1, 0, 0 ],
                                                        [ 0.8, -0.7, 1, 0 ],
                                                        [ -0.1, 0.9, 0.3, 1 ]], dtype=float)))
        self.assertTrue(numpy.allclose(u, numpy.array([ [ 4, -1, 2, 0 ],
                                                        [ 0, 6, 1, -3 ],
                                                        [ 0, 0, 5, 2 ],
                                                        [ 0, 0, 0, 0 ]], dtype=float)))

    @grade(1)
    def test_lu_solve_partial_pivoting(self):
        # In INGInious 2024-2025, this corresponds to "Question 7: Row pivoting"
        a = numpy.array([ [ 3, -3, 3 ],
                          [ -3, 5, 1 ],
                          [ 3, 1, 5 ]], dtype=float)
        b = numpy.array([ [ 9 ],
                          [ -7 ],
                          [ 12 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        assert(numpy.allclose(a, numpy.matmul(p, numpy.matmul(l, u))))
        x = student.lu_solve_partial_pivoting(p, l, u, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 3.5 ],
                                                        [ 2/3 ],
                                                        [ 1/6 ]], dtype=float)))

        a = numpy.array([ [ 0.6, -0.4, 1 ],
                          [ -0.3, 0.2, 0.5 ],
                          [ 0.6, -1, 0.5 ]], dtype=float)
        b = numpy.array([ [ 1 ],
                          [ 1 ],
                          [ 1 ]], dtype=float)
        (p, l, u) = student.lu_decomposition_partial_pivoting(a)
        assert(numpy.allclose(a, numpy.matmul(p, numpy.matmul(l, u))))
        x = student.lu_solve_partial_pivoting(p, l, u, b)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ -5/3 ],
                                                        [ -5/4 ],
                                                        [ 3/2 ]], dtype=float)))

if __name__ == '__main__':
    unittest.main()
