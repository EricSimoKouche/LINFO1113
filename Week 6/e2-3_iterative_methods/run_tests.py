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
    def test_jacobi(self):
        a = numpy.array([ [ 10, -1, 2, 0 ],
                          [ -1, 11, -1, 3 ],
                          [ 2, -1, 10, -1 ],
                          [ 0, 3, -1, 8 ]], dtype=float)
        b = numpy.array([ [ 6 ],
                          [ 25 ],
                          [ -11 ],
                          [ 15 ]], dtype=float)
        jacobi = student.Jacobi(a, b)
        x0 = numpy.zeros((4,1), dtype=float)
        x1 = jacobi.next(x0)
        self.assertTrue(numpy.allclose(x1, numpy.array([ [ 0.6 ],
                                                         [ 2.27272727 ],
                                                         [ -1.1 ],
                                                         [ 1.875 ]], dtype=float)))

        x2 = jacobi.next(x1)
        self.assertTrue(numpy.allclose(x2, numpy.array([ [ 1.04727273 ],
                                                         [ 1.71590909 ],
                                                         [ -0.80522727 ],
                                                         [ 0.88522727 ]], dtype=float)))

        x = x2
        for i in range(20):
            x = jacobi.next(x)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 1 ],
                                                        [ 2 ],
                                                        [ -1 ],
                                                        [ 1 ]], dtype=float)))

    @grade(1)
    def test_gauss_seidel(self):
        # This is example 2 from:
        # https://math.libretexts.org/Bookshelves/Linear_Algebra/Introduction_to_Matrix_Algebra_(Kaw)/01%3A_Chapters/1.08%3A_Gauss-Seidel_Method
        a = numpy.array([ [ 12, 3, -5 ],
                          [ 1, 5, 3 ],
                          [ 3, 7, 13 ] ], dtype=float)
        b = numpy.array([ [ 1 ],
                          [ 28 ],
                          [ 76 ] ], dtype=float)
        gauss_seidel = student.GaussSeidel(a, b)
        x0 = numpy.array([ [ 1 ],
                           [ 0 ],
                           [ 1 ] ], dtype=float)

        x1 = gauss_seidel.next(x0)
        self.assertTrue(numpy.allclose(x1, numpy.array([ [ 0.5 ],
                                                         [ 4.9 ],
                                                         [ 3.09230769 ] ], dtype=float)))

        x2 = gauss_seidel.next(x1)
        self.assertTrue(numpy.allclose(x2, numpy.array([ [ 0.14679487 ],
                                                         [ 3.71525641 ],
                                                         [ 3.81175542 ] ], dtype=float)))

        x = x2
        for i in range(15):
            x = gauss_seidel.next(x)
        self.assertTrue(numpy.allclose(x, numpy.array([ [ 1 ],
                                                        [ 3 ],
                                                        [ 4 ] ], dtype=float)))

if __name__ == '__main__':
    unittest.main()
