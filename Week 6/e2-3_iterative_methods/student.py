#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy



# Class that implements Jacobi's iterative method
class Jacobi():

    # Constructor of Jacobi's method for the system "a * x = b", where
    # "a" is a square matrix and "b" is a column vector.
    def __init__(self, a, b):
        # TODO
        self.a = a
        self.b = b

    # Function that takes an input vector "x" and that computes the
    # next iterate of Jacobi's method for this vector "x".
    def next(self, x):
        # TODO
        next_x = numpy.zeros(x.shape, dtype=float)
        for i in range (x.shape[0]) :
            next_x[i, 0] = self.b[i, 0]
            for j in range (x.shape[0]) :
                if i == j : continue

                next_x[i, 0] -= self.a[i, j] * x[j, 0]

            next_x[i, 0] /= self.a[i, i]
        return next_x
    
def solve_lower_triangular(l, b):
    # You can copy/paste the same code as in E2-2

    # TODO
    y = numpy.zeros(b.shape, dtype=float)

    for i in range(y.shape[0]) :
        for j in range(y.shape[1]) :
            y[i, j] = b[i, j]
            for k in range(i):
                y[i, j] -= l[i, k] * y[k, j]
            
            y[i, j] /= l[i, i]

    return y


# Class that implements Gauss-Seidel's iterative method
class GaussSeidel():

    # Constructor of Gauss-Seidel's method for the system "a * x = b",
    # where "a" is a square matrix and "b" is a column vector.
    def __init__(self, a, b):
        # TODO
        self.a = a
        self.b = b

    # Function that takes an input vector "x" and that computes the
    # next iterate of Gauss-Seidel's method for this vector "x".
    def next(self, x):
        # TODO
        next_x = numpy.zeros(x.shape, dtype=float)
        for i in range(x.shape[0]) :
            next_x[i, 0] = self.b[i, 0]
            for j in range(0, i) :
                next_x[i, 0] -= self.a[i, j] * next_x[j, 0]
            
            for j in range(i+1, x.shape[0]) :
                next_x[i, 0] -= self.a[i, j] * x[j, 0]
            
            next_x[i, 0] /= self.a[i, i]

        return next_x

if __name__ == "__main__" :
    a = numpy.array([ [ 12, 3, -5 ],
                        [ 1, 5, 3 ],
                        [ 3, 7, 13 ] ], dtype=float)
    b = numpy.array([ [ 1 ],
                    [ 28 ],
                    [ 76 ] ], dtype=float)
    seidel = GaussSeidel(a, b)
    x0 = numpy.array([ [ 1 ],
                    [ 0 ],
                    [ 1 ] ], dtype=float)
    x1 = seidel.next(x0)
    print(x1)
    x2 = seidel.next(x1)
    print(x2)