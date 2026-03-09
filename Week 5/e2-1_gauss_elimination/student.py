#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy


def gauss_elimination_without_pivoting(a, b):
    # You are given two matrices "a" and "b". "a" is a square matrix
    # and "b" is a column vector. Compute the Gauss elimination for
    # those matrices, without pivoting. Your function must return the
    # pair (a, b) obtained after Gauss elimination.
    #
    # You can assume that the "a" matrix is "nice": All the pivots are
    # non-zero, there is no need to perform partial pivoting, and back
    # substitution is smooth (none of the pivots are zero).

    assert(len(a.shape) == 2 and
           len(b.shape) == 2 and
           a.shape[0] == a.shape[1] and
           a.shape[0] == b.shape[0] and
           b.shape[1] == 1)
    
    k = 0 # current column
    for i in range(a.shape[0]):
        for j in range (i+1, a.shape[0]):
            """
                # second_factor = a[j, k] / a[i, k]

                # a[j] *= second_factor
                # b[j] *= second_factor

                # a[j] = (a[j] - a[i] ) / second_factor
                # b[j] = (b[j] - b[i]) / second_factor
            """
            factor = a[j, k] / a[i, k]

            a[j] -= factor * a[i]
            b[j] -= factor * b[i]

        k = k + 1

    return a, b


def solve_upper_triangular(u, b):
    # You are given two matrices "u" and "b". "u" is an upper
    # triangular matrix and "b" is a column vector. You must return
    # "x" such that "u * x = b" by back substitution.
    #
    # You can assume that the "u" matrix is "nice" (see above).

    # TODO
    x = numpy.zeros(b.shape, dtype=float)

    for i in range (x.shape[0]-1, -1, -1) :
        for j in range(x.shape[1]-1, -1, -1) :
            x[i, j] = b[i, j]
            for k in range(i+1, x.shape[0]):
                x[i, j] -= u[i, k] * x[k, j]

            x[i, j] /= u[i, i]
    return x


def gauss_solve_without_pivoting(a, b):
    # You are given two matrices "a" and "b". "a" is a square matrix
    # matrix and "b" is a column vector. Compute "x" such that "a * x
    # = b" using Gauss elimination.
    #
    # You can assume that the "a" matrix is "nice" (see above).
    #
    # Hint: Combine the functions defined above.

    # TODO
    u, c = gauss_elimination_without_pivoting(a, b)
    x = solve_upper_triangular(u, c)
    return x


def gauss_jordan_elimination_without_pivoting(a, b):
    # You are given two matrices "a" and "b". "a" is a square matrix
    # and "b" is a column vector. Compute the Gauss-Jordan elimination
    # for those matrices (i.e., make sure you obtain a matrix with
    # ones on the diagonal), without pivoting. Your function must
    # return the pair (a, b) obtained after Gauss-Jordan elimination.
    #
    # You can assume that the "a" matrix is "nice" (see above).
    #
    # Hint: The output matrix "a" must be the identity matrix and the
    # output "b" vector must contain the solution "x" of the system "a * x = b".

    # TODO
    # First we go from to bottom 
    k = 0 
    for i in range (a.shape[0]) :
        b[i] /= a[i, k]
        a[i] /= a[i, k]
        for j in range(i+1, a.shape[0]):
            b[j] -= a[j, k] * b[i]
            a[j] -= a[j, k] * a[i]

        k += 1

    # And then we go from bottom to top
    k = a.shape[1] - 1
    for i in range(a.shape[1]-1, -1, -1):
        for j in range(i-1, -1, -1) :
            b[j] -= a[j, k] * b[i]
            a[j] -= a[j, k] * a[i]
            
        k -= 1
    
    return a, b


def gauss_jordan_multiple_solve_without_pivoting(a, b):
    # Implement the Gauss-Jordan algorithm to simultaneously solve the
    # system "a * x = b" for multiple "b" vectors, where "a" is a
    # square matrix (of size "n"). You must return the "x" matrix that
    # contains the solutions to these systems. This output matrix is
    # of size "n x m", where "m" is the number of "b" vectors.
    #
    # You can assume that the "a" matrix is "nice" (see above).

    # TODO
    
    return gauss_jordan_elimination_without_pivoting(a, b)[1]


def invert_matrix(a):
    # This function must compute the inverse of the square,
    # non-singular matrix "a".
    #
    # Hint: Use your implementation of the Gauss-Jordan algorithm.

    # TODO
    b = numpy.eye(3, 3)
    return gauss_jordan_elimination_without_pivoting(a, b)[1]


def gauss_solve_partial_pivoting(a, b):
    # Create a new version of gauss_solve_without_pivoting() that can
    # deal with matrices "a" that are not "nice", by implementing
    # partial pivoting. Your function must return the triple (a, b, x),
    # where "a" and "b" is obtained after Gauss elimination, and where
    # "x" is the solution to system "a * x = b".

    # TODO
    def find_non_zero_line_and_swap(row, column, a, b) :
        for j in range(row+1, a.shape[0]):
            if (a[j, column] != 0.0) :
                a[[row, j]] = a[[j, row]]
                b[[row, j]] = b[[j, row]]
                break 
        return
    
    k = 0 # current column
    for i in range(a.shape[0]):
        if a[i, k] == 0.0 :
            find_non_zero_line_and_swap(i, k, a, b)
        for j in range (i+1, a.shape[0]):
            factor = a[j, k] / a[i, k]

            a[j] -= factor * a[i]
            b[j] -= factor * b[i]

        k = k + 1

    x = gauss_solve_without_pivoting(a, b)

    return (a, b, x)

if __name__ == '__main__':
    a = numpy.array([ [ 0, -4, 2 ],
                        [ -4, 0, 2 ],
                        [ 2, -4, 8 ]], dtype=float)
    b = numpy.array([ [ 1 ],
                        [ 1 ],
                        [ 1 ] ], dtype=float)
    print(gauss_solve_partial_pivoting(a, b))