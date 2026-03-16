#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy



def lu_decomposition_without_pivoting(a):
    # You are given a square matrix "a". Compute the LU decomposition
    # of "a" using Doolittle's method (where L is imposed to have ones
    # on its diagonal). Your function must return a pair (l, u)
    # providing the L and U parts of the LU decomposition.
    #
    # You can assume that the "a" matrix is "nice": All the pivots are
    # non-zero, there is no need to perform partial pivoting, and none
    # of the pivots are zero.

    assert(len(a.shape) == 2 and
           a.shape[0] == a.shape[1])
    
    # TODO
    l = numpy.eye(a.shape[0], a.shape[1], dtype=float)
    u = numpy.array(a)

    k = 0 # current column
    for i in range(u.shape[0]):
        for j in range(i+1, u.shape[0]) :
            factor = u[j, k] / u[i, k]

            l[j, i] = factor

            u[j] -= factor * u[i]

        k += 1 
    
    return (l, u)


def solve_upper_triangular(u, b):
    # You can copy/paste the same code as in E2-1

    # TODO
    x = numpy.zeros(b.shape, dtype=float)

    for i in range (x.shape[0]-1, -1, -1) :
        for j in range(x.shape[1]-1, -1, -1) :
            x[i, j] = b[i, j]
            for k in range(i+1, x.shape[0]):
                x[i, j] -= u[i, k] * x[k, j]

            x[i, j] /= u[i, i]
    return x


def solve_lower_triangular(l, b):
    # You are given two matrices "l" and "b". "l" is a lower
    # triangular matrix and "b" is a column vector. You must return
    # "y" such that "l * y = b" by forward substitution.
    #
    # You can assume that the "l" matrix is "nice" (see above).

    # TODO
    y = numpy.zeros(b.shape, dtype=float)

    for i in range(y.shape[0]) :
        for j in range(y.shape[1]) :
            y[i, j] = b[i, j]
            for k in range(i):
                y[i, j] -= l[i, k] * y[k, j]
            
            y[i, j] /= l[i, i]

    return y


def lu_solve_without_pivoting(l, u, b):
    # You are given the LU decomposition of a matrix "a" (i.e., "a =
    # l*u"). "b" is a column vector. Compute "x" such that "l*u*x =
    # b".
    #
    # Hint: Combine the functions defined above.

    # TODO
    y = solve_lower_triangular(l, b)
    x = solve_upper_triangular(u, y)
    return x


def cholesky_decomposition(a):
    # You are given a symmetric, positive-definite matrix "a". Compute
    # the Cholesky decomposition of "a". Your function must return the
    # matrix L (i.e., with values in the lower triangle).

    # TODO
    l =  numpy.zeros(a.shape, dtype=float)

    for k in range(l.shape[0]) :

        # First compute the value of l[k, k]
        temp = a[k, k]
        for j in range (k) :
            temp -= math.pow(l[k, j], 2)

        l[k, k] = math.sqrt(temp)

        # Then compute the values of l[i, k] below l[k, k]
        for i in range(k+1, l.shape[0]) :
            l[i, k] = a[i, k]

            for j in range(k) :
                l[i, k] -= l[i, j] * l[k, j]

            l[i, k] /= l[k, k]
         
    return l


def cholesky_solve(a, b):
    # You are given two matrices "a" and "b". "a" is a symmetric,
    # positive-definite matrix and "b" is a column vector. Compute "x"
    # such that "a * x = b" using the Cholesky decomposition of "a".
    #
    # Hint: Combine the functions defined above.

    # TODO
    l = cholesky_decomposition(a)
    y = solve_lower_triangular(l, b)
    x = solve_upper_triangular(l.T, y)
    return x


def lu_decomposition_partial_pivoting(a):
    # You are given a square matrix "a". Compute the PLU decomposition
    # of "a" using Doolittle's method (where L is imposed to have ones
    # on its diagonal) with partial pivoting, so that "a =
    # p*l*u". Your function must return the triple (p, l, u) providing
    # the L and U parts of the LU decomposition, together with square
    # matrix P that stores the row permutations resulting from partial
    # pivoting.

    assert(len(a.shape) == 2 and
           a.shape[0] == a.shape[1])

    # TODO

    def find_largest_absolute_value_and_swap(row, k, p, l, u) :
        
        max_value = abs(u[row, k])
        index = row
        for i in range(row+1, a.shape[0]) :
            if abs(u[i, k]) > max_value :
                max_value = abs(u[i, k])
                index = i
        
        p[:,[row, index]] = p[:,[index, row]]
        u[[row, index]] = u[[index, row]]
        l[[row, index]] = l[[index, row]]
        l[:,[row, index]] = l[:, [index, row]]
        return 

    l = numpy.eye(a.shape[0], a.shape[1], dtype=float)
    p = numpy.eye(a.shape[0], a.shape[1], dtype=float)
    u = numpy.array(a)

    k = 0 # current column
    for i in range(u.shape[0]):
        find_largest_absolute_value_and_swap(i, k, p, l, u)
        for j in range(i+1, a.shape[0]) :
            factor = u[j, k] / u[i, k]

            l[j, i] = factor

            u[j] -= factor * u[i]

        k += 1

    
    return (p, l, u)


def lu_solve_partial_pivoting(p, l, u, b):
    # You are given the PLU decomposition of a matrix "a" (i.e., "a =
    # p*l*u"). "b" is a column vector. Compute "x" such that "p*l*u*x
    # = b".
    #
    # Hints: Combine the functions defined above. The inverse of a
    # permutation matrix is the transpose of this matrix.

    # TODO
    new_b = p @ b
    y = solve_lower_triangular(l, new_b)
    x = solve_upper_triangular(u, y)
    return x