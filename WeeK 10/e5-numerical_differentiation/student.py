#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy


def first_derivative_by_first_central_difference(x, xs, ys):
    # Using the first central difference approximation, implement a
    # function that computes the first derivative at a point "x=x_i",
    # where "x_i" is one of the sample points in the array "xs". The
    # sampled function values are given by "ys". You may assume that
    # "xs" is sorted in increasing order and evenly spaced (with a
    # constant step "h"), and that "x_i" is not the first or last
    # element of "xs".

    h = xs[1] - xs[0]
    f_x_L = 0
    f_x_R = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x_L = ys[i - 1]
            f_x_R = ys[i + 1]

    return (f_x_R - f_x_L) / (2 * h)


def first_derivative_by_first_forward_difference(x, xs, ys):
    # Same specificication as above, but using the first forward
    # difference approximation.

    h = xs[1] - xs[0]
    f_x_h = 0
    f_x = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_h = ys[i + 1]

    return (f_x_h - f_x) / h


def first_derivative_by_first_backward_difference(x, xs, ys):
    # Same specificication as above, but using the first forward
    # difference approximation.

    h = xs[1] - xs[0]
    f_x_h = 0
    f_x = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_h = ys[i - 1]

    return (f_x - f_x_h) / h


def first_derivative_by_second_forward_difference(x, xs, ys):
    # Same specificication as above, but using the second forward
    # difference approximation.

    h = xs[1] - xs[0]
    f_x = 0
    f_x_h = 0
    f_x_2h = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_h = ys[i + 1]
            f_x_2h = ys[i + 2]

    return (-3 * f_x + 4 * f_x_h - f_x_2h) / (2 * h)


def first_derivative_by_second_backward_difference(x, xs, ys):
    # Same specificication as above, but using the second backward
    # difference approximation.

    h = xs[1] - xs[0]
    f_x = 0
    f_x_h = 0
    f_x_2h = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_h = ys[i - 1]
            f_x_2h = ys[i - 2]

    return (f_x_2h - 4 * f_x_h + 3 * f_x) / (2 * h)


def second_derivative_by_first_central_difference(x, xs, ys):
    # Same specificication as above, but compute the *second*
    # derivative using the first central difference approximation.

    h = xs[1] - xs[0]
    f_x_l = 0
    f_x_r = 0
    f_x = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_l = ys[i - 1]
            f_x_r = ys[i + 1]

    return (f_x_l - 2 * f_x + f_x_r) / (h ** 2)


def second_derivative_by_second_central_difference(x, xs, ys):
    # Same specificication as above, but compute the *second*
    # derivative using the *second* central difference approximation.
    # Note that this formula is not part of the slides, you'll have to
    # derive it by yourself.
    #
    # Hint: Adapt the linear system we used to compute the first
    # central difference approximation for f'''(x), or derive it by
    # hand using Taylor expansion with N=3. The resulting expression
    # must have 5 terms, instead of the 3 terms in
    # "second_derivative_by_first_central_difference().

    A = numpy.array([[1, 1, 1, 1, 1],
                     [2, 1, 0, -1, -2],
                     [2, 1/2, 0, 1/2, 2],
                     [4/3, 1/6, 0, -1/6, -4/3],
                     [2/3, 1/24, 0, 1/24, 2/3]], dtype=float)
    b = numpy.array([0, 0, 1, 0, 0])

    coeffs = numpy.linalg.solve(A, b)
    h = xs[1] - xs[0]
    f_x_2l = 0
    f_x_2r = 0
    f_x_l = 0
    f_x_r = 0
    f_x = 0
    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_l = ys[i - 1]
            f_x_r = ys[i + 1]
            f_x_2l = ys[i - 2]
            f_x_2r = ys[i + 2]

    num = (coeffs[0] * f_x_2r + coeffs[1] * f_x_r + coeffs[2] * f_x
           + coeffs[3] * f_x_l + coeffs[4] * f_x_2l)
    denum = h ** 2
    return num / denum


def first_derivative_by_third_central_difference(x, xs, ys):
    # Same specificication as above, but compute the *first*
    # derivative using the *third* central difference approximation.
    # Note that this formula is not part of the slides, you'll have to
    # derive it by yourself.
    #
    # Hint: Adapt the linear system we used to compute the first
    # central difference approximation for f'''(x), or derive it by
    # hand using Taylor expansion with N=3. The resulting expression
    # must have 5 terms, instead of the 3 terms in
    # "first_derivative_by_second_central_difference().

    A = numpy.array([[1, 1, 1, 1, 1],
                     [2, 1, 0, -1, -2],
                     [2, 1/2, 0, 1/2, 2],
                     [4/3, 1/6, 0, -1/6, -4/3],
                     [2/3, 1/24, 0, 1/24, 2/3]], dtype=float)

    b = numpy.array([0, 1, 0, 0, 0], dtype=float)

    coeffs = numpy.linalg.solve(A, b)
    h = xs[1] - xs[0]
    f_x, f_x_l, f_x_2l, f_x_r, f_x_2r = 0, 0, 0, 0, 0

    for i in range(len(xs)):
        if xs[i] == x:
            f_x = ys[i]
            f_x_l = ys[i - 1]
            f_x_2l = ys[i - 2]
            f_x_r = ys[i + 1]
            f_x_2r = ys[i + 2]

    num = (coeffs[0] * f_x_2r + coeffs[1] * f_x_r + coeffs[2] * f_x
           + coeffs[3] * f_x_l + coeffs[4] * f_x_2l)
    denum = h
    return num / denum


def derivatives_using_parabola(x, x1, y1, x2, y2, x3, y3):
    # You are given three non-equidistant sample points (x1,y1),
    # (x2,y2), and (x3,y3) of a function "f(x)". Fit a quadratic
    # polynomial (parabola) through these points and use it to
    # estimate the first and second derivatives at the given point
    # "x". Your function must return a pair of values, the first
    # containing f'(x) and the second containing f''(x).

    A = numpy.vander([x1, x2, x3], N=3)
    b = numpy.array([y1, y2, y3])

    coeffs = numpy.linalg.solve(A, b)

    df_x = 2 * coeffs[0] * x + coeffs[1]
    ddf_x = 2 * coeffs[0]
    return (df_x, ddf_x)
