#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy


def panel_left_hand_rule(f, a, b):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Return the approximate value of the integral of "f(x)" over
    # the interval "[a,b]", as obtained using the left-hand rule.

    return f(a) * (b - a)


def panel_right_hand_rule(f, a, b):
    # Same specification as "panel_left_hand_rule()", but using the
    # right-hand rule.

    return f(b) * (b - a)


def panel_midpoint_rule(f, a, b):
    # Same specification as "panel_left_hand_rule()", but using the
    # midpoint rule.

    return f((a + b) / 2) * (b - a)


def panel_trapezoidal_rule(f, a, b):
    # Same specification as "panel_left_hand_rule()", but using the
    # trapezoidal rule.

    LB, RB = f(a), f(b)
    return (LB + RB) * (b - a) / 2


def panel_simpson_1_3_rule(f, a, b):
    # Same specification as "panel_left_hand_rule()", but using
    # Simpson's 1/3 rule.

    h = (b - a) / 2
    f_a, f_a_h, f_a_2h = f(a), f(a + h), f(a + 2 * h)

    return (h / 3) * (f_a + 4 * f_a_h + f_a_2h)


def panel_simpson_3_8_rule(f, a, b):
    # Same specification as "panel_left_hand_rule()", but using
    # Simpson's 3/8 rule.

    h = (b - a) / 3
    f_a, f_a_h, f_a_2h, f_a_3h = f(a), f(a + h), f(a + 2 * h), f(a + 3 * h)

    return (3 * h / 8) * (f_a + 3 * f_a_h + 3 * f_a_2h + f_a_3h)


def composite_trapezoidal(f, a, b, N):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Return the approximate value of the integral of "f(x)" over
    # the interval "[a,b]", as obtained using the composite
    # trapezoidal rule. The integration domain must be divided into
    # "N" equal panels.

    h = (b - a) / N
    res = 0
    for i in range(N + 1):
        f_a_ih = f(a + i * h)
        res += f_a_ih if i != 0 and i != N else (f_a_ih / 2)

    return res * h


def recursive_trapezoidal_step(f, a, b, previous_integral, k):
    # You are given a function "f(x)" and two bounds "a" and "b". You
    # must implement one step of the recursive composite trapezoidal
    # rule. More precisely, you must compute the approximate value of
    # the integral of "f(x)" over the interval "[a,b]", if dividing
    # the integration domain into "N = 2^(k-1)" equal panels, with k
    # >= 1. To make this computation efficient, the parameter
    # "previous_integral" contains the approximate value of the
    # integral if using "2^(k-2)" equal panels. Note that
    # "previous_integral" is undefined if k = 1, which corresponds to
    # the base case of the recursion.

    if k == 1:
        return panel_trapezoidal_rule(f, a, b)
    else:
        h_k = (b - a) / (2 ** (k - 1))
        term = 0
        for i in range(1, 2 ** (k - 2) + 1):
            term += f(a + (2 * i - 1) * h_k)
        return (previous_integral / 2) + (h_k * term)


def recursive_trapezoidal(f, a, b, K):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Return the approximate value of the integral of "f(x)" over
    # the interval "[a,b]", as obtained using the recursive
    # trapezoidal rule. The integration domain must be divided into
    # "2^(K-1)" equal panels. To this end, you must make successive
    # calls to "recursive_trapezoidal_step()"

    result = recursive_trapezoidal_step(f, a, b, None, 1)
    for k in range(2, K + 1):
        result = recursive_trapezoidal_step(f, a, b, result, k)
    return result


def composite_simpson_1_3(f, a, b, N):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Return the approximate value of the integral of "f(x)" over
    # the interval "[a,b]", as obtained using the composite Simpson's
    # 1/3 rule. The integration domain must be divided into "N" equal
    # panels, and you must use 3 points in each of those panels.
    #
    # Hint: You are encouraged to reuse "panel_simpson_1_3_rule()".

    h = (b - a) / N
    result = 0
    for i in range(N):
        result += panel_simpson_1_3_rule(f, a + i * h, a + ((i + 1) * h))
    return result


def romberg_initialize_table(f, a, b, K):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Initialize the Romberg table for the given value of K
    # (i.e., up to "N = 2^(K-1)" panels). You must return a numpy
    # array of dimension K x K that contains only zeros, except on its
    # first column. Your algorithm must be as efficient as possible,
    # i.e., it must use the recursive trapezoidal rule to avoid
    # multiple evaluations of "f(x)" at the same value of "x".

    R = numpy.zeros(shape=(K, K), dtype=float)

    R[0, 0] = recursive_trapezoidal_step(f, a, b, None, 1)

    for k in range(1, K):

        h = (b-a)/(2**k)

        s = 0
        for i in range(1, 2 ** (k - 1) + 1):
            s += f(a + (2 * i - 1) * h)

        R[k, 0] = 0.5 * R[k-1, 0] + h * s

    return R


def romberg_fill_table(r, alpha):
    # You are given a Romberg table "r" of size K x K, with the first
    # column correctly set. Return a version of the table augmented
    # until the element "r[K - 1, K - 1]". You are given the value of
    # alpha that specifies the ratio between two successive
    # integration steps.

    K = r.shape[0]
    for j in range(1, K):
        for k in range(j, K):
            r_l, r_u_l = r[k, j-1], r[k-1, j-1]
            fac = math.pow(alpha, 2 * j)
            num = fac * r_l - r_u_l
            denum = fac - 1
            r[k, j] = num / denum

    return r


def romberg(f, a, b, K):
    # You are given a function "f(x)" and two bounds "a" and
    # "b". Return the approximate value of the integral of "f(x)" over
    # the interval "[a,b]", as obtained using Romberg's
    # integration. The parameter "K" specifies the largest number of
    # panels (i.e., "N = 2^(k-1)") to consider.
    #
    # Hint: Combine the two functions defined just above. Your
    # function must return the bottom-right element of the Romberg table.

    r_prev = romberg_initialize_table(f, a, b, K)
    r = romberg_fill_table(r_prev, 2)
    return r[-1, -1]
