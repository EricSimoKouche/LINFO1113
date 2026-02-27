#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math


def compute_absolute_error(exact, approximation):
    # Helper function for you to compute the absolute error between an
    # exact value and one of its approximations
    return abs(approximation - exact)


def compute_relative_error(exact, approximation):
    # Helper function for you to compute the relative error between an
    # exact value and one of its approximations
    return abs(approximation - exact) / abs(exact)

def factorial(n) :
    if n == 0 : return 1
    else : return n * factorial(n-1)

def fixed_point_scaling(x, max_relative_error):
    # You want to store real numbers using fixed-point representation
    # (yes, you want!). This function must return the minimum scaling
    # factor "k" (i.e., the number of bits after the comma) that is
    # needed to be able to represent the number "x" with less than
    # "max_relative_error" relative error.

    # TODO
    # je dois multiplier par K jusqu'a ce que le
    fixed_x = x
    k = 0
    while compute_relative_error(fixed_x, round(fixed_x)) > max_relative_error:
        fixed_x *= 2
        k+=1

    return k

def sin_term(x, n) :
    return (((-1) ** n) / factorial(2*n + 1)) * (x**(2*n + 1))

def maclaurin_sin(x, k):
    # This function must return the Taylor series of degree "k"
    # for the function "sin(x)" expanded around 0. This expansion
    # around 0 is also known as the Maclaurin series. The function
    # should compute the sum of the first "k" non-zero terms of this
    # series.
    # https://en.wikipedia.org/wiki/Taylor_series#Trigonometric_functions

    # TODO
    res = 0
    for i in range(k) :
        res += sin_term(x, i)
    return res


def exp_term (x, n) :
    return (x**n) / factorial(n)
def maclaurin_exp(x, k):
    # This function must return the Maclaurin series for the function
    # "exp(x)" expanded around 0, using the first "k" non-zero terms
    # of this series.
    # https://en.wikipedia.org/wiki/Taylor_series#Exponential_function

    # TODO
    res = 0
    for i in range(k) :
        res += exp_term(x, i)

    return res

def ln_term(x, n) :
    return ((-1)**(n+1)) * ((x**n) / n)

def maclaurin_ln(x, k):
    # This function must return the Maclaurin series for the function
    # "ln(1 + x)" expanded around 0, using the first "k" non-zero
    # terms of this series.
    # https://en.wikipedia.org/wiki/Taylor_series#Natural_logarithm

    # TODO
    res = 0
    for i in range(1, k+1):
        res += ln_term(x, i)

    return res


def optimize_series_truncation(exact_function, series, points, absolute_error):
    # You are given a function "exact_function(x)" that evaluates some
    # mathematical function at the point "x". You are also given a
    # function "series(x, k)" that evaluates a series approximation of
    # "exact_function(x)" at the point "x" using only the first "k"
    # terms of the series. The function "series(x, k)" may correspond
    # to any of the Maclaurin series defined above.
    #
    # Your function must determine the minimum number of terms needed
    # so that the absolute truncation error of the series
    # approximation at all the points "x" in the given list "points"
    # is smaller than the given tolerance "absolute_error".

    # TODO
    length = len(points)
    for k in range(100) :
        n = 0
        for point in points :
            if compute_absolute_error(exact_function(point), series(point, k)) < absolute_error :
                n += 1
        if n == length : break

    return k


def compute_exact_range(approx_x, number_of_exact_decimals):
    # You are given an approximate value "approx_x" of an exact number
    # "x". You are told that "approx_x" is correct to the given number
    # of decimal places. Return a pair of floating-point numbers
    # representing the minimum and maximum possible values of "x".

    # TODO
    minimum = approx_x - (10**(-number_of_exact_decimals-1))*5
    maximum = approx_x + (10**(-number_of_exact_decimals-1))*5
    return minimum, maximum


def compute_errors_upper_bound(f, df, approx_x, x_error):
    # You are given a function "f(x)" and its derivative "df(x)". Let
    # also "approx_x" be an approximation of an exact value "x" such
    # that "∣approx_x - x∣ <= x_error". What is the upper bound on the
    # propagated absolute and relative errors at "approx_x"?
    #
    # Your function must return a pair of floating-point numbers
    # corresponding to the absolute error and the relative error. The
    # relative error must be expressed in percents (%).

    # TODO
    absolute_error = abs(df(approx_x)) * x_error
    relative_error = (abs(df(approx_x)) / f(approx_x)) * x_error
    return absolute_error, relative_error*100


def bound_absolute_error_propagation(approx_x, approx_y, approx_z, e_x, e_y, e_z):
    # Consider the following function of 3 variables:
    #
    # f(x,y,z) = sin(x) * y^2 + z^3 / y - x + x * y * z
    #
    # This function must return an upper bound on the value of the
    # propagated absolute error, given approximate values of the
    # variables (approx_x, approx_y, approx_z) and their absolute
    # errors (e_x, e_y, e_z). The returned value must be one
    # floating-point number containing the upper bound.

    # TODO
    dfx = math.cos(approx_x) * approx_y ** 2 + approx_y * approx_z - 1
    dfy = 2 * approx_y * math.sin(approx_x) + approx_x * approx_z - ((approx_z ** 3) / (approx_y ** 2))
    dfz = ((3 * (approx_z ** 2)) / approx_y) + approx_x * approx_y

    return abs(dfx) * e_x + abs(dfy) * e_y + abs(dfz) * e_z

