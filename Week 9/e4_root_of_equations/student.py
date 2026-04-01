#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy


def incremental_root_search(f, a, b, dx):
    # Implement the incremental search method to find the approximate
    # location of a root of the function "f" over the interval "[a, b]".
    #
    # The algorithm should return the first interval "(x, x + dx)" of 
    # size "dx" that contains a root. If there is no root found in the 
    # domain, it should return None.

    # TODO
    return None


def bisection_root_search(f, a, b, tol):
    # Implement the bisection algorithm to find an approximation of the
    # root of the function "f" over the interval "[a, b]" (knowing that 
    # the root is bracketed within this interval). 
    #
    # The algorithm should terminate and return the estimated root
    # (i.e., the midpoint of the final interval) once the specified
    # tolerance "tol" is reached.

    # TODO
    return None


def newton_raphson_root_search(f, df, a, b, tol):
    # Implement the Newton-Raphson algorithm to find an approximation of
    # the root of the function "f" (with derivative "df"). 
    #
    # Start the search at the midpoint of the interval: x0 = (a + b) / 2. 
    # The algorithm successfully terminates and returns the estimated root when the 
    # step size falls within the specified tolerance "tol".
    #
    # If an iterate falls strictly outside the interval "[a, b]", consider
    # that the algorithm diverges and return None.

    # TODO
    return None


def improved_newton_raphson_root_search(f, df, a, b, tol):
    # Try to make the Newton-Raphson algorithm more robust by combining it 
    # with the bisection algorithm to ensure we never go outside the root 
    # bracket "[a, b]".
    #
    # At each step, attempt to compute the next estimate using the standard
    # Newton-Raphson formula. If the resulting iterate falls strictly outside
    # the current bracket "[a, b]", fall back to a bisection step to
    # guarantee convergence toward the root.
    #
    # Return the estimated root once the tolerance "tol" is met.

    # TODO
    return None
