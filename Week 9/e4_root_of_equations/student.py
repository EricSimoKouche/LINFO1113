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
    i = 0
    while (a + i * dx) < b - dx :
        f_left = f(a + i * dx)
        f_right = f(a + (i + 1) * dx)

        # Making conditions this way avoid float underflow
        if (f_left > 0) != (f_right > 0) or f_left == 0 or f_right == 0 :
            return (a + i * dx, a + (i + 1) * dx)
        
        i += 1
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
    left, right = a, b

    if f(left) == 0 : return left
    if f(right) == 0 : return right

    while abs(left - right) >= tol :
        mid = (left + right) / 2.0

        f_mid = f(mid)

        if f_mid == 0:
            return mid
        
        elif (f(left) < 0) != (f(mid) < 0) :
            right = mid
        else :
            left = mid
        
    return (left + right) / 2.0


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
    xi = (a + b) / 2.0

    while True :

        df_xi = df(xi)
        if df_xi == 0 :
            return None

        xi1 = xi - (f(xi)/df(xi))
        if xi1 < a or xi1 > b :
            return None
        
        if abs(xi1 - xi) < tol :
            return xi1

        xi = xi1


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
    left, right = a, b
    xi = (left + right) / 2
    
    while abs(right - left) >= tol :

        df_xi = df(xi) 

        if df_xi == 0 :
            use_newton = False
        else :
            xi1 = xi - (f(xi) / df(xi))
            use_newton = (left < xi1 < right)
        
        if use_newton :
            next_x = xi1
        else :
            next_x = (left + right) / 2.0

        if abs(next_x - xi) < tol :
            return next_x
        
        xi = next_x
        f_xi = f(xi) 
        if f_xi == 0 :
            return xi
        
        if (f(left) > 0) != (f_xi > 0) :
            right = xi
        else :
            left = xi

    return xi