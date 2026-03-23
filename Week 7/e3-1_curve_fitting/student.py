#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class Point():
    # This class is a simple container for 2D coordinates (x, y)
    
    def __init__(self, x, y):
        # Initialize the point with an x and y-coordinate
        self.x = x
        self.y = y


class Interpolator():
    # Base class for polynomial interpolation methods.
    # This class handles the common initialization of data points so that
    # child classes don't have to repeat it.

    def __init__(self, points):
        # NOTE: You may add any additional state attributes you need for your algorithms :)
        self.points = points
        self.n = len(self.points)


    def compute_max_error(self, f, x_eval):
        # You are given a true mathematical function "f" and an array of
        # testing coordinates "x_eval". These are not instances of Point, only an
        # array of values!
        #
        # Calculate the maximum absolute error between the true function's output
        # and the interpolator's approximation across all points in "x_eval". As a
        # reminder, the absolute error is computed as e = abs(x_true - x_approx).
        #
        # You can assume that fit() has already been called on this interpolator (i.e.
        # no need to worry about computing the coefficients for Newton, you can safely
        # call evaluate(x) directly). Your function must return the maximum error found.

        # TODO
        max_error = 0
        for x in x_eval :
            temp = abs(self.evaluate(x) - f(x))
            if temp > max_error :
                max_error = temp

        return max_error


    def evaluate(self, x):
        # You don't have to change this method, it is merely there to act as
        # an abstract method (cf. LEPL1402)
        raise NotImplementedError("Subclasses must implement evaluate(x)")


class Lagrange(Interpolator):
    # Implement Lagrange interpolation for a set of 2D data points.
    #
    # The goal is to build the interpolation polynomial using the Lagrange basis:
    #     p_n(x) = sum(y_i * l_i(x)) for i = 0 to n - 1
    # 
    # where the basis polynomials l_i(x) are defined as:
    #     l_i(x) = prod((x - x_j) / (x_i - x_j)) for j != i

    def evaluate(self, x):
        # You are given an x-value. Evaluate the interpolating polynomial at "x".
        # 
        # Your function must return the evaluated y-value.

        # TODO
        y = 0
        for i in range(self.n) :
            p_i = self.points[i].y
            for j in range(self.n) :
                if j == i : continue
                p_i *= (x - self.points[j].x) / (self.points[i].x - self.points[j].x)

            y += p_i
        return y


class Newton(Interpolator):
    # Implement Newton interpolation for a set of 2D data points.
    #
    # The goal is to build the interpolation polynomial in Newton form:
    #     p_n(x) = a_0 + a_1(x - x_0) + a_2(x - x_0)(x - x_1) + ... + a_n(x - x_0)...(x - x_{n - 1})

    def fit(self):
        # Compute the Newton polynomial coefficients based on the coordinates
        # of the provided points.
        # 
        # Store the resulting computed coefficients in the self.coeffs attribute.
        # self.coeffs[0] should yield a_0, self.coeffs[1] a_1, ...

        # TODO
        n = self.n
        self.coeffs = []
        k = 1
        table = np.zeros((n, n+1))
        
        for i in range(n) :
            table[i][0] = self.points[i].x
            table[i][1] = self.points[i].y 

        k = 1
        self.coeffs.append(table[0][1])    
        for j in range(2, n+1) :
            for i in range(k, n) :
                div = (table[i][j - 1] - table[k - 1][j - 1]) / (table[i][0] - table[k - 1][0])   
                table[i][j] = div

            self.coeffs.append(table[j-1][j])
            k += 1

    def evaluate(self, x):
        # You are given an x-value. Evaluate the interpolating polynomial at "x".
        # 
        # You can assume that the coefficients have already been computed by a
        # prior call to fit(). Your function must return the evaluated y-value.

        # TODO
        n = self.n 
        res = 0
        for i in range (1, n + 1) :
            res = self.coeffs[n - i] + (x - self.points[n - i].x) * res

        return res
