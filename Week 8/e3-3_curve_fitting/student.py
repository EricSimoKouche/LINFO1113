#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class Regression:
    # Base class for regression models.
    # This class handles the common initialization of data points so that
    # child classes don't have to repeat it.

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n = len(x)

    def fit(self):
        # You don't have to change this method, it is merely there to act as
        # an abstract method.
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self):
        # Same thing here!
        raise NotImplementedError("Subclasses must implement predict()")
    
    def compute_rmse(self, d):
        # You are given the number of degrees of freedom "d" lost to the parameters.
        #
        # Before implementing this method, make sure you understand what RMSE is
        # and how it is computed. The following resource is a great starting point:
        # https://statisticsbyjim.com/regression/root-mean-square-error-rmse/
        # Note that in the formula from the link, P corresponds to "d" in our case.
        #
        # You can assume that fit() has already been called on this model.
        # Your function must return the computed RMSE.

        # TODO
        y_predict = self.predict(self.x)

        num = np.sum((self.y - y_predict) ** 2)
        denum = self.n - d
        return np.sqrt(num / denum)

    
    def plot(self):
        # Plot the data points and the regression curve. You can assume that fit() has
        # already been called.
        # 
        # NOTE: This question isn't graded. You can tweak it as you like. Go wild!

        # TODO
        plt.scatter(self.x, self.y, color='blue', label='Data Points')
        x_min, x_max = np.min(self.x), np.max(self.x)
        x_plot = np.linspace(x_min, x_max, 100)
        y_plot = self.predict(x_plot)
        plt.plot(x_plot, y_plot, color='red', label='Regression Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Regression Plot')
        plt.legend()
        plt.show()
        return



class LinearRegression(Regression):
    # Implement ordinary least squares linear regression for a set of 2D data points.
    #
    # The goal is to fit a line of the form:
    #     f(x) = a_0 + a_1 * x

    def fit(self):
        # Compute the linear regression coefficients a_0 and a_1 based on the coordinates
        # of the provided points.
        #
        # Store the resulting computed coefficients in the self.a0 and self.a1 attributes.

        # TODO
        x_bar = np.mean(self.x)
        y_bar = np.mean(self.y)

        num = denum = 0
        for i in range(self.n) :
            num += self.y[i] * (self.x[i] - x_bar)
            denum += self.x[i] * (self.x[i] - x_bar)

        self.a1 = num / denum

        self.a0 = y_bar - (self.a1 * x_bar)


    def predict(self, x):
        # You are given an x-value (or an array of x-values).
        # Evaluate the regression line at "x".
        #
        # You can assume that the coefficients have already been computed by a
        # prior call to fit(). Your function must return the predicted y-value(s).
        #
        # NOTE: No need to check if x is an array or a single value. By using Numpy
        # arrays, you can take advantage of broadcasting!
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # TL;DR: [1, 2, 3] * 2 = [2, 4, 6]. No need for for-loops!

        # TODO
        return self.a1 * x + self.a0



class PolynomialRegression(Regression):
    # Implement polynomial regression for a set of 2D data points.
    #
    # The goal is to fit a polynomial of a given degree m:
    #     f(x) = c_m * x^m + c_{m - 1} * x^{m - 1} + ... + c_1 * x + c_0

    def __init__(self, x, y, degree):
        super().__init__(x, y)
        self.degree = degree
    
    def fit(self):
        # Compute the polynomial regression coefficients based on the data points.
        #
        # To implement this method, you will need to construct the Vandermonde
        # matrix V of shape (n, degree + 1), where each row i is defined as:
        #     [x_i ^ degree, x_i ^ (degree-1), ..., x_i, 1]
        #
        # Once you have V, you can solve the normal equations to find the coefficients a of the polynomial:
        #     (V^T V) a = V^T y
        # 
        # NOTE: Numpy has functions for both steps :)
        #
        # Store the resulting computed coefficients in the self.coeffs attribute.

        # TODO
        
        V = np.vander(self.x, N=(self.degree + 1))

        A = np.matmul(V.T, V)
        b = np.matmul(V.T, self.y)
        
        self.coeffs = np.linalg.solve(A, b)

    
    def predict(self, x):
        # You are given an x-value (or array of x-values).
        # Evaluate the polynomial regression model at "x".
        #
        # You can assume that the coefficients have already been computed by a
        # prior call to fit(). Your function must return the predicted y-value(s).

        # TODO
        res = np.zeros_like(x, dtype=float)
        for i in range(self.degree + 1) :
            res += self.coeffs[i] * (x ** (self.degree - i))
        return res


    
class MultivariateRegression(Regression):
    # Implement multiple linear regression for a set of multi-dimensional data points.
    #
    # The goal is to fit a linear plane/hyperplane:
    #     f(x) = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p

    def __init__(self, X, y):
        # NOTE: X is now a matrix (we still store it as self.x though)!
        super().__init__(X, y)

    def fit(self):
        # Compute the multivariate regression coefficients based on the features
        # of the provided points.
        #
        # You should construct the design matrix and solve the normal equations.
        # Store the resulting computed coefficients in the self.coeffs attribute.

        # TODO

        X_design = np.hstack((np.ones((self.n, 1)), self.x))
        A = np.matmul(X_design.T, X_design)
        b = np.matmul(X_design.T, self.y)


        self.coeffs = np.linalg.solve(A, b)


    def predict(self, X):
        # You are given a matrix of X-values.
        # Evaluate the regression model at "X".
        #
        # You can assume that the coefficients have already been computed by a
        # prior call to fit(). Your function must return the predicted y-value(s).

        # TODO
        X_s = np.asarray(X, dtype=float)
        return np.matmul(np.hstack((np.ones((X_s.shape[0], 1)), X)), self.coeffs)


    def plot(self):
        raise NotImplementedError("Plots are only supported for 2D regressions")
