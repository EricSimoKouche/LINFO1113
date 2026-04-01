#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt



class point():
    """
        2d points representation.
        You must use this class throughout the following exercises.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class piecewise_cubic_interpolation():
    """
        This class implements the piecewise cubic interpolation. Here is a summary of the classes it contains:

        __init__: # TODO
            @input:
            - points: the list of points the piecewise cubic spline must fit
            - df0: indexes of the points for which the first derivative is set to 0
            note: df0 is valid for the first and/or last points only

        build_linear_system: # TODO
            This function builds the linear system of equations to be solved to get the values of k.
            - self.A: must contain the coefficient matrix after the build_linear_system function has been executed
            - self.b: must contain the constant vector after the build_linear_system function has been executed
        
        solve:
            solves the linear system of equations described above and stores the values of k in self.ki
            note: Try to use a linear system solver that you've implemented in the previous workshops.
        
        interpolate: # TODO
            @input:
            - x: input value

            @output:
            - the image of x through the cubic spline
        
        approx_roots: # TODO
            This function approximates the roots of the cubic spline
            @input:
                - tol: error tolerance regarding the x root value
            
    """
    def __init__(self, points, df0=None):
        self.points = points
        self.df0 = df0

        self.n = len(points) # number of points
        self.h = points[1].x - points[0].x # x interval between each point

        
        self.build_linear_system() # build the linear system (see below)
        self.solve() # solve the system to find ki (see below)
        self.approx_roots() # Approximates the roots of the interpolation
    

    def build_linear_system(self):
        """
        Generates the linear system to be solved to find the k coefficients
        A: coefficient matrix
        b: constant vector
        """
        A = []
        b = []

        for i in range(self.n) :
            temp = []
            for j in range(self.n) :
                if i == 0 :
                    temp.append(1.0 if j == 0 else 0.0)
                elif i == self.n - 1 :
                    temp.append(1.0 if j == self.n - 1 else 0.0)
                else :
                    if j == i - 1 or j == i + 1 :
                        temp.append(1.0)
                    elif j == i :
                        temp.append(4.0)
                    else :
                        temp.append(0.0)

            # Append a copy of the temp list to A, otherwise we will end up with n references to the same list in A
            A.append(temp[::])

            if i == 0 or i == self.n - 1 :
                b.append(0.0)
            else :
                val = 6.0 / (self.h ** 2) * (self.points[i - 1].y - 2.0 * self.points[i].y + self.points[i + 1].y) 
                b.append(val)

        self.A = np.array(A)
        self.b = np.array(b)

    def solve(self):
        """
        Solve the linear system
        """
        self.ki = np.linalg.solve(self.A, self.b)
        
    def interpolate(self, x):
        """
        Find the interpolated image of x
        """
        
        # Find the interval [x_i, x_{i+1}] to which x belongs
        i = 0
        while i < self.n - 1 and x > self.points[i + 1].x :
            i += 1
        fx = 0

        # make sure to solve the system to get the ki values before trying to interpolate
        self.solve() 

        # Compute the image of x through the cubic spline
        if i < self.n - 1 :
            ki = self.ki[i]
            ki1 = self.ki[i+1]
            xi = self.points[i].x
            xi1 = self.points[i + 1].x
            yi = self.points[i].y
            yi1 = self.points[i + 1].y

            fx = - (ki / 6) * (((x - xi1) ** 3) / self.h - (x - xi1) * self.h) + (ki1 / 6) * (((x - xi) ** 3) / self.h - (x - xi) * self.h) - (yi * (x - xi1) - yi1 * (x - xi)) / self.h

        return fx
    

    def approx_roots(self, tol=1e-10):
        """
        Approximates the roots of the cubic spline.
        """
        x_roots = []

        for i in range(self.n - 1) :
            xi = self.points[i].x
            xi1 = self.points[i + 1].x

            # Check if the function changes sign in the interval [xi, xi1]
            if self.interpolate(xi) * self.interpolate(xi1) < 0 :
                # If it does, we can use the bisection method to find the root in this interval
                a = xi
                b = xi1
                while (b - a) / 2 > tol :
                    c = (a + b) / 2
                    if self.interpolate(c) == 0 : # We found an exact root
                        x_roots.append(c)
                        break
                    elif self.interpolate(a) * self.interpolate(c) < 0 : # The root is in [a, c]
                        b = c
                    else : # The root is in [c, b]
                        a = c

                x_roots.append((a + b) / 2)
            elif self.interpolate(xi) == 0 : # xi is a root
                if xi  not in x_roots :
                    x_roots.append(xi)
            elif self.interpolate(xi1) == 0 : # xi1 is a root
                if xi1 not in x_roots :
                    x_roots.append(xi1) 
        
        self.roots = x_roots