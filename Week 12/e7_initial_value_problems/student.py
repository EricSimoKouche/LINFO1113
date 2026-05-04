#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def euler(F, x0, y0, xStop, h):
    # Solve the IVP y'(x) = F(x, y), y(x0) = y0 using Euler's method.
    #
    # Euler's method follows from truncating Taylor's theorem at first order:
    # approximate y(x_{i+1}) using only the slope at the current point x_i.
    # Refer to the lecture slides for the resulting recurrence formula.
    #
    # Your function should return two arrays representing the approximated
    # x and y values on [x0, xStop] with step size h.
    #
    # Hint: y0 can be either a scalar or a 1-D numpy array (for systems of ODEs).
    # Use np.atleast_1d() to handle both cases uniformly, then decide at the end
    # whether to return a 1-D or 2-D y array.

    # TODO

    result_x = []
    result_y = []
    yi = y0
    i  = 0 

    while x0 + i * h < xStop + h:
        xi = x0 + i * h 

        result_x.append(xi)
        result_y.append(yi)

        yi = yi + h * F(xi, yi)
        i += 1
        
    return np.array(result_x), np.array(result_y)



def RK2(F, x0, y0, xStop, h):
    # Solve the IVP y'(x) = F(x, y), y(x0) = y0 using the 2nd-order
    # Runge-Kutta method (Modified Euler / Midpoint method).
    #
    # The key idea behind RK2 is to improve on Euler by evaluating F not at
    # the left endpoint x_i, but at the midpoint x_i + h/2. To reach that
    # midpoint, a provisional half-step is used first (the "predictor").
    # This method belongs to the RK2 family with parameters a=0, b=1,
    # alpha=1/2, beta=1/2 — see the lecture for the general family definition.
    #
    # Your function should return x and y arrays, same format as euler().

    # TODO

    result_x = []
    result_y = []
    yi = y0 
    i = 0 

    while (x0 + i * h) < xStop + h:
        xi = x0 + i * h 

        result_x.append(xi)
        result_y.append(yi)

        k0 = F(xi, yi)
        k1 = F(xi + 0.5 * h, yi + 0.5 * h * k0) 

        yi = yi + h * k1

        i += 1

    return np.array(result_x), np.array(result_y)


def heun(F, x0, y0, xStop, h):
    # Solve the IVP y'(x) = F(x, y), y(x0) = y0 using Heun's method.
    #
    # Heun's method is another member of the RK2 family (a=1/2, b=1/2,
    # alpha=1, beta=1). Rather than sampling F at the midpoint like RK2,
    # it uses a predictor-corrector strategy: first predict y_{i+1} with
    # a full Euler step, then correct by averaging the slopes at both
    # endpoints x_i and x_{i+1}. When F does not depend on y, this
    # corresponds to the trapezoidal integration rule.
    #
    # Your function should return x and y arrays, same format as euler().

    # TODO
    result_x = []
    result_y = []
    yi = y0 
    i = 0 

    while (x0 + i * h) < xStop + h:
        xi = x0 + i * h 

        result_x.append(xi)
        result_y.append(yi)

        k0 = F(xi, yi)
        k1 = F(xi + h, yi + h * k0) 

        yi = yi + h * (0.5 * k0 + 0.5 * k1)

        i += 1

    return np.array(result_x), np.array(result_y)



def RK4(F, x0, y0, xStop, h):
    # Solve the IVP y'(x) = F(x, y), y(x0) = y0 using the classical
    # 4th-order Runge-Kutta method (RK4).
    #
    # RK4 achieves high accuracy by combining four evaluations of F per step:
    # one at each endpoint and two at the midpoint, with the midpoint slopes
    # given double weight (echoing Simpson's rule in integration). The exact
    # coefficients come from matching a 4th-order Taylor expansion of y —
    # refer to the lecture for the full formula.
    #
    # Your function should return x and y arrays, same format as euler().

    # TODO
    result_x = []
    result_y = []
    yi = y0 
    i = 0 

    while (x0 + i * h) < xStop + h:
        xi = x0 + i * h 

        result_x.append(xi)
        result_y.append(yi)

        k0 = F(xi, yi)
        k1 = F(xi + 0.5 * h, yi + 0.5 * h * k0)
        k2 = F(xi + 0.5 * h, yi + 0.5 * h * k1)
        k3 = F(xi + h, yi + h * k2) 

        yi = yi + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3)

        i += 1

    return np.array(result_x), np.array(result_y)

