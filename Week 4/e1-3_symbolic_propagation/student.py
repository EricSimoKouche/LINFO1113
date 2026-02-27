#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math


# Abstract class representing a differentiable mathematical function
# f(x,y,z,...) of multiple variables in a symbolic form. This is
# similar to the theoretical slides, but the input vector is here
# represented as a dictionary mapping strings (the variable names) to
# floating-point values for improved clarity.
class DifferentiableFunction():

    # Evaluate the function at a given point. This point is specified
    # as a dictionary containing one value for each variable. The
    # returned value must be a floating-point number.
    def evaluate(self, variables):
        raise Exception('Not implemented')

    # Compute the symbolic derivative of the function with respect to
    # provided variable. The returned value must be a
    # DifferentiableFunction object.
    def derivative(self, variable):
        raise Exception('Not implemented')


# Mathematical function returning a constant.
# It is defined as "f(...) = value".
class Constant(DifferentiableFunction):
    def __init__(self, value):
        self.value = float(value)

    def evaluate(self, variables):
        # TODO
        return self.value

    def derivative(self, variable):
        # TODO
        return Constant(0.0)


# Mathematical function representing one variable.
# It is defined as "f(...,name,...) = name".
class Symbol(DifferentiableFunction):
    def __init__(self, name):
        assert(isinstance(name, str))
        self.name = name

    def evaluate(self, variables):
        # TODO
        return variables[self.name]

    def derivative(self, variable):
        # TODO
        if self.name == variable :
            return Constant(1.0)
        else : return Constant(0.0)


# Mathematical function representing the sum of two other functions.
# It is defined as "f(...) = left(...) + right(...)".
class Sum(DifferentiableFunction):
    def __init__(self, left, right):
        assert(isinstance(left, DifferentiableFunction))
        assert(isinstance(right, DifferentiableFunction))
        self.left = left
        self.right = right

    def evaluate(self, variables):
        # TODO
        return self.left.evaluate(variables) + self.right.evaluate(variables)

    def derivative(self, variable):
        # TODO
        left_term = self.left.derivative(variable)
        right_term = self.right.derivative(variable)
        return Sum(left_term, right_term)


# Mathematical function representing the product of two other functions.
# It is defined as "f(...) = left(...) * right(...)".
class Product(DifferentiableFunction):
    def __init__(self, left, right):
        assert(isinstance(left, DifferentiableFunction))
        assert(isinstance(right, DifferentiableFunction))
        self.left = left
        self.right = right

    def evaluate(self, variables):
        # TODO
        return self.left.evaluate(variables) * self.right.evaluate(variables)

    def derivative(self, variable):
        # TODO
        left_term = Product(self.left.derivative(variable), self.right)
        right_term = Product(self.left, self.right.derivative(variable))
        return Sum(left_term, right_term)


# Mathematical function representing the quotient of two other functions.
# It is defined as "f(...) = numerator(...) / denominator(...)".
class Quotient(DifferentiableFunction):
    def __init__(self, numerator, denominator):
        assert(isinstance(numerator, DifferentiableFunction))
        assert(isinstance(denominator, DifferentiableFunction))
        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, variables):
        # TODO
        return self.numerator.evaluate(variables) / self.denominator.evaluate(variables)

    def derivative(self, variable):
        # TODO
        left_term_numerator = Product(self.numerator.derivative(variable), self.denominator)
        right_term_numerator = Product(Constant(-1.0), Product(self.numerator, self.denominator.derivative(variable)))
        new_numerator = Sum(left_term_numerator, right_term_numerator)
        new_denominator = Product(self.denominator, self.denominator)
        return Quotient(new_numerator, new_denominator)


# Mathematical function representing the sine of a variable.
# It is defined as "f(...,name,...) = sin(name)".
class Sine(DifferentiableFunction):
    def __init__(self, name):
        assert(isinstance(name, str))
        self.name = name

    def evaluate(self, variables):
        # TODO
        return math.sin(variables[self.name])

    def derivative(self, variable):
        # TODO
        return Cosine(variable) if variable == self.name else Constant(0.0)


# Mathematical function representing the cosine of a variable.
# It is defined as "f(...,name,...) = cos(name)".
class Cosine(DifferentiableFunction):
    def __init__(self, name):
        assert(isinstance(name, str))
        self.name = name

    def evaluate(self, variables):
        # TODO
        return math.cos(variables[self.name])

    def derivative(self, variable):
        # TODO
        return Product(Constant(-1.0), Sine(self.name)) if variable == self.name else Constant(0.0)


# Mathematical function representing the power of another function.
# It is defined as "f(...) = g(...) ^ power", where "power" is a
# floating-point number.
class Power(DifferentiableFunction):
    def __init__(self, f, power):
        assert(isinstance(f, DifferentiableFunction))
        self.f = f
        self.power = float(power)

    def evaluate(self, variables):
        # TODO
        return self.f.evaluate(variables) ** self.power

    def derivative(self, variable):
        # TODO
        kernel_term = Product(Constant(self.power), self.f.derivative(variable))
        return Product(kernel_term, Power(self.f, self.power - 1))


def create_sphere_volume():
    # This function must create a DifferentiableFunction of two
    # variables "pi" and "r" that computes the volume enclosed by a
    # sphere of radius "r". In this exercise, the mathematical
    # constant "pi" is considered as a variable because its value
    # cannot be represented exactly by the computer.
    # https://en.wikipedia.org/wiki/Sphere#Enclosed_volume

    # TODO
    first_term = Constant(4/3)
    second_term = Symbol('pi')
    third_term = Power(Symbol('r'), 3)
    return Product(Product(first_term, second_term), third_term)


def create_kepler_first_law():
    # This function must create a DifferentiableFunction of three
    # variables "p", "e", and "theta" that implements Kepler's first
    # law of planetary motion:
    #
    # f(p, e, theta) = p / (1 + e * cos(theta))
    #
    # https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#First_law

    # TODO
    numerator = Symbol('p')
    denominator_first_term = Constant(1)
    denominator_second_term = Product(Symbol('e'), Cosine('theta'))
    denominator = Sum(denominator_first_term, denominator_second_term)
    return Quotient(numerator, denominator)


def compute_symbolic_absolute_error_propagation(f, variables, approx_x, errors_x):
    # This function must use our symbolic derivation engine to compute
    # an upper bound on the propagation of the absolute error of a
    # function "f", which is assumed to be of class
    # DifferentiableFunction. You are given the list of the variables
    # occurring in function "f", the approximated value of all those
    # variables in dictionary "approx_x", and the absolute errors
    # associated with those approximations in dictionary "errors_x".

    assert(isinstance(f, DifferentiableFunction))

    # TODO
    res = 0
    for var in variables :
        right_term = f.derivative(var).evaluate(approx_x)
        res += abs(right_term) * errors_x[var]

    return res


def compute_sphere_volume_errors(exact_r, number_of_r_decimals, number_of_pi_decimals):
    # This function must return an upper bound on the propagated errors
    # for the computation of a sphere of radius "r", provided that
    # both "r" and "pi" cannot be represented exactly by the computer.
    # You are given the exact value of "r" and the number of decimals
    # (in base 10) that are used to encode "r" and "pi". Your function
    # must return a pair containing both the propagated absolute and
    # relative errors.
    #
    # Hint: Combine "create_sphere_volume()" with
    # "compute_symbolic_absolute_error_propagation()".

    # TODO

    error_r = 0.5 * 10**(-number_of_r_decimals)
    error_pi = 0.5 * 10**(-number_of_pi_decimals)
    errors_x = {'r': error_r, 'pi': error_pi}


    approx_r = round(exact_r, number_of_r_decimals)
    approx_pi = round(math.pi, number_of_pi_decimals)
    approx_x = {'r': approx_r, 'pi': approx_pi}

    print(approx_x, errors_x)

    function = create_sphere_volume()
    absolute_error = compute_symbolic_absolute_error_propagation(
        function,
        ['pi', 'r'],
        approx_x,
        errors_x
    )

    approx_volume = function.evaluate({'pi': math.pi, 'r':exact_r})
    relative_error = absolute_error / abs(approx_volume)
    print(absolute_error, relative_error)
    return absolute_error, relative_error


def compute_kepler_first_law_errors(exact_p, exact_e, exact_theta_degrees, number_of_decimals):
    # This function must return an upper bound on the propagated
    # errors for the computation of Kepler's first law of planetary
    # motion, given variables "p", "e", and "theta" that cannot be
    # represented exactly by the computer. You are given the number of
    # decimals (in base 10) that are used to encode all those
    # variables. Note that "theta" is expressed in degrees, not in
    # radians. Your function must return a pair containing both the
    # propagated absolute and relative errors.
    #
    # Hint: Combine "create_kepler_first_law()" with
    # "compute_symbolic_absolute_error_propagation()".

    # TODO
    error_e = 0.5 * 10**(-number_of_decimals)
    error_p = 0.5 * 10**(-number_of_decimals)
    error_theta = 0.5 * 10**(-number_of_decimals)
    errors_x = {'theta': error_theta, 'p': error_p, 'e': error_e}

    approx_theta = round(math.radians(exact_theta_degrees), number_of_decimals)
    approx_p = round(exact_p, number_of_decimals)
    approx_e = round(exact_e, number_of_decimals)
    approx_x = {'theta': approx_theta, 'p': approx_p, 'e': approx_e}

    function = create_kepler_first_law()
    absolute_error = compute_symbolic_absolute_error_propagation(
        function,
        ['theta', 'p', 'e'],
        approx_x,
        errors_x
    )

    approx_evaluation = function.evaluate(
        {'theta': math.radians(exact_theta_degrees), 'p': exact_p, 'e':exact_e}
    )
    relative_error = absolute_error / abs(approx_evaluation)

    return absolute_error, relative_error


def rank_functions_by_increasing_relative_errors(funcs, variable, value, number_of_decimals):
    # You are given a list of DifferentiableFunction objects in the
    # "funcs" argument. Each of these functions has one argument whose
    # name is "variable". You are also given an exact value for this
    # variable in the "value" argument. You know that each function
    # would compute the same value if no approximation takes place.
    #
    # Now, if the exact value is approximated using the provided
    # number of decimals (base 10), sort the functions in the "funcs"
    # list by increasing propagated bound on the relative error at
    # this approximate value. Your function must return a list of
    # pairs, each pair providing the bound and the function.

    # TODO
    result = list()
    for func in funcs :
        error, approx = 0.5 * 10**(-number_of_decimals), round(value, number_of_decimals)
        # compute the absolute error
        temp = func.derivative(variable).evaluate({variable: approx})
        absolute_error = abs(temp) * error
        # compute the relative error
        approx_value = func.evaluate({variable : value})
        relative_error = absolute_error / abs(approx_value)

        result.append((relative_error, func))

    result.sort(key=lambda x: x[0])

    return result


if __name__ == '__main__' :
    compute_sphere_volume_errors(1.7, 4, 2)
