#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from contextlib import nullcontext


def encode_uint8(x):
    # This function must encode the input integer "x" using the
    # *unsigned* integer with 8 bits (uint8) binary representation.
    #
    # The output must be a string with the following format:
    # "XXXXXXXX", where the Xs are replaced by 0 or 1. If the input
    # integer can't be represented, leave it as "XXXXXXXX".

    # TODO
    MAX = 255
    MIN = 0

    if x < MIN or x > MAX:
        return 'XXXXXXXX'

    result = ['0', '0', '0', '0', '0', '0', '0', '0']
    for i in range(7, -1, -1) :
        bit = x % 2
        x = x // 2
        result[i] = str(bit)

    return ''.join(result)

def encode_int8(x):
    # This function must encode the input integer "x" using the *signed*
    # integer with 8 bits (uint8) binary representation.
    #
    # The output must be a string with the following format:
    # "XXXXXXXX", where the Xs are replaced by 0 or 1. If the input
    # integer can't be represented, leave it as "XXXXXXXX".

    # TODO
    MAX, MIN = 127, -128

    if x < MIN or x > MAX:
        return 'XXXXXXXX'

    result = ['0', '0', '0', '0', '0', '0', '0', '0']
    if x < 0 :
        result[0] = '1'
        x = 128 + x

    for i in range(7, 0, -1):
        bit = x % 2
        x = x // 2
        result[i] = str(bit)

    return ''.join(result)

def encode_fixed_point(x, num_bits, bits_after_comma):
    # This function must encode the provided floating-point number "x"
    # according to its fixed-point representation, with a given number
    # of bits ("num_bits") and a given number of bits after the comma
    # ("bits_after_comma"). If a certain number can't be represented
    # exactly, represent its closest approximation. The output must be
    # a string containing the characters "0" and "1".

    # TODO
    scale = 2 ** bits_after_comma
    MIN, MAX = -2**(num_bits-1) , 2**num_bits-1
    fixed_x = round(x * scale)

    if fixed_x < MIN :
        fixed_x = MIN
    elif fixed_x > MAX :
        fixed_x = MAX

    result = ['0'] * num_bits
    if fixed_x < 0 :
        result[0] = '1'
        fixed_x = -MIN + fixed_x
        # print('the approximate ' + str((fixed_x - 2 ** (num_bits - 1)) / 2 ** bits_after_comma))
    else :
       # print('the approximate ' + str(fixed_x / 2 ** bits_after_comma))
        pass

    for i in range (num_bits-1, 0, -1):
        bit = fixed_x % 2
        fixed_x = fixed_x // 2
        result[i] = str(bit)

    return ''.join(result)


def is_integer(n):
    return isinstance(n, (int, float)) and n == int(n)

def encode_floating_point(x):
    # This function must encode the provided floating-point number "x"
    # according to its floating-point representation with binary32
    # encoding (1bit for sign, 8 bits for the exponent, 23 bits for the
    # normalized significand). The output must be a string containing
    # the characters "0" and "1".

    # TODO
    # recupere l'information sur le signe d
    is_negative = x < 0
    fixed_x = x
    k = 0

    # je vais multiplier par 2 jusqu'a ce que le nombre devienne un entier
    while not is_integer(fixed_x):
        fixed_x *= 2
        k+=1

    fixed_x = round(abs(fixed_x))

    # j'encode la mantisse en binaire
    mantissa = ""
    while fixed_x > 0 :
        bit = fixed_x % 2
        fixed_x //= 2
        mantissa += str(bit)

    # Trouver la position du 1 le plus a gauche
    index = 0
    for i in range(len(mantissa)) :
        if mantissa[i] == '1' :
            index = i

    # le representation binaire de l'exponent.
    exponent = encode_uint8(index - k + 127)

    result = ["0"] * 32
    if is_negative : result[0] = "1"
    j = 1
    # je copie les bits de l'exposant
    for i in range(len(exponent)) :
        result[j] = exponent[i]
        j += 1

    # j'inverse l'ordre des bits de la mantisse
    mantissa = mantissa[::-1]

    # je copie les bits de la mantisse
    for i in range(1, len(mantissa)) :
        result[j] = mantissa[i]
        j += 1

    return ''.join(result)


def smart_sum(lst):
    # You are given a list "lst" containing positive floating-point
    # numbers. Some numbers in the list are very small, while others
    # are very large. Because of the limited precision of
    # floating-point arithmetic, simply adding the numbers in their
    # given order can lead to rounding errors, causing small values to
    # be lost in the final sum.
    #
    # This function must compute the sum of all numbers in the list
    # while minimizing numerical precision loss.

    # TODO
    lst.sort()
    return sum(lst)