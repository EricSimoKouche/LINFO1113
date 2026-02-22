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

def encode_floating_point(x):
    # This function must encode the provided floating-point number "x"
    # according to its floating-point representation with binary32
    # encoding (1bit for sign, 8 bits for the exponent, 23 bits for the
    # normalized significand). The output must be a string containing
    # the characters "0" and "1".

    # TODO
    k = 23
    result = ['0'] * 32
    fixed_x = None
    if (x < 0) :
        result[0] = '1'

    fixed_x = encode_fixed_point(abs(x), 32, k)
    print(fixed_x)

    i = 0
    while i < 32 and fixed_x[i] == '0':
        i += 1

    # encode the exponent
    exponent = encode_uint8((abs(i-32)-1)-k+127)

    # Write the exponent in the representation
    for j in range (len(exponent)):
        result[j+1] = exponent[j]

    # Write the significant in the representation
    for k in range(i+1, len(fixed_x)) :
        result[k-i+8] = fixed_x[k]

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

    return sum(lst)

if __name__ == '__main__':
    print(encode_floating_point(33.6875))
    # print(encode_uint8(130))