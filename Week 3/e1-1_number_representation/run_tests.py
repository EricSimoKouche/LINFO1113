#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import student

import sys
for forbidden_module in [ 'numpy', 'sympy', 'ctypes' ]:
    if forbidden_module in sys.modules.keys():
        raise Exception('You are not allowed to import module "%s"' % forbidden_module)

def forbidden(*args, **kwargs):
    raise RuntimeError('This function is not allowed for this exercise.')

import struct
struct.pack = forbidden

import unittest

from grading_toolbox import grade, grade_feedback

class Tests(unittest.TestCase):
    @grade(1)
    def test_encode_uint8(self):
        # In INGInious 2024-2025, this corresponds to "Question 1: Integer representation (1/2)"
        self.assertEqual('00011011', student.encode_uint8(27))
        self.assertEqual('00110100', student.encode_uint8(52))
        self.assertEqual('10000001', student.encode_uint8(129))
        self.assertEqual('11111111', student.encode_uint8(255))
        self.assertEqual('XXXXXXXX', student.encode_uint8(-1))
        self.assertEqual('XXXXXXXX', student.encode_uint8(256))

    @grade(1)
    def test_encode_int8(self):
        # In INGInious 2024-2025, this corresponds to "Question 2: Integer representation (2/2)"
        self.assertEqual('00011011', student.encode_int8(27))
        self.assertEqual('00110100', student.encode_int8(52))
        self.assertEqual('01111111', student.encode_int8(127))
        self.assertEqual('10000000', student.encode_int8(-128))
        self.assertEqual('11111111', student.encode_int8(-1))
        self.assertEqual('XXXXXXXX', student.encode_int8(128))
        self.assertEqual('XXXXXXXX', student.encode_int8(-129))
        self.assertEqual('11100101', student.encode_int8(-27))
        self.assertEqual('11001100', student.encode_int8(-52))

    @grade(1)
    def test_encode_fixed_point(self):
        # In INGInious 2024-2025, this corresponds to "Question 3: Decimal number representation (1/2)"
        self.assertEqual('000100001110', student.encode_fixed_point(33.6875, 12, 3))
        self.assertEqual('010000110110', student.encode_fixed_point(33.6875, 12, 5))

    @grade(1)
    def test_encode_floating_point(self):
        # In INGInious 2024-2025, this corresponds to "Question 4: Decimal number representation (2/2)"
        self.assertEqual('01000010000001101100000000000000', student.encode_floating_point(33.6875))
        self.assertEqual('00111110001000000000000000000000', student.encode_floating_point(0.15625)) # First example in the slides
        self.assertEqual('11000001010110100000000000000000', student.encode_floating_point(-13.625)) # Second example in the slides

    @grade(1)
    def test_smart_sum(self):
        # In INGInious 2024-2025, this corresponds to "Question 5: Order of operations"
        self.assertAlmostEqual(1.0000000000000006e+16, student.smart_sum([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0**16]))
        self.assertAlmostEqual(1.0000000000000006e+16, student.smart_sum([10.0**16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

if __name__ == '__main__':
    unittest.main()
