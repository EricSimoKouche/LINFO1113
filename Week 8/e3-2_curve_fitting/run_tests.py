#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# def ForbiddenFunction(*args):
#     raise Exception('You are supposed to implement the algorithms by yourself :-)')
# Forbidden functions must be done before "import student"
# np.linalg.solve = ForbiddenFunction
# np.linalg.inv = ForbiddenFunction
# import sys
# for forbidden_module in [ 'sympy' ]:
#     if forbidden_module in sys.modules.keys():
#         raise Exception('You are not allowed to import module "%s"' % forbidden_module)

import student
from student import point
import unittest
from grading_toolbox import grade, grade_feedback

def format_equation(eq, b):
    terms = []
    for j, coeff in enumerate(eq):
        if coeff != 0:
            terms.append(f"{coeff}*k{j}")
    lhs = " + ".join(terms) if terms else "0"
    return f"{lhs} = {b}"

class Tests(unittest.TestCase):

    @grade(1)
    def test_piecewise(self):
        points = [ # cm4, p14 example
            point(1, 0),
            point(2, 1),
            point(3, 0),
            point(4, 1),
            point(5, 0)
        ]

        A = np.asarray([
            [1, 0, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 1, 4, 1, 0],
            [0, 0, 1, 4, 1,],
            [0, 0, 0, 0, 1],
        ])

        b = np.asarray([0.0, -12.0, 12.0, -12.0, 0.0])

        student_spline = student.piecewise_cubic_interpolation(points)
        # student_spline.plot_interpolation()

        for i in range(len(points)-1):
            exp_eq = A[i]
            stud_eq = student_spline.A[i]
            
            exp_b = b[i]
            stud_b = student_spline.b[i]

            exp_str = format_equation(exp_eq, exp_b)
            stud_str = format_equation(stud_eq, stud_b)

            message = (
                f"\nExpected {i}th equation:\n"
                f"{exp_str}\n"
                f"Your {i}th equation:\n"
                f"{stud_str}\n"
            )

            self.assertTrue(np.array_equal(exp_eq, stud_eq), message)
            self.assertAlmostEqual(exp_b, stud_b, message)

        stud_k = student_spline.ki
        exp_k = [0.0, -30/7, 36/7, -30/7, 0.0]

        for i in range(len(points)):
            stud_ki = stud_k[i]
            exp_ki = exp_k[i]

            self.assertAlmostEqual(exp_ki, stud_ki, msg=f"\nExpected: k{i} = {exp_ki}\nYour answer: k{i} = {stud_ki}")

        exp_fx = 0.7676
        stud_fx = student_spline.interpolate(1.5)

        self.assertAlmostEqual(exp_fx, stud_fx, delta=0.0005)

        n_stud_roots = len(student_spline.roots)
        self.assertEqual(3, len(student_spline.roots), f"You found {n_stud_roots} but you were expected to find three.")


        for exp_root, stud_root in zip([1, 2.9999999999999796, 4.999999999999937], student_spline.roots):
            self.assertAlmostEqual(exp_root, stud_root, msg=f"You found a wrong root. Your answer {stud_root}, expected answer: {exp_root}.", delta=1e-4)

        points_2 = [ # root test
            point(0.2, 1.15),
            point(0.4, 0.855),
            point(0.6, 0.377),
            point(0.8, -0.666),
            point(1, -1.049)
        ]

        student_spline_2 = student.piecewise_cubic_interpolation(points_2)

        n_stud_roots_2 = len(student_spline_2.roots)
        self.assertEqual(1, n_stud_roots_2, f"You found {n_stud_roots_2} but you were expected to find one.")

        stud_root_2 = student_spline_2.roots[0]
        exp_root_2 = 0.6734821456670766
        self.assertAlmostEqual(exp_root_2, stud_root_2, msg=f"You found a wrong root. Your answer {stud_root_2}, expected answer: {exp_root_2}.", delta=1e-4)

if __name__ == '__main__':
    unittest.main()
