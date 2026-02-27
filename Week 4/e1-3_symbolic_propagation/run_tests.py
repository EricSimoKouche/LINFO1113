#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import student

import sys
for forbidden_module in [ 'numpy', 'sympy', 'ctypes' ]:
    if forbidden_module in sys.modules.keys():
        raise Exception('You are not allowed to import module "%s"' % forbidden_module)

import math
import unittest

from grading_toolbox import grade, grade_feedback

class Tests(unittest.TestCase):
    @grade(1)
    def test_symbol(self):
        f = student.Symbol('x')
        self.assertAlmostEqual(-2.3, f.evaluate({'x' : -2.3, 'y' : 1.3 }))
        df = f.derivative('x')
        self.assertTrue(isinstance(df, student.Constant))
        self.assertAlmostEqual(1, df.value)
        df = f.derivative('y')
        self.assertTrue(isinstance(df, student.Constant))
        self.assertAlmostEqual(0, df.value)

    @grade(1)
    def test_constant(self):
        f = student.Constant(4.9)
        self.assertAlmostEqual(4.9, f.evaluate({'x' : -2.3, 'y' : 1.3 }))
        df = f.derivative('x')
        self.assertTrue(isinstance(df, student.Constant))
        self.assertAlmostEqual(0, df.value)

    @grade(1)
    def test_sum(self):
        f = student.Sum(student.Constant(3), student.Symbol('y'))
        self.assertAlmostEqual(4.3, f.evaluate({'x' : -2.3, 'y' : 1.3, 'z' : 4 }))

        df = f.derivative('x')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Constant))
        self.assertTrue(isinstance(df.right, student.Constant))
        self.assertAlmostEqual(0, df.left.value)
        self.assertAlmostEqual(0, df.right.value)

        df = f.derivative('y')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Constant))
        self.assertTrue(isinstance(df.right, student.Constant))
        self.assertAlmostEqual(0, df.left.value)
        self.assertAlmostEqual(1, df.right.value)

        df = f.derivative('z')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Constant))
        self.assertTrue(isinstance(df.right, student.Constant))
        self.assertAlmostEqual(0, df.left.value)
        self.assertAlmostEqual(0, df.right.value)

    @grade(1)
    def test_product(self):
        f = student.Product(student.Symbol('x'), student.Symbol('y'))
        self.assertAlmostEqual(-2.99, f.evaluate({'x' : -2.3, 'y' : 1.3, 'z' : 4 }))

        df = f.derivative('x')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Product))
        self.assertTrue(isinstance(df.left.left, student.Constant))
        self.assertAlmostEqual(1, df.left.left.value)
        self.assertTrue(isinstance(df.left.right, student.Symbol))
        self.assertEqual('y', df.left.right.name)
        self.assertTrue(isinstance(df.right, student.Product))
        self.assertTrue(isinstance(df.right.left, student.Symbol))
        self.assertEqual('x', df.right.left.name)
        self.assertTrue(isinstance(df.right.right, student.Constant))
        self.assertAlmostEqual(0, df.right.right.value)

        df = f.derivative('y')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Product))
        self.assertTrue(isinstance(df.left.left, student.Constant))
        self.assertAlmostEqual(0, df.left.left.value)
        self.assertTrue(isinstance(df.left.right, student.Symbol))
        self.assertEqual('y', df.left.right.name)
        self.assertTrue(isinstance(df.right, student.Product))
        self.assertTrue(isinstance(df.right.left, student.Symbol))
        self.assertEqual('x', df.right.left.name)
        self.assertTrue(isinstance(df.right.right, student.Constant))
        self.assertAlmostEqual(1, df.right.right.value)

        df = f.derivative('z')
        self.assertTrue(isinstance(df, student.Sum))
        self.assertTrue(isinstance(df.left, student.Product))
        self.assertTrue(isinstance(df.left.left, student.Constant))
        self.assertAlmostEqual(0, df.left.left.value)
        self.assertTrue(isinstance(df.left.right, student.Symbol))
        self.assertEqual('y', df.left.right.name)
        self.assertTrue(isinstance(df.right, student.Product))
        self.assertTrue(isinstance(df.right.left, student.Symbol))
        self.assertEqual('x', df.right.left.name)
        self.assertTrue(isinstance(df.right.right, student.Constant))
        self.assertAlmostEqual(0, df.right.right.value)

    @grade(1)
    def test_create_sphere_volume(self):
        f = student.create_sphere_volume()
        self.assertAlmostEqual(4, f.evaluate({ 'pi' : 3, 'r' : 1 }))
        self.assertAlmostEqual(4.186666666666667, f.evaluate({ 'pi' : 3.14, 'r' : 1 }))
        self.assertAlmostEqual(13.5, f.evaluate({ 'pi' : 3, 'r' : 1.5 }))
        self.assertAlmostEqual(82.445526, f.evaluate({ 'pi' : 3.1415, 'r' : 2.7 }))

    @grade(1)
    def test_create_kepler_first_law(self):
        f = student.create_kepler_first_law()
        self.assertAlmostEqual(0.5314649884318062, f.evaluate({ 'p' : 2, 'e' : 3, 'theta' : 0.4 }))
        self.assertAlmostEqual(0.8988841921911321, f.evaluate({ 'p' : 1.5, 'e' : 0.7, 'theta' : -0.3 }))

    @grade(1)
    def test_compute_symbolic_absolute_error_propagation(self):
        f = student.Symbol('x')
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x' ], { 'x' : 0.2 }, { 'x' : 0.01 })
        self.assertAlmostEqual(0.01, e)

        f = student.Constant(5)
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'y' ], { 'y' : 0.2 }, { 'y' : 0.01 })
        self.assertAlmostEqual(0, e)

        f = student.Sum(student.Symbol('x'), student.Symbol('y'))
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(0.71, e)

        f = student.Product(student.Symbol('x'), student.Symbol('y'))
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(0.141, e)

        f = student.Quotient(student.Symbol('x'), student.Symbol('y'))
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(14.1, e)

        f = student.Sine('x')
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(0.00980066577841242, e)

        f = student.Cosine('y')
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(0.0698833916527797, e)

        f = student.Power(student.Symbol('x'), -3.7)
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(71.3445403770011, e)

        f = student.Power(student.Symbol('x'), 2.9)
        e = student.compute_symbolic_absolute_error_propagation(f, [ 'x', 'y' ], { 'x' : 0.2, 'y' : -0.1 }, { 'x' : 0.01, 'y' : 0.7 })
        self.assertAlmostEqual(0.00136255797398210, e)

    @grade(1)
    def test_bound_absolute_error_propagation_symbolic(self):
        # In INGInious 2024-2025, this corresponds to "Question 8: Bounds on error propagation (1/4)"
        # This is the same as "bound_absolute_error_propagation()" in
        # e1-2, but using the symbolic derivation engine
        a = student.Product(student.Sine('x'), student.Power(student.Symbol('y'), 2))
        b = student.Quotient(student.Power(student.Symbol('z'), 3), student.Symbol('y'))
        c = student.Product(student.Constant(-1), student.Symbol('x'))
        d = student.Product(student.Symbol('x'), student.Product(student.Symbol('y'), student.Symbol('z')))
        f = student.Sum(a, student.Sum(b, student.Sum(c, d)))

        variables = [ 'x', 'y', 'z' ]
        approx = {
            'x' : 0.3722155379767289,
            'y' : 0.16755531475417695,
            'z' : 0.8475555988229495
        }
        errors = {
            'x' : 0.27575971443165814,
            'y' : 0.7232577540143574,
            'z' : 0.7634552494572096,
        }
        self.assertAlmostEqual(25.46492573835946, student.compute_symbolic_absolute_error_propagation(f, variables, approx, errors))

        approx = {
            'x' : 0.19853090814255114,
            'y' : 0.1314940222801284,
            'z' : 0.626009899016065,
        }
        errors = {
            'x' : 0.03859593033235187,
            'y' : 0.5941479165349165,
            'z' : 0.11293239379734477,
        }
        self.assertAlmostEqual(9.372734303395738, student.compute_symbolic_absolute_error_propagation(f, variables, approx, errors))

        approx = {
            'x' : 0.06975515572012037,
            'y' : 0.6905368106321245,
            'z' : 0.7295514359464482,
        }
        errors = {
            'x' : 0.6768203017581806,
            'y' : 0.7597055183033526,
            'z' : 0.09406204204782742,
        }
        self.assertAlmostEqual(0.7427831276375904, student.compute_symbolic_absolute_error_propagation(f, variables, approx, errors))

        approx = {
            'x' : 0.5236169040282306,
            'y' : 0.24355592719442065,
            'z' : 0.09172551400730389,
        }
        errors = {
            'x' : 0.2942336727566225,
            'y' : 0.8430683795836534,
            'z' : 0.3803830373900636,
        }
        self.assertAlmostEqual(0.5953403488632976, student.compute_symbolic_absolute_error_propagation(f, variables, approx, errors))

        approx = {
            'x' : 0.7540361114611339,
            'y' : 0.755556532950095,
            'z' : 0.5556628846408851,
        }
        errors = {
            'x' : 0.6463729875887189,
            'y' : 0.16750770137574889,
            'z' : 0.5674418646932039,
        }
        self.assertAlmostEqual(1.3181026679356553, student.compute_symbolic_absolute_error_propagation(f, variables, approx, errors))

    @grade(1)
    def test_compute_sphere_volume_errors(self):
        # In INGInious 2024-2025, this corresponds to "Question 9: Bounds on error propagation (2/4)"
        e = student.compute_sphere_volume_errors(2.05, 3, 3)
        self.assertAlmostEqual(0.032151926666666664, e[0])
        self.assertAlmostEqual(0.0008909571350841911, e[1])

        e = student.compute_sphere_volume_errors(1.7, 4, 2)
        self.assertAlmostEqual(0.034568253333333326, e[0])
        self.assertAlmostEqual(0.0016797399935028154, e[1])

    @grade(1)
    def test_compute_kepler_first_law_errors(self):
        # In INGInious 2024-2025, this corresponds to "Question 11: Bounds on error propagation (4/4)"
        e = student.compute_kepler_first_law_errors(1.45, 0.7, 45, 2)
        self.assertAlmostEqual(0.007257830715357717, e[0])
        self.assertAlmostEqual(0.007482947335331816, e[1])

    @grade(1)
    def test_rank_functions_by_increasing_relative_errors(self):
        # In INGInious 2024-2025, this corresponds to "Question 10: Bounds on error propagation (3/4)"

        # Consider the following 4 mathematical expressions:
        # f1 = (2 - x) ^ 6
        # f2 = (2 + x) ^ -6
        # f3 = (7 - 4 * x) ^ 3
        # f4 = (7 + 4 * x) ^ -3

        f1 = student.Power(student.Sum(student.Constant(2), student.Product(student.Constant(-1), student.Symbol('x'))), 6)
        f2 = student.Power(student.Sum(student.Constant(2), student.Product(student.Constant(1),  student.Symbol('x'))), -6)
        f3 = student.Power(student.Sum(student.Constant(7), student.Product(student.Constant(-4), student.Symbol('x'))), 3)
        f4 = student.Power(student.Sum(student.Constant(7), student.Product(student.Constant(4),  student.Symbol('x'))), -3)

        # It is easy to prove that these 4 expressions have the same
        # value if "x = sqrt(3)". However, they do not exhibit the
        # same numerical stability if an approximation of "x" is used
        # in the place of the exact value.

        e = student.rank_functions_by_increasing_relative_errors([ f1, f2, f3, f4 ], 'x', math.sqrt(3), 3)

        self.assertEqual(4, len(e))
        self.assertAlmostEqual(0.000430805761821944, e[0][0])
        self.assertAlmostEqual(0.000803924185670831, e[1][0])
        self.assertAlmostEqual(0.0112067713211527, e[2][0])
        self.assertAlmostEqual(0.0840429964885242, e[3][0])
        self.assertEqual(f4, e[0][1])  # f4 is the best expression
        self.assertEqual(f2, e[1][1])
        self.assertEqual(f1, e[2][1])
        self.assertEqual(f3, e[3][1])  # f3 is the worst expression

if __name__ == '__main__':
    unittest.main()
