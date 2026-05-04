#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as nt
import unittest

import student
from grading_toolbox import grade


class Tests(unittest.TestCase):
    @grade(1)
    # Corresponds to E7.2 from INGInious 2024-2025
    def test_euler(self):
        def F(x, y):
            return 1/2 * y
        x0, y0, h, xStop = 0, 1, 0.1, 5

        xStudent, yStudent = student.euler(F, x0, y0, xStop, h)

        x = np.arange(x0, xStop + h, h)
        y = np.array([ 1.        ,  1.05      ,  1.1025    ,  1.157625  ,  1.21550625,
        1.27628156,  1.34009564,  1.40710042,  1.47745544,  1.55132822,
        1.62889463,  1.71033936,  1.79585633,  1.88564914,  1.9799316 ,
        2.07892818,  2.18287459,  2.29201832,  2.40661923,  2.5269502 ,
        2.65329771,  2.78596259,  2.92526072,  3.07152376,  3.22509994,
        3.38635494,  3.55567269,  3.73345632,  3.92012914,  4.1161356 ,
        4.32194238,  4.53803949,  4.76494147,  5.00318854,  5.25334797,
        5.51601537,  5.79181614,  6.08140694,  6.38547729,  6.70475115,
        7.03998871,  7.39198815,  7.76158756,  8.14966693,  8.55715028,
        8.98500779,  9.43425818,  9.90597109, 10.40126965, 10.92133313,
        11.46739979])

        nt.assert_array_almost_equal(xStudent, x)
        nt.assert_array_almost_equal(yStudent, y)

    @grade(1)
    # Corresponds to E7.3 from INGInious 2024-2025
    def test_euler_2(self):
        def F(x, y):
            return np.array([y[1], -4 * y[0]])

        x0, y0, xStop, h = 0, np.array([1,0]), 5, 0.2
        xStudent, yStudent = student.euler(F, x0, y0, xStop, h)

        x = np.arange(x0, xStop + h, h)
        y = np.array([[ 1.        ,  0.        ],
       [ 1.        , -0.8       ],
       [ 0.84      , -1.6       ],
       [ 0.52      , -2.272     ],
       [ 0.0656    , -2.688     ],
       [-0.472     , -2.74048   ],
       [-1.020096  , -2.36288   ],
       [-1.492672  , -1.5468032 ],
       [-1.80203264, -0.3526656 ],
       [-1.87256576,  1.08896051],
       [-1.65477366,  2.58701312],
       [-1.13737103,  3.91083205],
       [-0.35520462,  4.82072887],
       [ 0.60894115,  5.10489257],
       [ 1.62991966,  4.61773965],
       [ 2.5534676 ,  3.31380392],
       [ 3.21622838,  1.27102984],
       [ 3.47043435, -1.30195286],
       [ 3.21004378, -4.07830034],
       [ 2.39438371, -6.64633536],
       [ 1.06511664, -8.56184233],
       [-0.64725183, -9.41393564],
       [-2.53003896, -8.89613417],
       [-4.30926579, -6.87210301],
       [-5.68368639, -3.42469038],
       [-6.36862447,  1.12225874]])

        nt.assert_array_almost_equal(xStudent, x)
        nt.assert_array_almost_equal(yStudent, y)

    @grade(1)
    def test_RK2(self):
        def F(x, y):
            return 1/2 * y
        x0, y0, h, xStop = 0, 1, 0.1, 5
        xStudent, yStudent = student.RK2(F, x0, y0, xStop, h)

        x = np.arange(x0, xStop + h, h)
        y = np.array([ 1.        ,  1.05125   ,  1.10512656,  1.1617643 ,  1.22130472,
        1.28389659,  1.34969629,  1.41886822,  1.49158522,  1.56802896,
        1.64839044,  1.73287045,  1.82168006,  1.91504117,  2.01318703,
        2.11636286,  2.22482646,  2.33884882,  2.45871482,  2.58472395,
        2.71719105,  2.8564471 ,  3.00284001,  3.15673556,  3.31851826,
        3.48859232,  3.66738267,  3.85533604,  4.05292201,  4.26063426,
        4.47899177,  4.7085401 ,  4.94985278,  5.20353273,  5.47021378,
        5.75056224,  6.04527855,  6.35509908,  6.68079791,  7.0231888 ,
        7.38312723,  7.7615125 ,  8.15929001,  8.57745362,  9.01704812,
        9.47917184,  9.9649794 , 10.47568459, 11.01256343, 11.5769573 ,
        12.17027636])

        nt.assert_array_almost_equal(xStudent, x)
        nt.assert_array_almost_equal(yStudent, y)

    @grade(1)
    # Heun's method (RK2 with a=b=1/2, alpha=beta=1, trapezoidal variant).
    # Uses F(x,y) = x*y so that the x-advance in K1 (full step vs. half step)
    # produces different results than the Midpoint/Modified-Euler method.
    def test_heun(self):
        def F(x, y):
            return x * y
        x0, y0, h, xStop = 1, 1.0, 0.5, 2.5

        xStudent, yStudent = student.heun(F, x0, y0, xStop, h)

        x = np.arange(x0, xStop + h, h)
        # Reference values computed from Heun's recurrence:
        #   K0 = h * F(x_i, y_i)
        #   K1 = h * F(x_i + h, y_i + K0)
        #   y_{i+1} = y_i + (K0 + K1) / 2
        # Step 1: K0=0.5, K1=0.5*1.5*1.5=1.125,  y=1+(0.5+1.125)/2=1.8125
        # Step 2: K0=1.359375, K1=3.171875,        y=1.8125+2.265625=4.078125
        # Step 3: K0=4.078125, K1=10.1953125,      y=4.078125+7.136719=11.214844
        y = np.array([ 1.          ,  1.8125      ,  4.078125    , 11.21484375 ])

        nt.assert_array_almost_equal(xStudent, x)
        nt.assert_array_almost_equal(yStudent, y)

    @grade(1)
    # Corresponds to E7.7 from INGInious 2024-2025
    def test_RK4(self):
        def F(x, y):
            return np.array([y[1], -4 * y[0]])
        x0, y0, h, xStop = 0, np.array([1, 0]), 0.2, 5

        xStudent, yStudent = student.RK4(F, x0, y0, xStop, h)

        x = np.arange(x0, xStop + h, h)
        y = np.array([[ 1.        ,  0.        ],
        [ 0.92106667, -0.77866667],
        [ 0.69678336, -1.43440782],
        [ 0.36255254, -1.86374721],
        [-0.0288744 , -1.998943  ],
        [-0.41572282, -1.81867624],
        [-0.73694407, -1.35141256],
        [-0.9418496 , -0.67090728],
        [-0.99810955,  0.11543656],
        [-0.89685379,  0.8835194 ],
        [-0.65407035,  1.51213042],
        [-0.30808101,  1.90207571],
        [ 0.08650759,  1.99183095],
        [ 0.46742235,  1.76724851],
        [ 0.77455152,  1.2637875 ],
        [ 0.95943089,  0.56091509],
        [ 0.99289128, -0.23043666],
        [ 0.86966072, -0.98537887],
        [ 0.60919508, -1.58477545],
        [ 0.25260633, -1.93404374],
        [-0.14382658, -1.97807936],
        [-0.51753998, -1.70995   ],
        [-0.80955909, -1.17198681],
        [-0.97380466, -0.44910131],
        [-0.98436407,  0.34461698],
        [-0.83957949,  1.0839067 ]])

        nt.assert_array_almost_equal(xStudent, x)
        nt.assert_array_almost_equal(yStudent, y)

    @grade(1)
    # Corresponds to E7.9 from INGInious 2024-2025 :
    # The movement of a spring-mass system is defined by the following differential equation:

    # m*d^2x/dt^2 + c*dx/dt + kx = 0,

    # where:

    # x(t) is the body position at time t [m]
    # t is the time variable [s]
    # m is the mass of the body attached to the spring [kg]
    # c is the damping coefficient [kg/s]
    # k is the spring constant [kg/s2]
    # Solve the system (with your RK4 implementation) for

    # c=0.95
    # k=5
    # m=1.5
    # x(0)=10
    # x′(0)=0
    # for the timespan t∈[0,20] with time steps of 0.25.
    # Store the t and x values into the vectors t and x, respectively.

    def test_RK4_2(self):
        m = 1.5
        c = 0.95
        k = 5
        h = 0.25
        t0 = 0
        t_end = 20
        x0 = 10
        v0 = 0

        def F(t, X):
            x1, x2 = X
            dx1 = x2
            dx2 = (-c * x2 - k * x1) / m
            return np.array([dx1, dx2])

        tStudent, yStudent = student.RK4(F, t0, np.array([x0, v0]), t_end, h)

        t = np.arange(t0, t_end + h, h)
        y = np.array([[ 1.00000000e+01,  0.00000000e+00],
       [ 9.02921851e+00, -7.44060671e+00],
       [ 6.49179984e+00, -1.23846828e+01],
       [ 3.09710130e+00, -1.42618522e+01],
       [-3.87064552e-01, -1.31655494e+01],
       [-3.28827930e+00, -9.73822889e+00],
       [-5.14280917e+00, -4.96947206e+00],
       [-5.75283139e+00,  4.20589927e-02],
       [-5.18496883e+00,  4.31248563e+00],
       [-3.71899637e+00,  7.14210621e+00],
       [-1.76371498e+00,  8.20623223e+00],
       [ 2.39283603e-01,  7.56176976e+00],
       [ 1.90397904e+00,  5.58062669e+00],
       [ 2.96484173e+00,  2.83325214e+00],
       [ 3.30945383e+00, -4.83580484e-02],
       [ 2.97738378e+00, -2.49926153e+00],
       [ 2.13046421e+00, -4.11866760e+00],
       [ 1.00428112e+00, -4.72176727e+00],
       [-1.47197035e-01, -4.34310947e+00],
       [-1.10236850e+00, -3.19797290e+00],
       [-1.70919837e+00, -1.61518825e+00],
       [-1.90381197e+00,  4.17001554e-02],
       [-1.70968519e+00,  1.44830838e+00],
       [-1.22042333e+00,  2.37506923e+00],
       [-5.71788209e-01,  2.71680374e+00],
       [ 9.01599765e-02,  2.49442789e+00],
       [ 6.38209119e-01,  1.83254787e+00],
       [ 9.85310999e-01,  9.20711117e-01],
       [ 1.09517831e+00, -3.19633142e-02],
       [ 9.81725634e-01, -8.39220781e-01],
       [ 6.99092173e-01, -1.36957229e+00],
       [ 3.25512135e-01, -1.56316521e+00],
       [-5.50149080e-02, -1.43262975e+00],
       [-3.69463199e-01, -1.05008522e+00],
       [-5.67994530e-01, -5.24789703e-01],
       [-6.29997286e-01,  2.29686228e-02],
       [-5.63711301e-01,  4.86247966e-01],
       [-4.00447855e-01,  7.89737704e-01],
       [-1.85289288e-01,  8.99382504e-01],
       [ 3.34567979e-02,  8.22791508e-01],
       [ 2.13870915e-01,  6.01703252e-01],
       [ 3.27419840e-01,  2.99094370e-01],
       [ 3.62397835e-01, -1.58448119e-02],
       [ 3.23680073e-01, -2.81712598e-01],
       [ 2.29374432e-01, -4.55375933e-01],
       [ 1.05458990e-01, -5.17460182e-01],
       [-2.02853052e-02, -4.72539856e-01],
       [-1.23795542e-01, -3.44769382e-01],
       [-1.88736501e-01, -1.70447840e-01],
       [-2.08461372e-01,  1.06268000e-02],
       [-1.85852232e-01,  1.63200751e-01],
       [-1.31380663e-01,  2.62570921e-01],
       [-6.00158629e-02,  2.97716155e-01],
       [ 1.22660305e-02,  2.71381277e-01],
       [ 7.16525076e-02,  1.97543828e-01],
       [ 1.08791993e-01,  9.71257535e-02],
       [ 1.19910904e-01, -6.98169427e-03],
       [ 1.06711734e-01, -9.45378990e-02],
       [ 7.52497764e-02, -1.51395372e-01],
       [ 3.41504648e-02, -1.71285593e-01],
       [-7.39886125e-03, -1.55852617e-01],
       [-4.14697343e-02, -1.13184409e-01],
       [-6.27087495e-02, -5.53396250e-02],
       [-6.89739117e-02,  4.51520496e-03],
       [-6.12701761e-02,  5.47593308e-02],
       [-4.30989014e-02,  8.72907145e-02],
       [-1.94300636e-02,  9.85444694e-02],
       [ 4.45309024e-03,  8.95037187e-02],
       [ 2.39996516e-02,  6.48482166e-02],
       [ 3.61451121e-02,  3.15279834e-02],
       [ 3.96738313e-02, -2.88401041e-03],
       [ 3.51786055e-02, -3.17160566e-02],
       [ 2.46839305e-02, -5.03283900e-02],
       [ 1.10534475e-02, -5.66939436e-02],
       [-2.67472082e-03, -5.13997267e-02],
       [-1.38884184e-02, -3.71533239e-02],
       [-2.08334546e-02, -1.79602999e-02],
       [-2.28200472e-02,  1.82368203e-03],
       [-2.01976403e-02,  1.83683253e-02],
       [-1.41367462e-02,  2.90166738e-02],
       [-6.28732729e-03,  3.26162542e-02]])

        nt.assert_array_almost_equal(tStudent, t)
        nt.assert_array_almost_equal(yStudent[:, 0], y[:, 0])


# ---------------------------------------------------------------------------
# Simple debug tests — not graded.
# These use small step counts and known results so you can verify each step
# by hand or with a calculator before running the full graded suite.
# ---------------------------------------------------------------------------

class DebugTests(unittest.TestCase):

    def test_euler_constant_derivative(self):
        # y' = 2 (constant), y(0) = 0 → exact solution y = 2x.
        # Euler is exact for constant derivatives (no truncation error).
        # Expected by hand: y_1 = 0 + 1*2 = 2, y_2 = 2 + 2 = 4, y_3 = 6.
        F = lambda x, y: 2.0
        x, y = student.euler(F, x0=0, y0=0, xStop=3, h=1)
        nt.assert_array_almost_equal(x, [0, 1, 2, 3])
        nt.assert_array_almost_equal(y, [0, 2, 4, 6])

    def test_euler_one_step(self):
        # y' = y/2, y(0) = 1 → exact solution y = e^(x/2).
        # Check a single Euler step:
        #   y_1 = y_0 + h * F(x_0, y_0) = 1 + 0.5 * (1/2) = 1.25
        F = lambda _x, y: y / 2
        x, y = student.euler(F, x0=0, y0=1, xStop=0.5, h=0.5)
        nt.assert_array_almost_equal(x, [0, 0.5])
        nt.assert_array_almost_equal(y, [1, 1.25])

    def test_euler_system_one_step(self):
        # System: y' = [y[1], -y[0]], y(0) = [1, 0] (simple harmonic oscillator).
        # One Euler step with h=0.1:
        #   y_1 = [1, 0] + 0.1 * [0, -1] = [1.0, -0.1]
        F = lambda _x, y: np.array([y[1], -y[0]])
        _, y = student.euler(F, x0=0, y0=np.array([1, 0]), xStop=0.1, h=0.1)
        nt.assert_array_almost_equal(y[1], [1.0, -0.1])

    def test_RK2_one_step(self):
        # Modified Euler (midpoint), y' = y/2, y(0) = 1, h = 0.5.
        # By hand:
        #   K0 = 0.5 * (1/2) = 0.25
        #   K1 = 0.5 * (1 + 0.25/2) / 2 = 0.5 * 0.5625 = 0.28125
        #   y_1 = 1 + 0.28125 = 1.28125
        # Compare with Euler (1.25): RK2 is closer to exact e^0.25 ≈ 1.2840.
        F = lambda _x, y: y / 2
        x, y = student.RK2(F, x0=0, y0=1, xStop=0.5, h=0.5)
        nt.assert_array_almost_equal(x, [0, 0.5])
        nt.assert_array_almost_equal(y, [1, 1.28125])

    def test_heun_one_step(self):
        # Heun's method, y' = x*y, y(1) = 1, h = 0.5.
        # F depends on x, so the x-advance in K1 (full h for Heun vs h/2 for midpoint)
        # gives a different result — use this to check you implemented the right formula.
        # By hand:
        #   K0 = 0.5 * F(1, 1) = 0.5 * 1*1 = 0.5
        #   K1 = 0.5 * F(1 + 0.5, 1 + 0.5) = 0.5 * 1.5*1.5 = 1.125
        #   y_1 = 1 + (0.5 + 1.125) / 2 = 1.8125
        # Note: midpoint (RK2) would give y_1 = 1.78125 (different!).
        F = lambda x, y: x * y
        x, y = student.heun(F, x0=1, y0=1, xStop=1.5, h=0.5)
        nt.assert_array_almost_equal(x, [1, 1.5])
        nt.assert_array_almost_equal(y, [1, 1.8125])

    def test_RK4_one_step(self):
        # RK4, y' = y/2, y(0) = 1, h = 0.5.
        # By hand:
        #   K0 = 0.5 * 0.5 = 0.25
        #   K1 = 0.5 * (1 + 0.125)/2 = 0.28125
        #   K2 = 0.5 * (1 + 0.140625)/2 = 0.5 * 0.570313 = 0.285156
        #   K3 = 0.5 * (1 + 0.285156)/2 = 0.5 * 0.642578 = 0.321289
        #   y_1 = 1 + (0.25 + 2*0.28125 + 2*0.285156 + 0.321289)/6 ≈ 1.284025
        # Exact: e^0.25 ≈ 1.284025 — RK4 is very close!
        F = lambda _x, y: y / 2
        x, y = student.RK4(F, x0=0, y0=1, xStop=0.5, h=0.5)
        nt.assert_array_almost_equal(x, [0, 0.5])
        nt.assert_almost_equal(y[1], np.exp(0.25), decimal=5)

    def test_RK4_accuracy_vs_exact(self):
        # RK4 should track the exact solution y = e^x very closely for h = 0.1.
        # Global error for RK4 is O(h^4) ≈ 10^-4, well within 6 decimal places.
        # This test also checks that your implementation handles the full interval.
        F = lambda _x, y: y
        x, y = student.RK4(F, x0=0, y0=1, xStop=1, h=0.1)
        nt.assert_array_almost_equal(y, np.exp(x), decimal=5)

    def test_methods_accuracy_comparison(self):
        # All three methods should approximate y = e^x, but with different accuracy.
        # For h = 0.1 on [0, 1]: Euler error ~ O(h), RK2/Heun ~ O(h^2), RK4 ~ O(h^4).
        # The errors should satisfy: euler_err >> RK2_err >> RK4_err.
        F = lambda _x, y: y
        x_ref = np.arange(0, 1.1, 0.1)
        y_ref = np.exp(x_ref)

        _, y_euler = student.euler(F, 0, 1, 1, 0.1)
        _, y_rk2   = student.RK2(F,   0, 1, 1, 0.1)
        _, y_rk4   = student.RK4(F,   0, 1, 1, 0.1)

        err_euler = np.max(np.abs(y_euler - y_ref))
        err_rk2   = np.max(np.abs(y_rk2   - y_ref))
        err_rk4   = np.max(np.abs(y_rk4   - y_ref))

        self.assertGreater(err_euler, err_rk2,
            "Euler should be less accurate than RK2 for h=0.1")
        self.assertGreater(err_rk2, err_rk4,
            "RK2 should be less accurate than RK4 for h=0.1")


if __name__ == "__main__":
    unittest.main()
