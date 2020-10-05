import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import math


def interpolation(f, psi, points):
    N = len(psi) - 1
    A = sym.zeros(N + 1, N + 1)
    b = sym.zeros(N + 1, 1)
    psi_sym = psi  # save symbolic expression
    # Turn psi and f into Python functions
    x = sym.Symbol('x')
    psi = [sym.lambdify([x], psi[i]) for i in range(N + 1)]
    f = sym.lambdify([x], f)
    for i in range(N + 1):
        for j in range(N + 1):
            A[i, j] = psi[j](points[i])
        b[i, 0] = f(points[i])
    c = A.LUsolve(b)
    # c is a sympy Matrix object, turn to list
    c = [sym.simplify(c[i, 0]) for i in range(c.shape[0])]
    u = sym.simplify(sum(c[i] * psi_sym[i] for i in range(N + 1)))
    return u, c


def least_squares(f, psi, omega):
    N = len(psi) - 1
    A = sym.zeros(N + 1, N + 1)
    b = sym.zeros(N + 1, 1)
    x = sym.Symbol('x')
    for i in range(N + 1):
        for j in range(i, N + 1):
            integrand = psi[i] * psi[j]
            I = sym.integrate(integrand, (x, omega[0], omega[1]))
            if isinstance(I, sym.Integral):
                # Could not integrate symbolically, fall back
                # on numerical integration with mpmath.quad
                integrand = sym.lambdify([x], integrand)
                I = sym.mpmath.quad(integrand, [omega[0], omega[1]])
            A[i, j] = A[j, i] = I
        integrand = psi[i] * f
        integrand = sym.sympify(integrand)
        # I = sym.integrate(integrand, (x, omega[0], omega[1]))
        I = sym.N(sym.Integral(integrand, (x, omega[0], omega[1])))
        if isinstance(I, sym.Integral):
            integrand = sym.lambdify([x], integrand)
            I = sym.mpmath.quad(integrand, [omega[0], omega[1]])
        b[i, 0] = I
    c = A.LUsolve(b)
    c = [sym.simplify(c[i, 0]) for i in range(c.shape[0])]
    u = sum(c[i] * psi[i] for i in range(len(psi)))
    return u, c


def comparison_plot(f, coeffs, omega, order):
    # x = sym.Symbol('x')
    # u = sym.lambdify([x], u, modules="numpy")
    resolution = 401  # no of points in plot
    xcoor = np.linspace(omega[0], omega[1], resolution)
    exact = f(xcoor)
    # approx = u(xcoor)
    # Add functionality to compute polynomial with coeffs as coefficients over xcoor
    # all highest power first ?

    orders = list(range(order+1))
    o = orders[::-1]
    approx = np.zeros(resolution)
    for i in range(order+1):
        ap = coeffs[i]*(xcoor**o[i])
        approx = approx + ap

    plt.plot(xcoor, approx)
    plt.plot(xcoor, exact)
    plt.legend(['approximation', 'exact'])
    plt.show()


def get_coeffs(sym_exp, x, order):
    """Get the coefficients of a sympy generated polynomial"""
    coeff_list = []
    orders = [1, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7]
    i = 0
    expression = sym_exp.expand()
    while i < len(orders):
        coeff_list.append(expression.as_coefficients_dict()[orders[i]])
        i += 1

    coeffs = np.array(coeff_list[:order + 1])
    coeffs = coeffs[::-1]

    return coeffs


# from chebyshev.approximation import Approximation
def approximate_chebyshev(order, interval, fnc):
    x = sym.Symbol('x')
    n_points = order
    approx = Chebyshev(interval[0], interval[1], order, fnc)
    u = approx.eval(x)

    coeffs = get_coeffs(u, x, n_points)

    return coeffs, u


class Chebyshev:
    """
    Chebyshev(a, b, n, func)
    Given a function func, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.
    """

    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                             for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a, b = self.a, self.b
        # assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)
        for cj in self.c[-2:0:-1]:  # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]
