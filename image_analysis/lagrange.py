from approx_utils import get_coeffs, least_squares
import sympy as sym


def Lagrange_polynomial(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k]) / (points[i] - points[k])
    return p


def Chebyshev_nodes(a, b, N):
    from math import cos, pi
    return [0.5 * (a + b) + 0.5 * (b - a) * cos((float(2 * i + 1) / (2 * N + 1)) * pi) for i in range(N + 1)]


def Lagrange_polynomials(x, N, omega, point_distribution):
    if point_distribution == 'uniform':
        h = (omega[1] - omega[0]) / float(N)
        points = [omega[0] + i * h for i in range(N + 1)]
    elif point_distribution == 'chebyshev':
        points = Chebyshev_nodes(omega[0], omega[1], N)
    psi = [Lagrange_polynomial(x, i, points) for i in range(N + 1)]

    return psi, points


def approximate_lagrange(order, fnc_lam, fnc_sym, omega, point_dist='uniform', method='interpolation'):
    x = sym.Symbol('x')
    # f = sym.Max(x, 0)
    # func = lambda y: (y > 0) * y
    n_points = order
    # omega = [-1, 1]
    psi, points = Lagrange_polynomials(x, n_points, omega, point_distribution=point_dist)
    # print(psi, points)
    if method == 'interpolation':
        from scipy.interpolate import lagrange
        poly = lagrange(points, [fnc_lam(i) for i in points])
        # u, c = interpolation(fnc, psi, points)
        from numpy.polynomial.polynomial import Polynomial
        coeffs = Polynomial(poly).coef
        u = Polynomial(poly).coef
    else:
        u, c = least_squares(fnc_sym, psi, omega)
        coeffs = get_coeffs(u, x, order)

    return coeffs, u