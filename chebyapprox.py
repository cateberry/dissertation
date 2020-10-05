def approximate_chebyshev(order, interval, fnc):
    x = sym.Symbol('x')
    n_points = order
    approx = Chebyshev(interval[0], interval[1], order, fnc)
    u = approx.eval(x)

    coeffs = get_coeffs(u, x, n_points)

    from sympy import *
    from numpy import sqrt
    from pyProximation import Measure, OrthSystem
    # the symbolic variable
    x = Symbol('x')
    # set a limit to the order
    n = 6
    # define the measure
    D = [(-1, 1)]
    w = lambda x: 1. / sqrt(1. - x ** 2)
    M = Measure(D, w)
    S = OrthSystem([x], D, 'sympy')
    # link the measure to S
    S.SetMeasure(M)
    # set B = {1, x, x^2, ..., x^n}
    B = S.PolyBasis(n)
    # link B to S
    S.Basis(B)
    # generate the orthonormal basis
    S.FormBasis()
    # print the result
    print
    S.OrthBase
    # set f(x) = sin(x)e^x
    f = (x > 0) * x
    # extract the coefficients
    Coeffs = S.Series(f)
    # form the approximation
    f_app = sum([S.OrthBase[i] * Coeffs[i] for i in range(m)])
    print
    f_app

    return coeffs, u