import numpy as np
import sys
from datetime import datetime, timedelta
from functools import reduce
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor, stack, USE_SPECIALIZED_TRIPLE, \
    REUSE_MASK, generate_conv_triple, generate_convbw_triple, generate_conv_pool_bw_triple, \
    generate_conv_pool_delta_triple
import math
import time
import pond
import chebyshev
from scipy.interpolate import approximate_taylor_polynomial
import pickle
import matplotlib.pyplot as plt
import sympy as sym
import tensorflow as tf
import torch
import math
from collections import namedtuple
import torch.nn.functional as F
import torch.nn as nn


class Layer:
    pass


class Dense(Layer):

    def __init__(self, num_nodes, num_features, initial_scale=.01, l2reg_lambda=0.0):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.initial_scale = initial_scale
        self.l2reg_lambda = l2reg_lambda
        self.weights = None
        self.bias = None
        self.initializer = None
        self.cache = None
        self.quant_weights = None
        self.quant_bias = None
        self.fc = True
        self.collect_stats = True
        self.name = 'fc'

    def initialize(self, input_shape, initializer=None, **_):
        if initializer is not None:
            self.weights = initializer(np.random.randn(self.num_features, self.num_nodes) * self.initial_scale)
            self.bias = initializer(np.zeros((1, self.num_nodes)))
        output_shape = [input_shape[0]] + [self.num_nodes]
        return output_shape

    def quantize(self):
        # quanted_weights = quant_weights_array(self.weights)
        # self.quant_weights = quanted_weights
        #
        # quanted_bias = quant_weights_array(self.bias)
        # self.quant_bias = quanted_bias
        pass

    def forward(self, x, predict=False):
        if predict is False:
            x = x.reshape(x.shape[0], -1)
            y = x.dot(self.weights) + self.bias
            self.cache = x
            return y
        else:
            # y = x.dot(self.quant_weights) + self.quant_bias
            # return y
            y = x.dot(self.weights) + self.bias
            self.cache = x
            return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_x = d_y.dot(self.weights.transpose())
        d_weights = x.transpose().dot(d_y)
        if self.l2reg_lambda > 0:
            d_weights = d_weights + self.weights * (self.l2reg_lambda / x.shape[0])

        d_bias = d_y.sum(axis=0)
        # update weights and bias
        self.weights = (d_weights * learning_rate).neg() + self.weights
        self.bias = (d_bias * learning_rate).neg() + self.bias

        return d_x


class BatchNorm(Layer):
    def __init__(self):
        self.cache = None
        self.epsilon = 2 ** -10
        self.beta = None
        self.gamma = None
        self.moving_mean = None
        self.moving_var = None
        self.initializer = None
        self.collect_stats = False
        self.name = 'bn'

    def initialize(self, input_shape, initializer=None, **_):
        if initializer is not None:
            weight_shape = [input_shape[-1], ]
            self.gamma = initializer(np.ones(weight_shape))  # initialise scale parameter at 1
            self.beta = initializer(np.zeros(weight_shape))  # initialise shift parameter at 0
            # self.moving_mean = initializer(np.zeros(weight_shape))
            # self.moving_var = initializer(np.zeros(weight_shape))  # TODO: cost/benefit analysis
        return input_shape  # shape doesn't change after BN

    def quantize(self):
        pass

    def forward(self, x, predict=False):
        """
        Batch norm forward pass
        """

        denom = 1 / x.shape[1]  # mean in channel dimension
        batch_mean = x.sum(axis=1, keepdims=True) * denom  # tested with Tensors and seems like it works
        # self.moving_mean = batch_mean * 0.1 + self.moving_mean * 0.9  # for use in prediction

        batch_var_sum = (x - batch_mean).square()
        batch_var = batch_var_sum.sum(axis=1, keepdims=True) * denom  # tested against plaintext and same to 3dp
        # self.moving_var = batch_var * 0.1 + self.moving_var * 0.9

        numerator = x - batch_mean
        denominator = (batch_var + self.epsilon).sqrt()
        denom_inv = denominator.inv()
        norm = numerator * denom_inv
        bn_approx = norm * self.gamma + self.beta
        self.cache = numerator, denominator, norm, self.gamma, denom_inv

        return bn_approx

    def backward(self, d_y, learning_rate):  # dy here is dL/dy = dL/bn_approx
        N, D = d_y.shape[-2], d_y.shape[-1]

        numerator, denominator, norm, gamma, denom_inv = self.cache
        d_gamma = (d_y * norm).sum(axis=1, keepdims=True)  # dL/dgamma = dL/dy * dy/dgamma
        d_beta = d_y.sum(axis=1, keepdims=True)

        d_norm = d_y * gamma
        # For debugging:
        # d_x1 = d_norm * N
        # d_x2 = d_norm.sum(axis=1, keepdims=True)  # .expand_dims(axis=1)
        # d_x3 = norm * (d_norm * norm).sum(axis=1, keepdims=True)  # .expand_dims(axis=1)
        # d_x4 = denominator * (d_x1 - d_x2 - d_x3)
        # d_x = d_x4.inv() * (1 / N)

        d_x = (denominator * (d_norm * N - d_norm.sum(axis=1, keepdims=True) - norm * (d_norm * norm).sum(axis=1,
                                                                                                          keepdims=True))).inv() * (
                      1 / N)

        self.gamma = (d_gamma * learning_rate).neg() + self.gamma
        self.beta = (d_beta * learning_rate).neg() + self.beta

        return d_x


class SigmoidExact(Layer):
    def __init__(self):
        self.cache = None

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        y = (x.neg().exp() + 1).inv()
        self.cache = y
        return y

    def backward(self, d_y, *_):
        y = self.cache
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class Sigmoid(Layer):

    def __init__(self):
        self.cache = None

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        w0 = 0.5
        w1 = 0.2159198015
        w3 = -0.0082176259
        w5 = 0.0001825597
        w7 = -0.0000018848
        w9 = 0.0000000072

        x2 = x * x
        x3 = x2 * x
        x5 = x2 * x3
        x7 = x2 * x5
        x9 = x2 * x7
        y = x9 * w9 + x7 * w7 + x5 * w5 + x3 * w3 + x * w1 + w0

        self.cache = y
        return y

    def backward(self, d_y, _):
        y = self.cache
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class SoftmaxStable(Layer):

    def __init__(self):
        self.cache = None
        self.collect_stats = False
        self.name = 'softmax'
        pass

    def initialize(self, input_shape, **_):
        self.cache = None
        return input_shape

    def quantize(self):
        pass

    def forward(self, x, predict=False):
        # we add the - x.max() for numerical stability, i.e. to prevent overflow
        likelihoods = (x - x.max(axis=1, keepdims=True)).clip(-10.0, np.inf).exp()
        probs = likelihoods.div(likelihoods.sum(axis=1, keepdims=True))
        self.cache = probs
        return probs

    def backward(self, d_probs, _):
        probs = self.cache
        batch_size = probs.shape[0]
        d_scores = probs - d_probs
        d_scores = d_scores.div(batch_size)
        return d_scores


class Softmax(Layer):

    def __init__(self):
        self.cache = None
        pass

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def quantize(self):
        pass

    def forward(self, x, predict=False):
        exp = x.exp()
        probs = exp.div(exp.sum(axis=1, keepdims=True))
        self.cache = probs
        return probs

    def backward(self, d_probs, _):
        probs = self.cache
        batch_size = probs.shape[0]
        d_scores = probs - d_probs
        d_scores = d_scores.div(batch_size)
        return d_scores


class ReluExact(Layer):

    def __init__(self):
        self.cache = None

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        y = x * (x > 0)
        self.cache = x
        return y

    def backward(self, d_y, _):
        x = self.cache
        d_x = (x > 0) * d_y
        return d_x


class Relu(Layer):

    def __init__(self, order=3, domain=(-1, 1), n=1000):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.coeff = NativeTensor(self.compute_coefficients_relu(order, domain, n))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]
        self.initializer = None
        self.collect_stats = False
        self.name = 'relu'
        assert order > 2

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def quantize(self):
        # quanted_coeffs = quant_weights_MPC(self.coeff)
        # self.quant_coeff = quanted_coeffs
        #
        # quanted_coeff_der = quant_weights_MPC(self.coeff_der)
        # self.quant_coeff_der = quanted_coeff_der
        pass

    def forward(self, x, predict=False):
        if predict is False:
            self.initializer = type(x)
            n_dims = len(x.shape)
            powers = [x, x.square()]
            for i in range(self.order - 2):
                powers.append(x * powers[-1])
            # stack list into tensor
            forward_powers = stack(powers).flip(axis=n_dims)
            y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # cache all powers except the last
            self.cache = stack(powers[:-1]).flip(axis=n_dims)
            return y
        else:
            # self.initializer = type(x)
            # n_dims = len(x.shape)
            # powers = [x, x.square()]
            # for i in range(self.order - 2):
            #     powers.append(x * powers[-1])
            # forward_powers = stack(powers).flip(axis=n_dims)
            # y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # return y
            self.initializer = type(x)
            n_dims = len(x.shape)
            powers = [x, x.square()]
            for i in range(self.order - 2):
                powers.append(x * powers[-1])
            # stack list into tensor
            forward_powers = stack(powers).flip(axis=n_dims)
            y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # cache all powers except the last
            self.cache = stack(powers[:-1]).flip(axis=n_dims)
            return y

    def backward(self, d_y, _):
        # the powers of the forward phase: x^1 ...x^order-1
        powers = self.cache
        c = d_y * self.coeff_der[-1]
        d_y.expand_dims(axis=-1)
        d_x = (d_y * powers).dot(self.coeff_der[:-1]) + c
        return d_x

    @staticmethod
    def compute_coefficients_relu(order, domain, n):
        assert domain[0] < 0 < domain[1]
        x = np.linspace(domain[0], domain[1], n)
        y = (x > 0) * x
        return np.polyfit(x, y, order)


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


def least_squares2(f, psi, omega):
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
        I = sym.integrate(integrand, (x, omega[0], omega[1]))
        if isinstance(I, sym.Integral):
            integrand = sym.lambdify([x], integrand)
            I = sym.mpmath.quad(integrand, [omega[0], omega[1]])
        b[i, 0] = I
    c = A.LUsolve(b)
    c = [sym.simplify(c[i, 0]) for i in range(c.shape[0])]
    u = sum(c[i] * psi[i] for i in range(len(psi)))
    return u, c


# def least_squares1(f, psi, Omega):
#     N = len(psi) - 1
#     A = sym.zeros(N+1, N+1)
#     b = sym.zeros(N+1, 1)
#     x = sym.Symbol('x')
#     for i in range(N+1):
#         for j in range(i, N+1):
#             integrand = psi[i]*psi[j]
#             I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
#             if isinstance(I, sym.Integral):
#                 # Could not integrate symbolically, fall back
#                 # on numerical integration with mpmath.quad
#                 integrand = sym.lambdify([x], integrand)
#                 I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
#             A[i, j] = A[j, i] = I
#             integrand = psi[i]*f  # TODO: look at source to figure out how to fix this (f is a fn in my case)
#         I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
#         if isinstance(I, sym.Integral):
#             integrand = sym.lambdify([x], integrand)
#             I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
#         b[i, 0] = I
#     c = A.LUsolve(b)
#     c = [sym.simplify(c[i, 0]) for i in range(c.shape[0])]
#     u = sum(c[i]*psi[i] for i in range(len(psi)))
#     return u, c


def Lagrange_polynomial(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k]) / (points[i] - points[k])
    return p


def Chebyshev_nodes(a, b, N):
    from math import cos, pi
    return [0.5 * (a + b) + 0.5 * (b - a) * cos((float(2 * i + 1) / (2 * N + 1)) * pi) for i in range(N + 1)]


def Lagrange_polynomials(x, N, omega, point_distribution='uniform'):
    if point_distribution == 'uniform':
        if isinstance(x, sym.Symbol):
            # h = sym.Rational(omega[1] - omega[0], N)
            h = (omega[1] - omega[0]) / float(N)
        else:
            h = (omega[1] - omega[0]) / float(N)
        points = [omega[0] + i * h for i in range(N + 1)]
    elif point_distribution == 'chebyshev':
        points = Chebyshev_nodes(omega[0], omega[1], N)
    psi = [Lagrange_polynomial(x, i, points) for i in range(N + 1)]
    return psi, points


def comparison_plot(f, u, Omega):
    x = sym.Symbol('x')
    u = sym.lambdify([x], u, modules="numpy")
    resolution = 401  # no of points in plot
    xcoor = np.linspace(Omega[0], Omega[1], resolution)
    exact = f(xcoor)
    approx = u(xcoor)
    plt.plot(xcoor, approx)

    plt.plot(xcoor, exact)
    plt.legend(['approximation', 'exact'])
    plt.show()


def approximate_lagrange(order, point_dist='uniform', method='interpolation'):
    x = sym.Symbol('x')
    f = sym.Max(x, 0)
    f2 = lambda y: (y > 0) * y
    n_points = order
    omega = [-1, 1]
    psi, points = Lagrange_polynomials(x, n_points, omega, point_distribution=point_dist)
    print(psi, points)
    if method == 'interpolation':
        u, c = interpolation(f, psi, points)
    else:
        u, c = least_squares2(f, psi, omega)

    coeff_list = []
    order = [1, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7]
    i = 0
    expression = u.expand()
    while i < len(order):
        coeff_list.append(expression.as_coefficients_dict()[order[i]])
        i += 1

    coeffs = np.array(coeff_list[:n_points + 1])
    comparison_plot(f2, u, omega)
    return coeffs


from chebyshev.approximation import Approximation


def approximate_chebyshev(order, interval):
    # x = sym.Symbol('x')
    f = sym.Max(x, 0)
    polynomial_degree = order
    taylor_degree = 20
    point = 0
    approx = Approximation(f, interval, polynomial_degree, taylor_degree, point)
    coeffs = approx.coeffs

    return coeffs


# class Chebyshev:
#     """
#     Chebyshev(a, b, n, func)
#     Given a function func, lower and upper limits of the interval [a,b],
#     and maximum degree n, this class computes a Chebyshev approximation
#     of the function.
#     Method eval(x) yields the approximated function value.
#     """
#     def __init__(self, a, b, n, func):
#         self.a = a
#         self.b = b
#         self.func = func
#
#         bma = 0.5 * (b - a)
#         bpa = 0.5 * (b + a)
#         f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
#         fac = 2.0 / n
#         self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
#                   for k in range(n)]) for j in range(n)]
#
#     def eval(self, x):
#         a,b = self.a, self.b
#         assert(a <= x <= b)
#         y = (2.0 * x - a - b) * (1.0 / (b - a))
#         y2 = 2.0 * y
#         (d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
#         for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
#             (d, dd) = (y2 * d - dd + cj, d)
#         return y * d - dd + 0.5 * self.c[0]   # Last step is different


class ReluNormal(Layer):

    def __init__(self, order, mu=0.0, sigma=1.0, n=1000, approx_type='regression'):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.saved_coeffs = []
        self.coeff = NativeTensor(self.compute_coeffs_normal(order, mu, sigma, n, approx_type))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]
        self.initializer = None
        self.quant_coeff = None
        self.quant_coeff_der = None
        assert order > 2

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def quantize(self):
        # quanted_coeffs = quant_weights_array(self.coeff)
        # self.quant_coeff = quanted_coeffs
        #
        # quanted_coeff_der = quant_weights_array(self.coeff_der)
        # self.quant_coeff_der = quanted_coeff_der
        pass

    def forward(self, x, predict=False):
        if predict is False:
            self.initializer = type(x)
            n_dims = len(x.shape)
            powers = [x, x.square()]
            for i in range(self.order - 2):
                powers.append(x * powers[-1])
            # stack list into tensor
            forward_powers = stack(powers).flip(axis=n_dims)
            y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # cache all powers except the last
            self.cache = stack(powers[:-1]).flip(axis=n_dims)
            return y
        else:
            # self.initializer = type(x)
            # n_dims = len(x.shape)
            # powers = [x, x.square()]
            # for i in range(self.order - 2):
            #     powers.append(x * powers[-1])
            # forward_powers = stack(powers).flip(axis=n_dims)
            # y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # return y
            self.initializer = type(x)
            n_dims = len(x.shape)
            powers = [x, x.square()]
            for i in range(self.order - 2):
                powers.append(x * powers[-1])
            # stack list into tensor
            forward_powers = stack(powers).flip(axis=n_dims)
            y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]
            # cache all powers except the last
            self.cache = stack(powers[:-1]).flip(axis=n_dims)
            return y

    def backward(self, d_y, _):
        # the powers of the forward phase: x^1 ...x^order-1
        powers = self.cache
        c = d_y * self.coeff_der[-1]
        d_y.expand_dims(axis=-1)
        d_x = (d_y * powers).dot(self.coeff_der[:-1]) + c
        return d_x

    @staticmethod
    def relu(x):
        y = (x > 0) * x
        return y

    def compute_coeffs_normal(self, order, mu, sigma, n, approx_type='regression', n_points=3):
        # Sample x from normal distribution
        x = np.random.normal(mu, sigma, n)

        if approx_type == 'regression':
            # Fit a polynomial to the sample data (least squares)
            y = self.relu(x)
            coeffs = np.polyfit(x, y, order)
        elif approx_type == 'taylor':
            # Fit a Taylor polynomial
            taylor_approx = approximate_taylor_polynomial(self.relu, 0, order, 3)  # TODO: experiment with scale
            coeffs = taylor_approx.coeffs
        elif approx_type == 'lagrange-uniform-interpolate':  # TODO: experiment with parameters
            # Fit a Lagrange polynomial with equidistant points over interval
            coeffs = approximate_lagrange(order, point_dist='uniform', method='interpolation')
        elif approx_type == 'lagrange-uniform-leastsquares':
            coeffs = approximate_lagrange(order, point_dist='uniform', method='leastsquares')
        elif approx_type == 'lagrange-chebyshev':  # TODO: test chebyshev
            # Fit a Lagrange polynomial with Chebyshev points to counteract oscillations
            coeffs = approximate_lagrange(order, point_dist='chebyshev', method='interpolation')
        elif approx_type == 'lagrange-chebyshev':  # TODO: experiment with number of points
            # Fit a Lagrange polynomial with Chebyshev points to counteract oscillations
            coeffs = approximate_lagrange(order, point_dist='chebyshev', method='leastsquares')
        elif approx_type == 'chebyshev':
            interval = (-1, 1)
            coeffs = approximate_chebyshev(order, interval)
        else:
            pass

        print(coeffs)

        return coeffs


class ReluGalois(Layer):

    def __init__(self, order, mu=0.0, sigma=1.0, n=1000):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.saved_coeffs = []
        self.coeff = NativeTensor(self.compute_coeffs_galois(order, mu, sigma, n))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]
        self.initializer = None
        assert order > 2

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        self.initializer = type(x)

        n_dims = len(x.shape)

        powers = [x, x.square()]
        for i in range(self.order - 2):
            powers.append(x * powers[-1])

        # try:
        #     x_4_1 = powers[3] + 1
        # except:
        #     x_4_1 = x * powers[2] + 1

        # stack list into tensor
        forward_powers = stack(powers).flip(axis=n_dims)
        # y = forward_powers.dot_QG(self.coeff[:-1], QG=x_4_1) + self.coeff[-1]
        y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]

        # quantize to 8-bit ints
        # in_range = max(y) - min(y)
        # out_range = 255
        # slope = out_range / in_range
        # y = np.round(- 127 + slope * (y - min(y))).astype(np.int8)

        # cache all powers except the last
        self.cache = stack(powers[:-1]).flip(axis=n_dims)
        return y

    def backward(self, d_y, _):
        # the powers of the forward phase: x^1 ...x^order-1
        powers = self.cache
        c = d_y * self.coeff_der[-1]
        d_y.expand_dims(axis=-1)
        d_x = (d_y * powers).dot(self.coeff_der[:-1]) + c
        return d_x

    def compute_coeffs_galois(self, order, mu, sigma, n):
        x = np.random.normal(mu, sigma, n)
        y = (x > 0) * x
        coeffs = np.polyfit(x, y, order)

        # Map x to [-127, 127]
        in_range = max(coeffs) - min(coeffs)
        out_range = 255
        slope = out_range / in_range
        normed = np.round(- + slope * (coeffs - min(coeffs))).astype(np.uint8)
        # TODO: try uint8?
        # self.saved_coeffs.append(coeffs)
        print(coeffs)
        print(normed)

        return normed


class Square(Layer):
    def __init__(self):
        self.cache = None

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        y = x.square()
        self.cache = x
        return y

    def backward(self, d_y, _):
        x = self.cache
        d_x = (x * 2) * d_y  # dy/dx * dL/dy
        return d_x


class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x):
        pass

    def backward(self, dx):
        pass


class Flatten(Layer):
    def __init__(self):
        self.shape = None
        self.collect_stats = False
        self.name = 'flatten'

    @staticmethod
    def initialize(input_shape, **_):
        return [input_shape[0]] + [np.prod(input_shape[1:])]

    def quantize(self):
        pass

    def forward(self, x, predict=False):
        self.shape = x.shape
        y = x.reshape(x.shape[0], -1)
        return y

    def backward(self, d_y, _):
        return d_y.reshape(self.shape)


class Conv2D:
    def __init__(self, fshape, strides=1, padding=0, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 l2reg_lambda=0.0, channels_first=True):
        """ 2 Dimensional convolutional layer, expects NCHW data format
            fshape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example: Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01,
            size=shp))

            shape = HWCN (N=output channels)
        """
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.filter_init = filter_init
        self.l2reg_lambda = l2reg_lambda
        self.cache = None
        self.cached_x_col = None
        self.cached_input_shape = None
        self.initializer = None
        self.weights = None
        self.bias = None
        self.model = None
        self.quant_filters = None
        self.quant_bias = None
        self.scales = None
        self.zero_points = None
        self.conv = True
        self.collect_stats = True
        self.name = 'conv1'
        assert channels_first

    def initialize(self, input_shape, model=None, initializer=None):
        self.model = model

        h_filter, w_filter, d_filters, n_filters = self.fshape
        n_x, d_x, h_x, w_x = input_shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

        self.bias = initializer(np.zeros((n_filters, h_out, w_out)))
        self.weights = initializer(self.filter_init(self.fshape))

        return [n_x, n_filters, h_out, w_out]

    def quantize(self):
        quanted_filters, scales, zero_points = quant_weights_filters2(self.weights)
        self.quant_filters = PrivateEncodedTensor(quanted_filters.astype(object))
        self.scales = scales
        self.zero_points = zero_points

        # quanted_bias = quant_weights_bias(self.bias)
        # self.quant_bias = quanted_bias

    # def dequantize(self, out):
    #     """out is PrivateEncodedTensor, scales and zero_points are float/int"""
    #     dequanted_out = dequant_weights_filters2(out, self.scales, self.zero_points)
    #
    #     return dequanted_out

    def forward(self, x, predict=False):
        if predict is False:
            self.cached_input_shape = x.shape
            self.cache = x
            out, self.cached_x_col = conv2d(x, self.weights, self.strides, self.padding)

            return out + self.bias
        else:
            out, X_col = conv2d(x, self.quant_filters, self.strides, self.padding)

            out_q = out + self.bias
            out_deq = self.dequantize(out_q)

            return out_deq

    def backward(self, d_y, learning_rate):
        x = self.cache
        h_filter, w_filter, d_filter, n_filter = self.weights.shape
        dx = None

        if self.model.layers.index(self) != 0:
            W_reshaped = self.weights.reshape(n_filter, -1).transpose()
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dx = W_reshaped.dot(dout_reshaped).col2im(imshape=self.cached_input_shape, field_height=h_filter,
                                                      field_width=w_filter, padding=self.padding, stride=self.strides)

        d_w = conv2d_bw(x, d_y, self.cached_x_col, self.weights.shape, padding=self.padding, strides=self.strides)
        d_bias = d_y.sum(axis=0)

        if self.l2reg_lambda > 0:
            d_w = d_w + self.weights * (self.l2reg_lambda / self.cached_input_shape[0])

        self.weights = (d_w * learning_rate).neg() + self.weights
        self.bias = (d_bias * learning_rate).neg() + self.bias

        return dx


class AveragePooling2D:

    def __init__(self, pool_size, strides=None, channels_first=True):
        """ Average Pooling layer NCHW
            pool_size: (n x m) tuple
            strides: int with stride size
            Example: AveragePooling2D(pool_size=(2,2))
        """
        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        self.cache = None
        self.initializer = None
        self.collect_stats = False
        self.name = 'avgpool'
        if strides is None:
            self.strides = pool_size[0]
        else:
            self.strides = strides

        assert channels_first

    def initialize(self, input_shape, **_):
        s = (input_shape[2] - self.pool_size[0]) // self.strides + 1
        return input_shape[:2] + [s, s]

    def quantize(self):
        pass

    def forward(self, x, predict=False):
        # forward pass of average pooling, assumes NCHW data format
        s = (x.shape[2] - self.pool_size[0]) // self.strides + 1
        self.initializer = type(x)
        pooled = self.initializer(np.zeros((x.shape[0], x.shape[1], s, s)))
        for j in range(s):
            for i in range(s):
                pooled[:, :, j, i] = x[:, :, j * self.strides:j * self.strides + self.pool_size[0],
                                     i * self.strides:i * self.strides + self.pool_size[1]].sum(axis=(2, 3))

        pooled = pooled / self.pool_area
        return pooled

    def backward(self, d_y, _):
        d_y_expanded = d_y.repeat(self.pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=3)
        d_x = d_y_expanded / self.pool_area
        return d_x


class ConvAveragePooling2D:
    def __init__(self, fshape, strides=1, padding=0, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 l2reg_lambda=0.0, pool_size=(2, 2), pool_strides=None, channels_first=True):
        """ 2 Dimensional convolutional layer followed by average pooling layer
            , expects NCHW data format and is optimized for communication
            fshape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
        """

        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.filter_init = filter_init
        self.l2reg_lambda = l2reg_lambda
        self.cache = None
        self.cache2 = None
        self.cached_input_shape = None
        self.initializer = None
        self.filters = None
        self.bias = None
        self.model = None
        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        if pool_strides is None:
            self.pool_strides = pool_size[0]
        else:
            self.pool_strides = pool_strides

        assert channels_first

    def initialize(self, input_shape, model=None, initializer=None):
        # because this layer only makes sense to optimize for communication, use_specialized_triple and reuse_mask
        # are allways set to True
        pond.tensor.USE_SPECIALIZED_TRIPLE = True
        pond.tensor.REUSE_MASK = True
        self.model = model

        h_filter, w_filter, d_filters, n_filters = self.fshape
        n_x, d_x, h_x, w_x = input_shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

        self.bias = initializer(np.zeros((n_filters, h_out, w_out)))
        self.filters = initializer(self.filter_init(self.fshape))
        s = (h_out - self.pool_size[0]) // self.pool_strides + 1

        return [n_x, n_filters, s, s]

    def forward(self, x):
        self.initializer = type(x)
        self.cached_input_shape = x.shape
        self.cache = x

        out, self.cache2 = conv2d(x, self.filters, self.strides, self.padding)
        x_pool = out + self.bias

        s = (x_pool.shape[2] - self.pool_size[0]) // self.pool_strides + 1
        pooled = self.initializer(np.zeros((x_pool.shape[0], x_pool.shape[1], s, s)))
        for j in range(s):
            for i in range(s):
                pooled[:, :, j, i] = x_pool[:, :, j * self.pool_strides:j * self.pool_strides + self.pool_size[0],
                                     i * self.pool_strides:i * self.pool_strides + self.pool_size[1]] \
                    .sum(axis=(2, 3))

        pooled = pooled / self.pool_area
        return pooled

    def backward(self, d_y, learning_rate):
        d_y_expanded = d_y.copy().repeat(self.pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=3)
        d_y_conv = d_y_expanded / self.pool_area
        dx = None
        x = self.cache

        if self.model.layers.index(self) != 0:
            dx = convavgpool_delta(d_y, self.filters, self.cached_input_shape, padding=self.padding,
                                   strides=self.strides, pool_size=self.pool_size, pool_strides=self.pool_strides)

        d_w = convavgpool_bw(x, d_y, self.cache2, self.filters.shape, pool_size=self.pool_size,
                             pool_strides=self.pool_strides)
        d_bias = d_y_conv.sum(axis=0)

        if self.l2reg_lambda > 0:
            d_w = d_w + self.filters * (self.l2reg_lambda / self.cached_input_shape[0])

        self.filters = (d_w * learning_rate).neg() + self.filters
        self.bias = ((d_bias * learning_rate).neg() + self.bias)

        return dx


class Reveal(Layer):

    def __init__(self):
        self.collect_stats = False
        self.name = 'reveal'
        pass

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def quantize(self):
        pass

    @staticmethod
    def forward(x, predict=False):
        return x.reveal()

    @staticmethod
    def backward(d_y, _):
        return d_y


class Loss:
    pass


class Diff(Loss):

    @staticmethod
    def derive(y_pred, y_train):
        return y_pred - y_train


class CrossEntropy(Loss):

    @staticmethod
    def evaluate(probs_pred, probs_correct):
        batch_size = probs_pred.shape[0]
        losses = (probs_correct * probs_pred.log()).neg().sum(axis=1)
        loss = losses.sum(axis=0, keepdims=True).div(batch_size)
        return loss

    @staticmethod
    def derive(_, y_correct):
        return y_correct


class SoftmaxCrossEntropy(Loss):
    pass


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    try:
        unquant_weights = x.unwrap().astype(np.float32)  # need to convert to float
        x = torch.from_numpy(unquant_weights)
    except:
        pass
    # Convert to tensors for compatibility with Tensorflow

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def quantizeLayer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers
    # x = torch.from_numpy(x.unwrap().astype(np.float32))

    # cache old values
    W = layer.weights
    B = layer.bias

    W_t = torch.from_numpy(layer.weights.unwrap().astype(np.float32))
    b_t = torch.from_numpy(layer.bias.unwrap().astype(np.float32))

    # quantise weights, activations are already quantised
    w = quantize_tensor(W_t)
    b = quantize_tensor(b_t)

    layer.weights = w.tensor.float()
    layer.bias = b.tensor.float()

    # This is Quantisation arithmetic
    scale_w = w.scale  # float tensors
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Preparing input by shifting
    X = x.float() - zp_x
    layer.weights = scale_x * scale_w * (layer.weights - zp_w)
    layer.weights = NativeTensor(layer.weights.numpy().astype(object))
    layer.bias = scale_b * (layer.bias + zp_b)
    layer.bias = NativeTensor(layer.bias.numpy().astype(object))

    X2 = NativeTensor(X.numpy().astype(object))
    # All unencrypted computation
    scale_next2 = NativeTensor(scale_next.numpy().astype(object))
    x = (layer.forward(X2) / scale_next2) + zero_point_next

    # Reset weights for next forward pass
    layer.weights = W
    layer.bias = B

    return x, scale_next, zero_point_next


# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    return stats


# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(x, stats, key):
    x2 = torch.from_numpy(x.unwrap().astype(np.float32))
    stats = updateStats(x2.clone().view(x2.shape[0], -1), stats, key)

    return stats


def conv2d(x, y, strides, padding, precomputed=None, save_mask=True):
    if isinstance(x, NativeTensor) or isinstance(x, PublicEncodedTensor):
        # shapes, assuming NCHW
        h_filter, w_filter, d_filters, n_filters = y.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
        w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

        X_col = x.im2col(h_filter, w_filter, padding, strides)
        W_col = y.transpose(3, 2, 0, 1).reshape(n_filters, -1)
        out = W_col.dot(X_col)

        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        return out, X_col
    elif isinstance(x, PrivateEncodedTensor):
        h_filter, w_filter, d_y, n_filters = y.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
        w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

        if isinstance(y, PublicEncodedTensor):
            X_col = x.im2col(h_filter, w_filter, padding, strides)
            y_col = y.transpose(3, 2, 0, 1).reshape(n_filters, -1)
            out = y_col.dot(X_col).reshape(n_filters, h_out, w_out, n_x).transpose(3, 0, 1, 2)
            return out, X_col

        if isinstance(y, PrivateEncodedTensor):
            if pond.tensor.USE_SPECIALIZED_TRIPLE:
                if precomputed is None: precomputed = generate_conv_triple(x.shape, y.shape, strides, padding)

                a, b, a_conv_b, a_col = precomputed
                alpha = (x - a).reveal()
                beta = (y - b).reveal()

                alpha_col = alpha.im2col(h_filter, w_filter, padding, strides)
                beta_col = beta.transpose(3, 2, 0, 1).reshape(n_filters, -1)
                b_col = b.transpose(3, 2, 0, 1).reshape(n_filters, -1)

                alpha_conv_beta = beta_col.dot(alpha_col)
                alpha_conv_b = b_col.dot(alpha_col)
                a_conv_beta = beta_col.dot(a_col)

                z = (alpha_conv_beta + alpha_conv_b + a_conv_beta + a_conv_b).reshape(n_filters, h_out, w_out,
                                                                                      n_x).transpose(3, 0, 1, 2)
                if save_mask:
                    x.mask, x.masked, x.mask_transformed, x.masked_transformed = a, alpha, a_col, alpha_col
                    y.mask, y.masked, y.mask_transformed, y.masked_transformed = b, beta, b_col, beta_col

                return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate(), None

            else:
                X_col = x.im2col(h_filter, w_filter, padding, strides)
                W_col = y.transpose(3, 2, 0, 1).reshape(n_filters, -1)
                out = W_col.dot(X_col).reshape(n_filters, h_out, w_out, n_x).transpose(3, 0, 1, 2)
                return out, X_col

        raise TypeError("%s does not support %s" % (type(x), type(y)))
    raise TypeError("%s does not support %s" % (type(x), type(y)))


def conv2d_bw(x, d_y, x_col, filter_shape, padding=None, strides=None):
    if isinstance(x, NativeTensor) or isinstance(x, PublicEncodedTensor):
        if isinstance(d_y, NativeTensor) or isinstance(d_y, PublicEncodedTensor):
            assert x_col is not None
            h_filter, w_filter, d_filter, n_filter = filter_shape
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dw = dout_reshaped.dot(x_col.transpose())
            dw = dw.reshape(filter_shape)
            return dw
        else:
            raise TypeError("%s does not support %s" % (type(x), type(d_y)))

    elif isinstance(x, PrivateEncodedTensor):
        h_filter, w_filter, d_filter, n_filter = filter_shape
        d_y_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)

        if isinstance(d_y, PublicEncodedTensor) or isinstance(d_y, NativeTensor):
            dw = d_y_reshaped.dot(x_col.transpose())
            return dw.reshape(filter_shape)
        if isinstance(d_y, PrivateEncodedTensor):
            if pond.tensor.USE_SPECIALIZED_TRIPLE:
                if pond.tensor.USE_SPECIALIZED_TRIPLE:
                    a, a_col, alpha_col = x.mask, x.mask_transformed, x.masked_transformed
                    a, b, a_convbw_b = generate_convbw_triple(a.shape, d_y_reshaped.shape, shares_a=a,
                                                              shares_a_col=a_col)
                    beta = (d_y_reshaped - b).reveal()

                    alpha_convbw_beta = beta.dot(alpha_col.transpose())
                    alpha_convbw_b = b.dot(alpha_col.transpose())
                    a_convbw_beta = beta.dot(a_col.transpose())

                    z = (alpha_convbw_beta + alpha_convbw_b + a_convbw_beta + a_convbw_b).reshape(filter_shape)
                    return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()

                else:
                    a, b, a_convbw_b = generate_convbw_triple(x.shape, d_y_reshaped.shape)
                    alpha = (x - a).reveal()
                    beta = (d_y_reshaped - b).reveal()

                    alpha_col = alpha.im2col(h_filter, w_filter, padding, strides)
                    a_col = a.im2col(h_filter, w_filter, padding, strides)

                    alpha_convbw_beta = beta.dot(alpha_col.transpose())
                    alpha_convbw_b = b.dot(alpha_col.transpose())
                    a_convbw_beta = beta.dot(a_col.transpose())

                    z = (alpha_convbw_beta + alpha_convbw_b + a_convbw_beta + a_convbw_b).reshape(filter_shape)
                    return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
            else:
                dw = d_y_reshaped.dot(x_col.transpose())
                return dw.reshape(filter_shape)
        raise TypeError("%s does not support %s" % (type(x), type(d_y)))
    raise TypeError("%s does not support %s" % (type(x), type(d_y)))


def convavgpool_bw(x, d_y, cache, filter_shape, pool_size=None, pool_strides=None):
    h_filter, w_filter, d_filter, n_filter = filter_shape
    pool_area = pool_size[0] * pool_size[1]

    if isinstance(d_y, PublicEncodedTensor) or isinstance(d_y, NativeTensor) or isinstance(x, NativeTensor) \
            or isinstance(x, PublicEncodedTensor):
        d_y_expanded = d_y.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3)
        d_y_conv = d_y_expanded / pool_area
        X_col = cache
        d_y_conv_reshaped = d_y_conv.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dw = d_y_conv_reshaped.dot(X_col.transpose())
        return dw.reshape(filter_shape)
    if isinstance(d_y, PrivateEncodedTensor):
        assert pond.tensor.USE_SPECIALIZED_TRIPLE and pond.tensor.REUSE_MASK
        assert pool_size[0] == pool_strides and pool_size[1] == pool_strides, (pool_size, pool_strides)

        a, a_col, alpha_col = x.mask, x.mask_transformed, x.masked_transformed
        b, b_expanded, beta_expanded = d_y.mask, d_y.mask_transformed, d_y.masked_transformed

        a, b, a_conv_pool_bw_b, b_expanded = generate_conv_pool_bw_triple(a.shape, d_y.shape, pool_size=pool_size,
                                                                          n_filter=n_filter, shares_a=a,
                                                                          shares_a_col=a_col, shares_b=b,
                                                                          shares_b_expanded=b_expanded)
        if beta_expanded is None:
            beta = ((d_y / pool_area) - b).reveal()  # divide by pool area before specialized triplet
            beta_expanded = beta.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3).transpose(1, 2, 3, 0) \
                .reshape(n_filter, -1)

        alpha_conv_pool_bw_beta = beta_expanded.dot(alpha_col.transpose())
        alpha_conv_pool_bw_b = b_expanded.dot(alpha_col.transpose())
        a_conv_pool_bw_beta = beta_expanded.dot(a_col.transpose())

        z = (alpha_conv_pool_bw_beta + alpha_conv_pool_bw_b + a_conv_pool_bw_beta + a_conv_pool_bw_b
             ).reshape(filter_shape)
        return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()


def convavgpool_delta(d_y, w, cached_input_shape, padding=None, strides=None, pool_size=None, pool_strides=None):
    h_filter, w_filter, d_filter, n_filter = w.shape
    pool_area = pool_size[0] * pool_size[1]

    if isinstance(d_y, PublicEncodedTensor) or isinstance(d_y, NativeTensor):
        d_y_expanded = d_y.copy().repeat(pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(pool_size[1], axis=3)
        d_y_conv = d_y_expanded / pool_area
        W_reshape = w.reshape(n_filter, -1)
        dout_reshaped = d_y_conv.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dx_col = W_reshape.transpose().dot(dout_reshaped)
        dx = dx_col.col2im(imshape=cached_input_shape, field_height=h_filter, field_width=w_filter,
                           padding=padding, stride=strides)
        return dx
    if isinstance(d_y, PrivateEncodedTensor):
        assert pond.tensor.use_specialized_triple and pond.tensor.reuse_mask
        assert pool_size[0] == pool_strides and pool_size[1] == pool_strides

        a, alpha = w.mask, w.masked
        a, b, a_conv_pool_delta_b, b_expanded = generate_conv_pool_delta_triple(a.shape, d_y.shape, pool_size,
                                                                                n_filter, shares_a=a)

        a_reshaped = a.reshape(n_filter, -1).transpose()
        alpha_reshaped = alpha.reshape(n_filter, -1).transpose()
        beta = ((d_y / pool_area) - b).reveal()  # divide by pool area before specialized triplet
        beta_expanded = beta.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3).transpose(1, 2, 3, 0) \
            .reshape(n_filter, -1)

        alpha_conv_pool_delta_beta = alpha_reshaped.dot(beta_expanded)
        alpha_conv_pool_delta_b = alpha_reshaped.dot(b_expanded)
        a_conv_pool_delta_beta = a_reshaped.dot(beta_expanded)

        z = alpha_conv_pool_delta_beta + alpha_conv_pool_delta_b + a_conv_pool_delta_beta + a_conv_pool_delta_b
        dx_col = PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()

        d_y.mask, d_y.masked, d_y.mask_transformed, d_y.masked_transformed = b, beta, b_expanded, beta_expanded

        return dx_col.col2im(imshape=cached_input_shape, field_height=h_filter,
                             field_width=w_filter, padding=padding, stride=strides)


class DataLoader:

    def __init__(self, data, wrapper=lambda x: x):
        self.data = data
        self.wrapper = wrapper

    def batches(self, batch_size=None, shuffle_indices=None):
        if shuffle_indices is not None:
            self.data = self.data[shuffle_indices]
        if batch_size is None:
            batch_size = self.data.shape[0]
        return (
            self.wrapper(self.data[i:i + batch_size])
            for i in range(0, self.data.shape[0], batch_size)
        )

    def all_data(self):
        return self.wrapper(self.data)


class Model:
    pass


class Sequential(Model):

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

    def initialize(self, input_shape, initializer, **_):
        for layer in self.layers:
            input_shape = layer.initialize(input_shape=input_shape, initializer=initializer, model=self)

    def forward(self, x, predict=False):
        for layer in self.layers:
            x = layer.forward(x, predict)
        return x

    def quantForward(self, x, stats):
        """Weights unencrypted, data encrypted"""
        for layer in self.layers:
            if layer.name == 'conv1':
                x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
                # x_t = PrivateEncodedTensor(x.tensor.astype(object))
                #
                # x = layer.forward(x_t, True)
                x, scale_next, zero_point_next = quantizeLayer(x.tensor, layer, stats['fc'], x.scale, x.zero_point)
                #x = PrivateEncodedTensor(x.tensor.numpy().astype(object))
                #x = NativeTensor(x)
                # TODO: x is quantized anyway so why make it private? just use nativetensors
            elif layer.name == 'fc':
                x = torch.from_numpy(layer.weights.unwrap().astype(np.float32))
                x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
                x = PrivateEncodedTensor(x.numpy().astype(object))
                x = layer.forward(x, False)
            else:
                x = layer.forward(x, False)

        # return PrivateEncodedTensor(x.tensor.numpy().astype(object))
        return x

    def backward(self, d_y, learning_rate):
        for layer in reversed(self.layers):
            d_y = layer.backward(d_y, learning_rate)

    @staticmethod
    def print_progress(batch_index, n_batches, batch_size, epoch_start, train_loss=None, train_acc=None,
                       val_loss=None, val_acc=None):
        sys.stdout.write('\r')
        sys.stdout.flush()
        progress = (batch_index / n_batches)

        eta = timedelta(
            seconds=round((1. - progress) * (time.time() - epoch_start) / progress, 0)) if progress > 0 else " "
        n_eq = int(progress * 30)
        n_dot = 30 - n_eq
        progress_bar = "=" * n_eq + ">" + n_dot * "."

        if val_loss is None:
            message = "{}/{} [{}] - ETA: {} - train_loss: {:.5f} - train_acc {:.5f}"
            sys.stdout.write(message.format((batch_index + 1) * batch_size, n_batches * batch_size, progress_bar,
                                            eta, train_loss, train_acc))
        else:
            message = "{}/{} [{}] - ETA: {} - train_loss: {:.5f} - train_acc {:.5f} - val_loss {:.5f} - val_acc {:.5f}"
            sys.stdout.write(message.format((batch_index + 1) * batch_size, n_batches * batch_size, progress_bar,
                                            eta, train_loss, train_acc, val_loss, val_acc))
        sys.stdout.flush()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, loss=None, batch_size=32, epochs=1000,
            learning_rate=.01, verbose=0, eval_n_batches=None):

        if not isinstance(x_train, DataLoader): x_train = DataLoader(x_train)
        if not isinstance(y_train, DataLoader): y_train = DataLoader(y_train)

        if x_valid is not None:
            if not isinstance(x_valid, DataLoader): x_valid = DataLoader(x_valid)
            if not isinstance(y_train, DataLoader): y_valid = DataLoader(y_valid)

        n_batches = math.ceil(len(x_train.data) / batch_size)
        if eval_n_batches is None:
            eval_n_batches = n_batches

        for epoch in range(epochs):
            epoch_start = time.time()
            if verbose >= 1:
                print(datetime.now(), "Epoch {}/{}".format(epoch + 1, epochs))

            # Create batches on shuffled data
            shuffle = np.random.permutation(x_train.data.shape[0])
            batches = zip(x_train.batches(batch_size, shuffle_indices=shuffle),
                          y_train.batches(batch_size, shuffle_indices=shuffle))

            for batch_index, (x_batch, y_batch) in enumerate(batches):
                if verbose >= 2:
                    print(datetime.now(), "Batch %s" % batch_index)

                y_pred = self.forward(x_batch, False)
                train_loss = loss.evaluate(y_pred, y_batch).unwrap()[0]
                acc = np.mean(y_batch.unwrap().argmax(axis=1) == y_pred.unwrap().argmax(axis=1))
                d_y = loss.derive(y_pred, y_batch)
                self.backward(d_y, learning_rate)

                # print status
                if verbose >= 1:
                    if batch_index != 0 and (batch_index + 1) % eval_n_batches == 0:
                        # validation print
                        y_pred_val = self.predict(x_valid)
                        val_loss = np.sum(loss.evaluate(y_pred_val, y_valid.all_data()).unwrap())
                        val_acc = np.mean(
                            y_valid.all_data().unwrap().argmax(axis=1) == y_pred_val.unwrap().argmax(axis=1))
                        self.print_progress(batch_index, n_batches, batch_size, epoch_start, train_acc=acc,
                                            train_loss=train_loss,
                                            val_loss=val_loss, val_acc=val_acc)
                        print()
                    else:
                        # normal print
                        self.print_progress(batch_index, n_batches, batch_size, epoch_start, train_acc=acc,
                                            train_loss=train_loss)
        # Newline after progressbar.
        print()

    # Entry function to get stats of all functions.
    def gatherStats(self, x):
        test_loss = 0
        correct = 0
        stats = {}
        x = PrivateEncodedTensor(x.astype(object))
        for layer in self.layers:
            if layer.collect_stats is True:
                stats = gatherActivationStats(x, stats, layer.name)
                x = layer.forward(x, predict=False)
            else:
                x = layer.forward(x, predict=False)

        final_stats = {}
        for key, value in stats.items():
            final_stats[key] = {"max": value["max"] / value["total"], "min": value["min"] / value["total"]}
        return final_stats

    def predict(self, x, batch_size=32, verbose=0):
        # self.quantize()
        predict = True
        # predict = False
        #if not isinstance(x, DataLoader): x = DataLoader(x)
        all_data = x.all_data().unwrap()
        stats_data = all_data[0:64, :, :, :]  # TODO: axes probably wrong
        #stats_data = DataLoader(stats_data, wrapper=NativeTensor)
        stats = self.gatherStats(stats_data)

        val_data = all_data[64:, :, :, :]
        val_data = DataLoader(val_data, wrapper=PrivateEncodedTensor)
        batches = []
        for batch_index, x_batch in enumerate(val_data.batches(batch_size)):
            if verbose >= 2: print(datetime.now(), "Batch %s" % batch_index)
            y_batch = self.quantForward(x_batch, stats)
            batches.append(y_batch)
        return reduce(lambda x_, y: x_.concatenate(y), batches)

    def quantize(self):
        for layer in self.layers:
            layer.quantize()
