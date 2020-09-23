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

    def initialize(self, input_shape, initializer=None, **_):
        if initializer is not None:
            self.weights = initializer(np.random.randn(self.num_features, self.num_nodes) * self.initial_scale)
            self.bias = initializer(np.zeros((1, self.num_nodes)))
        output_shape = [input_shape[0]] + [self.num_nodes]
        return output_shape

    def forward(self, x, predict=False):
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
        denominator = (batch_var + self.epsilon).sqrt()  # TODO: confirmed that sqrt IS being used and works
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


class PPoly(Layer):
    def __init__(self, order=2):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.initializer = None
        self.c0 = None
        self.c1 = None
        self.c2 = None

    def initialize(self, input_shape, initializer=None, **_):
        if initializer is not None:
            x = np.random.normal(0.0, 1.0, 1000)
            # Fit a polynomial to the sample data (least squares)
            y = (x > 0) * x
            coeffs = np.polyfit(x, y, 2)

            self.c0 = initializer(coeffs[2])
            self.c1 = initializer(coeffs[1])
            self.c2 = initializer(coeffs[0])

            # self.c1 = initializer(np.random.randn(input_shape[0], input_shape[1], 1, 1))
            # self.c1 = initializer(np.random.uniform(low=-1.5, high=1.5, size=(input_shape[0], input_shape[1], 1, 1)))
            # self.c2 = initializer(np.random.randn(input_shape[0], input_shape[1], 1, 1))
            # self.c2 = initializer(np.random.uniform(low=-1.5, high=1.5, size=(input_shape[0], input_shape[1], 1, 1)))
            # TODO: initialise with relu approximation and then tune

        return input_shape

    def forward(self, x, predict=False):
        self.initializer = type(x)
        y = self.c0 + x * self.c1 + x.square() * self.c2
        self.cache = x, x.square()

        return y

    def backward(self, d_y, learning_rate):
        x, x_square = self.cache
        d_c1 = (d_y * x).sum(axis=1, keepdims=True)
        d_c2 = (d_y * x.square()).sum(axis=1, keepdims=True)
        d_x = d_y * (self.c1 + self.c2 * x * 2)
        self.c1 = (d_c1 * learning_rate).neg() + self.c1
        self.c2 = (d_c2 * learning_rate).neg() + self.c2

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
        # assert order > 2

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x, predict=False):
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


def approximate_lagrange(order, fnc, omega, point_dist='uniform', method='interpolation'):
    x = sym.Symbol('x')
    # f = sym.Max(x, 0)
    # func = lambda y: (y > 0) * y
    n_points = order
    # omega = [-1, 1]
    psi, points = Lagrange_polynomials(x, n_points, omega, point_distribution=point_dist)
    print(psi, points)
    if method == 'interpolation':
        u, c = interpolation(fnc, psi, points)
    else:
        u, c = least_squares(fnc, psi, omega)

    coeffs = get_coeffs(u, x, order)

    # coeffs = np.array(coeff_list[:n_points + 1])
    # comparison_plot(func, u, omega)
    return coeffs, u


# from chebyshev.approximation import Approximation
def approximate_chebyshev(order, interval, fnc):
    #     f = sym.Max(x, 0)
    #     polynomial_degree = order
    #     taylor_degree = 20
    #     point = 0
    #     approx = Approximation(f, interval, polynomial_degree, taylor_degree, point)
    #     coeffs = approx.coeffs
    x = sym.Symbol('x')
    n_points = order
    approx = Chebyshev(interval[0], interval[1], order, fnc)
    u = approx.eval(x)

    coeffs = get_coeffs(u, x, n_points)
    # comparison_plot(func, u, interval)
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
        (d, dd) = (self.c[-1], 0)  # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:  # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]  # Last step is different


class ReluNormal(Layer):

    def __init__(self, order, a=0.01, mu=0.0, sigma=1.0, n=1000, approx_type='regression', function='relu',
                 method='interpolation', interval=(-3, 3), point_dist='uniform', omega=[-1, 1]):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.saved_coeffs = []
        self.coeff = NativeTensor(self.compute_coeffs(order, a, mu, sigma, n, approx_type, function, method, interval,
                                                      point_dist, omega))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]
        self.initializer = None
        # assert order > 2

    @staticmethod
    def initialize(input_shape, **_):
        return input_shape

    def forward(self, x, predict=False):
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

    @staticmethod
    def sigmoid(x):
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def tanh(x):
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return y

    @staticmethod
    def leaky_relu(x, a):
        y = (x < 0) * a * x + (x >= 0) * x
        return y

    @staticmethod
    def swish(x):
        y = x * (1 / (1 + np.exp(-x)))
        return y

    @staticmethod
    def softplus(x):
        y = np.log(1 + np.exp(x))
        return y

    def compute_coeffs(self, order, a=0.01, mu=0.0, sigma=1.0, n=1000, approx_type='regression', function='relu',
                       method='interpolation', interval=(-3, 3), point_dist='uniform', omega=[-1, 1]):
        x = sym.Symbol('x')
        if function == 'relu':
            fnc_lam = lambda y: self.relu(y)
            fnc_sym = sym.Max(x, 0)
        elif function == 'sigmoid':
            fnc_lam = lambda y: self.sigmoid(y)
            fnc_sym = 1 / (1 + sym.exp(-x))
        elif function == 'tanh':
            fnc_lam = lambda y: self.tanh(y)
            fnc_sym = (sym.exp(x) - sym.exp(-x)) / (sym.exp(x) + sym.exp(-x))
        elif function == 'leaky-relu':
            fnc_lam = lambda y: self.leaky_relu(y, a)
            fnc_sym = sym.Max(x, a * x)
        elif function == 'swish':
            fnc_lam = lambda y: self.swish(y)
            fnc_sym = x * (1 / (1 + sym.exp(-x)))
        elif function == 'softplus':
            fnc_lam = lambda y: self.softplus(y)
            fnc_sym = sym.log(1 + sym.exp(x))
        else:
            pass

        if approx_type == 'regression':
            # Sample x from normal distribution
            x = np.random.normal(mu, sigma, n)
            # Fit a polynomial to the sample data (least squares)
            y = fnc_lam(x)
            coeffs = np.polyfit(x, y, order)  # returns array, highest power first
        elif approx_type == 'taylor':
            # Fit a Taylor polynomial
            taylor_approx = approximate_taylor_polynomial(fnc_lam, 0, order, 3)  # TODO: experiment with scale
            coeffs = taylor_approx.coeffs  # highest power first
        elif approx_type == 'lagrange':
            coeffs, u = approximate_lagrange(order, fnc_sym, omega, point_dist=point_dist, method=method)
            # coeffs = coeffs[::-1]
            comparison_plot(fnc_lam, u, omega)
        elif approx_type == 'chebyshev':
            # interval = (-3, 3)
            # function = lambda y: (y > 0) * y
            coeffs, u = approximate_chebyshev(order, interval, fnc_lam)
            comparison_plot(fnc_lam, u, interval)
        elif approx_type == 'chebyshev-mod':
            """Instead of approximating the ReLU function, we simulate the structure of the derivative of the ReLU 
            function, a Step function. Sigmoid function is a bounded, continuous and infinitely differentiable function,
            it also has a structure like derivative of the ReLU function in large intervals. We approximate the Sigmoid 
            function with the polynomial, calculate the integral of the polynomial, and use it as the activation 
            function"""
            # interval = (-3, 3)
            fnc = lambda y: np.exp(-1 / (np.exp(1) - 5 + y**2))
            coeffs, u = approximate_chebyshev(order, interval, fnc)
            # Integrate the polynomial approximation of sigmoid function
            x = sym.Symbol('x')
            # omega = [-8, 8]  # interval of integration
            integral = sym.integrate(u, x)
            n_points = order
            # x = sym.Symbol('x')
            coeffs = get_coeffs(integral, x, n_points)
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

            shape = HWCN (N=batchsize)
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
        self.filters = None
        self.bias = None
        self.model = None
        self.quant_filters = None
        self.quant_bias = None
        assert channels_first

    def initialize(self, input_shape, model=None, initializer=None):
        self.model = model

        h_filter, w_filter, d_filters, n_filters = self.fshape
        n_x, d_x, h_x, w_x = input_shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

        self.bias = initializer(np.zeros((n_filters, h_out, w_out)))
        self.filters = initializer(self.filter_init(self.fshape))

        return [n_x, n_filters, h_out, w_out]

    def forward(self, x, predict=False):
        self.cached_input_shape = x.shape
        self.cache = x
        out, self.cached_x_col = conv2d(x, self.filters, self.strides, self.padding)

        return out + self.bias


    def backward(self, d_y, learning_rate):
        x = self.cache
        h_filter, w_filter, d_filter, n_filter = self.filters.shape
        dx = None

        if self.model.layers.index(self) != 0:
            W_reshaped = self.filters.reshape(n_filter, -1).transpose()
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dx = W_reshaped.dot(dout_reshaped).col2im(imshape=self.cached_input_shape, field_height=h_filter,
                                                      field_width=w_filter, padding=self.padding, stride=self.strides)

        d_w = conv2d_bw(x, d_y, self.cached_x_col, self.filters.shape, padding=self.padding, strides=self.strides)
        d_bias = d_y.sum(axis=0)

        if self.l2reg_lambda > 0:
            d_w = d_w + self.filters * (self.l2reg_lambda / self.cached_input_shape[0])

        self.filters = (d_w * learning_rate).neg() + self.filters
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


def quant_weights(weights):
    maxs = np.max(weights)
    mins = np.min(weights)
    in_range = maxs - mins
    out_range = 255
    slope = out_range / in_range  # TODO: try in_range / out_range
    quanted_weights = np.round(0 + slope * (weights - mins)).astype(np.uint8)

    return quanted_weights


def quant_weights_filters(weights):
    """For NativeTensors"""
    weights = weights.unwrap()  # .astype(np.float32)  # TODO: operate on object
    mins = np.min(weights, axis=(0, 1), keepdims=True)
    maxs = np.max(weights, axis=(0, 1), keepdims=True)
    in_range = maxs - mins
    out_range = 255
    quanted_weights = np.zeros((weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))
    # slope = out_range / in_range
    slope = in_range / out_range
    for i in range(weights.shape[2]):
        for j in range(weights.shape[3]):
            int1 = (weights[:, :, i, j] - mins[:, :, i, j])
            sl1 = slope[:, :, i, j]
            int2 = 0 + (sl1 * int1)
            int3 = np.round(int2)
            quanted_weights[:, :, i, j] = int3  # .astype(np.uint8)

    # quanted_weights2 = ((weights - mins) * slope + 0).round().convert_uint8()
    return PrivateEncodedTensor(quanted_weights.astype(object))


def quant_weights_bias(weights):
    """For NativeTensors"""
    weights = weights.unwrap()  # .astype(np.float32)
    mins = np.min(weights, axis=(1, 2), keepdims=True)
    maxs = np.max(weights, axis=(1, 2), keepdims=True)
    in_range = maxs - mins
    out_range = 255
    quanted_weights = np.zeros((weights.shape[0], weights.shape[1], weights.shape[2]))
    # slope = out_range / in_range
    slope = in_range / out_range
    for i in range(weights.shape[0]):
        quanted_weights[i, :, :] = np.round(
            (0 + (slope[i, :, :] * (weights[i, :, :] - mins[i, :, :]))))  # .astype(np.uint8)
    return PrivateEncodedTensor(quanted_weights.astype(object))


def quant_weights_array(weights):
    """For NativeTensors"""
    weights = weights.unwrap()  # .astype(np.float32)
    maxs = np.max(weights)
    mins = np.min(weights)
    in_range = maxs - mins
    out_range = 255
    # quanted_weights = np.zeros(weights.shape[0])
    # slope = out_range / in_range
    slope = in_range / out_range
    quanted_weights = np.round((0 + (slope * (weights - mins))))  # .astype(np.uint8)
    return PrivateEncodedTensor(quanted_weights.astype(object))


#
# def quant_weights_tensor(weights):
#     """For NativeTensors
#     Input tensor format: NCHW
#     N is number of filters
#     Want to quantize each filter individually?
#     """
#     # Represent weights as ndarrays rather than NativeTensors
#     unquant_weights = weights.reveal().values.astype(np.float32)
#     # Convert to tensors for compatibility with Tensorflow
#     unquant_tensors = tf.convert_to_tensor(unquant_weights, dtype=tf.float32)
#
#     min_range = unquant_tensors.min()
#     max_range = unquant_tensors.max()
#     T = 'quint8'
#
#     quanted_weights = tf.quantization.quantize(
#         unquant_tensors, min_range, max_range, T, mode='MIN_COMBINED',
#         round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
#         ensure_minimum_range=0.01)
#
#     return PrivateEncodedTensor(quanted_weights.astype(object))
#
#
def quant_weights_tensor2(weights):
    from torch.quantization import QConfig, MinMaxObserver, default_observer
    """For NativeTensors
    Input tensor format: NCHW
    N is number of filters
    Want to quantize each filter individually?
    """
    # Represent weights as ndarrays rather than NativeTensors
    unquant_weights = weights.unwrap().astype(np.float32)
    # Convert to tensors for compatibility with Tensorflow
    unquant_tensors = torch.from_numpy(unquant_weights)
    """
    We want to quantize each filter in each batch separately, so we need to find
    the min and max for each filter
    """

    quant_scheme = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    calc = quant_scheme.forward(unquant_tensors)
    params = quant_scheme.calculate_qparams()
    print(params)

    quant_tensors = torch.quantize_per_tensor(unquant_tensors, scale=params[0].numpy()[0],
                                              zero_point=params[1].numpy()[0].astype(int), dtype=torch.quint8)

    # quanted_weights = np.zeros((weights.shape[0], weights.shape[1], weights.shape[2],
    #                            weights.shape[3]))
    # T = 'quint8'
    # for i in range(weights.shape[2]):
    #     for j in range(weights.shape[3]):
    #         min_range = unquant_weights[:, :, i, j].min(axis=(0, 1), keepdims=True)
    #         max_range = unquant_weights[:, :, i, j].max(axis=(0, 1), keepdims=True)
    #
    #         quanted_weights[:, :, i, j] = tf.quantization.quantize(
    #             unquant_tensors[:, :, i, j], min_range, max_range, T, mode='MIN_COMBINED',
    #             round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
    #             ensure_minimum_range=0.01)

    quanted_array = quant_tensors.int_repr().numpy()  # .astype(object)

    return PrivateEncodedTensor(quanted_array)


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

        # if isinstance(y, NativeTensor):
        #     X_col = x.im2col(h_filter, w_filter, padding, strides)
        #     y_col = y.transpose(3, 2, 0, 1).reshape(n_filters, -1)
        #     out = y_col.dot(X_col).reshape(n_filters, h_out, w_out, n_x).transpose(3, 0, 1, 2)
        #     return out, X_col

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

    # TODO: way to load weights, quantize them and initialise them to layers for prediction/testing phase
    def predict(self, x, batch_size=32, verbose=0):
        # self.quantize()
        # predict = True
        predict = False
        if not isinstance(x, DataLoader): x = DataLoader(x)
        batches = []
        for batch_index, x_batch in enumerate(x.batches(batch_size)):
            if verbose >= 2: print(datetime.now(), "Batch %s" % batch_index)
            y_batch = self.forward(x_batch, predict)
            batches.append(y_batch)
        return reduce(lambda x_, y: x_.concatenate(y), batches)

    """
    Implement saving of parameters of network (weights of conv and dense layers, relu approx parameters, batch norm
    parameters?)
        Take batch norm parameters from whole of test set? or from whole of training set?
    """

    # TODO: check time taken for evaluation to see if quantization has an effect
    def quantize(self):
        for layer in self.layers:
            layer.quantize()
