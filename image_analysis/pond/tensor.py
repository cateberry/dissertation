import random
import numpy as np
from math import log
from im2col.im2col import im2col_indices, col2im_indices

try:
    from im2col.im2col_cython_float import im2col_cython_float, col2im_cython_float
    from im2col.im2col_cython_object import im2col_cython_object, col2im_cython_object

    use_cython = True
except ImportError as e:
    im2col_cython_float, col2im_cython_float = None, None
    im2col_cython_object, col2im_cython_object = None, None
    print(e)
    print('\nRun the following from the project root directory to use cython:')
    print('python setup.py build_ext --inplace\n')
    use_cython = False


def im2col(x, h_filter, w_filter, padding, strides):
    if use_cython:
        if x.dtype == np.dtype('float64'):
            return im2col_cython_float(x, h_filter, w_filter, padding, strides)
        else:
            return im2col_cython_object(x, h_filter, w_filter, padding, strides)
    else:
        return im2col_indices(x, h_filter, w_filter, padding, strides)


def col2im(x, imshape, field_height, field_width, padding, stride):
    if use_cython:
        if x.dtype == np.dtype('float64'):
            return col2im_cython_float(x, imshape[0], imshape[1], imshape[2], imshape[3],
                                       field_height, field_width, padding, stride)
        else:
            return col2im_cython_object(x, imshape[0], imshape[1], imshape[2], imshape[3],
                                        field_height, field_width, padding, stride)
    else:
        return col2im_indices(x, imshape, field_height, field_width, padding, stride)


class NativeTensor:

    def __init__(self, values):
        self.values = values

    @staticmethod
    def from_values(values):
        return NativeTensor(values)

    @property
    def size(self):
        return self.values.size

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return NativeTensor(self.values[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, NativeTensor)
        self.values[idx] = other.values

    def concatenate(self, other):
        assert isinstance(other, NativeTensor), type(other)
        return NativeTensor.from_values(np.concatenate([self.values, other.values]))

    def reveal(self):
        return self

    def unwrap(self):
        return self.values

    def __repr__(self):
        return "NativeTensor(%s)" % self.values

    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return NativeTensor.from_values(np.array([y]))
        if isinstance(y, np.ndarray): return NativeTensor.from_values(y)
        return y

    def flip(x, axis):
        x.values = np.flip(x.values, axis)
        return x

    def add(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values + y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def __iadd__(self, y):
        if isinstance(y, NativeTensor):
            self.values = self.values + y.values
        elif isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_values(self.values).add(y)
        elif isinstance(y, PrivateEncodedTensor):
            return PublicEncodedTensor.from_values(self.values).add(y)
        else:
            raise TypeError("does not support %s" % (type(y)))
        return self

    def add_at(self, indices, y):
        if isinstance(y, NativeTensor):
            np.add.at(self.values, indices, y.values)
        else:
            raise TypeError("%s does not support %s" % (type(self), type(y)))

    def sub(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values - y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values * y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        if isinstance(y, float): return NativeTensor(x.values * y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values.dot(y.values))
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def div(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values / y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __div__(x, y):
        return x.div(y)

    def __truediv__(x, y):
        return x.div(y)

    def __gt__(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values > y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def pow(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values ** y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __pow__(x, y):
        return x.pow(y)

    def square(x):
        return NativeTensor(np.power(x.values, 2))

    def transpose(x, *axes):
        return NativeTensor(x.values.transpose(*axes))

    def copy(x):
        return NativeTensor(x.values.copy())

    def neg(x):
        return NativeTensor(0 - x.values)

    def sum(x, axis=None, keepdims=False):
        return NativeTensor(x.values.sum(axis=axis, keepdims=keepdims))

    def clip(x, minimum, maximum):
        return NativeTensor.from_values(np.clip(x.values, minimum, maximum))

    def argmax(x, axis):
        return NativeTensor.from_values(x.values.argmax(axis=axis))

    def max(x, axis=None, keepdims=False):
        return NativeTensor.from_values(x.values.max(axis=axis, keepdims=keepdims))

    def min(x, axis=None, keepdims=False):
        return NativeTensor.from_values(x.values.min(axis=axis, keepdims=keepdims))

    def exp(x):
        return NativeTensor(np.exp(x.values))

    def log(x):
        # use this log to set log 0 -> -10^2
        return NativeTensor(np.ma.log(x.values).filled(-1e2))

    def inv(x):
        return NativeTensor(1. / x.values)

    def repeat(self, repeats, axis=None):
        self.values = np.repeat(self.values, repeats, axis=axis)
        return self

    def reshape(self, *shape):
        return NativeTensor(self.values.reshape(*shape))

    def expand_dims(self, axis=0):
        self.values = np.expand_dims(self.values, axis=axis)
        return self

    def im2col(x, h_filter, w_filter, padding, strides):
        return NativeTensor(im2col(x.values, h_filter, w_filter, padding, strides))

    def col2im(x, imshape, field_height, field_width, padding, stride):
        return NativeTensor(col2im(x.values, imshape, field_height, field_width, padding, stride))


DTYPE = 'object'
#Q = 2657003489534545107915232808830590043  # prime
#Q = 2658455991569831745807614120560689152  # 2**121
two_exp = 121
Q = 2 ** two_exp


# For arbitrary precision integers.
def log2(x):
    return log(x) / log(2)


# We need room for summing MAX_SUM values of MAX_DEGREE before during modulus reduction.
MAX_DEGREE = 2
MAX_SUM = 2 ** 12
assert MAX_DEGREE * log2(Q) + log2(MAX_SUM) < 256

BASE = 2
PRECISION_INTEGRAL = 4  # 5
PRECISION_FRACTIONAL = 12  # 13

# We need room for double precision before truncating.
# assert PRECISION_INTEGRAL + 2 * PRECISION_FRACTIONAL < log(Q) / log(BASE)

COMMUNICATION_ROUNDS = 0
COMMUNICATED_VALUES = 0
USE_SPECIALIZED_TRIPLE = False
REUSE_MASK = False


def encode(rationals):
    return (rationals * BASE ** PRECISION_FRACTIONAL).astype('int').astype(DTYPE) % Q


def decode(elements):
    map_negative_range = np.vectorize(lambda element: element if element <= Q / 2 else element - Q)  #, otypes=['int64'])
    return map_negative_range(elements) / BASE ** PRECISION_FRACTIONAL


def wrap_if_needed(y):
    if isinstance(y, int) or isinstance(y, float): return PublicEncodedTensor.from_values(np.array([y]))
    if isinstance(y, np.ndarray): return PublicEncodedTensor.from_values(y)
    if isinstance(y, NativeTensor): return PublicEncodedTensor.from_values(y.values)
    return y


class PublicEncodedTensor:

    def __init__(self, values, elements=None):
        if values is not None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            elements = encode(values)
        assert isinstance(elements, np.ndarray), "%s, %s, %s" % (values, elements, type(elements))
        self.elements = elements

    @staticmethod
    def from_values(values):
        return PublicEncodedTensor(values)

    @staticmethod
    def from_elements(elements):
        return PublicEncodedTensor(None, elements)

    def __repr__(self):
        return "PublicEncodedTensor(%s)" % decode(self.elements)

    def __getitem__(self, index):
        return PublicEncodedTensor.from_elements(self.elements[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, PublicEncodedTensor)
        self.elements[idx] = other.elements

    def concatenate(self, other):
        if isinstance(other, PublicEncodedTensor):
            return PublicEncodedTensor.from_elements(np.concatenate([self.elements, other.elements]))
        raise TypeError("%s does not support %s" % (type(self), type(other)))

    @property
    def shape(self):
        return self.elements.shape

    @property
    def size(self):
        return self.elements.size

    def copy(x):
        return PublicEncodedTensor.from_elements(x.elements.copy())

    def unwrap(self):
        return decode(self.elements)

    def reveal(self):
        return NativeTensor.from_values(decode(self.elements))

    def truncate(self, amount=PRECISION_FRACTIONAL):
        positive_numbers = (self.elements <= Q // 2).astype(int)
        elements = self.elements
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q  # x if x <= Q//2 else Q - x
        elements = np.floor_divide(elements, BASE ** amount)  # x // BASE**amount
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q  # x if x <= Q//2 else Q - x
        return PublicEncodedTensor.from_elements(elements.astype(DTYPE))

    def flip(x, axis):
        x.elements = np.flip(x.elements, axis)
        return x

    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 = y.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements((x.elements - y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor): return x.add(y.neg())
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def __gt__(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_values((x.elements - y.elements) % Q <= 0.5 * Q)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def mul(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_elements((x.elements * y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def square(x):
        return PublicEncodedTensor.from_elements(np.power(x.elements, 2) % Q).truncate()

    def dot(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements(
            x.elements.dot(y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = x.elements.dot(y.shares0) % Q
            shares1 = x.elements.dot(y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __div__(x, y):
        return x.div(y)

    def __truediv__(x, y):
        return x.div(y)

    def transpose(x, *axes):
        return PublicEncodedTensor.from_elements(x.elements.transpose(*axes))

    def sum(x, axis=None, keepdims=False):
        return PublicEncodedTensor.from_elements(x.elements.sum(axis=axis, keepdims=keepdims))

    def argmax(x, axis):
        return PublicEncodedTensor.from_values(decode(x.elements).argmax(axis=axis))

    def neg(x):
        return PublicEncodedTensor.from_values(decode(x.elements) * -1)

    def inv(x):
        return PublicEncodedTensor.from_values(1. / decode(x.elements))

    def repeat(self, repeats, axis=None):
        self.elements = np.repeat(self.elements, repeats, axis=axis)
        return self

    def reshape(self, *shape):
        return PublicEncodedTensor.from_elements(self.elements.reshape(*shape))

    def expand_dims(self, axis=0):
        self.elements = np.expand_dims(self.elements, axis=axis)
        return self

    def im2col(x, h_filter, w_filter, padding, strides):
        return PublicEncodedTensor.from_elements(im2col(x.elements, h_filter, w_filter, padding, strides))

    def col2im(x, imshape, field_height, field_width, padding, stride):
        return PublicEncodedTensor.from_elements(
            col2im(x.elements, imshape, field_height, field_width, padding, stride))


class PublicFieldTensor:

    def __init__(self, elements):
        self.elements = elements

    @staticmethod
    def from_elements(elements):
        return PublicFieldTensor(elements)

    def __repr__(self):
        return "PublicFieldTensor(%s)" % self.elements

    def __getitem__(self, index):
        return PublicFieldTensor.from_elements(self.elements[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, PublicFieldTensor)
        self.elements[idx] = other.elements

    def copy(self):
        return PublicFieldTensor.from_elements(self.elements.copy())

    def flip(x, axis):
        x.elements = np.flip(x.elements, axis)
        return x

    @property
    def size(self):
        return self.elements.size

    @property
    def shape(self):
        return self.elements.shape

    def add(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 = y.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def mul(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements.dot(y.elements)) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements.dot(y.shares0)) % Q
            shares1 = (x.elements.dot(y.shares1)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def expand_dims(x, axis):
        x.elements = np.expand_dims(x.elements, axis=axis)
        return x

    def transpose(x, *axes):
        return PublicFieldTensor.from_elements(x.elements.transpose(*axes))

    def reshape(self, *shape):
        return PublicFieldTensor.from_elements(self.elements.reshape(*shape))

    def im2col(x, h_filter, w_filter, padding, strides):
        return PublicFieldTensor.from_elements(im2col(x.elements, h_filter, w_filter, padding, strides))

    def col2im(x, imshape, field_height, field_width, padding, stride):
        return PublicFieldTensor.from_elements(col2im(x.elements, imshape, field_height, field_width, padding, stride))

    def repeat(self, repeats, axis=None):
        self.elements = np.repeat(self.elements, repeats, axis=axis)
        return self


def share(elements):
    shares0 = np.array([random.randrange(Q) for _ in range(elements.size)]).astype(DTYPE).reshape(elements.shape)
    shares1 = ((elements - shares0) % Q).astype(DTYPE)
    return shares0, shares1


def reconstruct(shares0, shares1):
    return (shares0 + shares1) % Q


class PrivateFieldTensor:

    def __init__(self, elements, shares0=None, shares1=None):
        if elements is not None:
            shares0, shares1 = share(elements)
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (elements, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (elements, shares1, type(shares1))
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1

    @staticmethod
    def from_elements(elements):
        return PrivateFieldTensor(elements)

    @staticmethod
    def from_shares(shares0, shares1):
        return PrivateFieldTensor(None, shares0, shares1)

    def reveal(self, count_communication=True):
        if count_communication:
            global COMMUNICATION_ROUNDS, COMMUNICATED_VALUES
            COMMUNICATION_ROUNDS += 1
            COMMUNICATED_VALUES += np.prod(self.shape)
        return PublicFieldTensor.from_elements(reconstruct(self.shares0, self.shares1))

    def __repr__(self):
        return "PrivateFieldTensor(%s)" % self.reveal().elements

    def __getitem__(self, index):
        return PrivateFieldTensor.from_shares(self.shares0[index], self.shares1[index])

    def __setitem__(self, idx, other):
        if isinstance(other, PrivateFieldTensor):
            self.shares0[idx] = other.shares0
            self.shares1[idx] = other.shares1
        else:
            raise TypeError("%s does not support %s" % (type(self), type(other)))

    def copy(self):
        return PrivateFieldTensor.from_shares(self.shares0.copy(), self.shares1.copy())

    @property
    def size(self):
        return self.shares0.size

    @property
    def shape(self):
        return self.shares0.shape

    def flip(x, axis):
        x.shares0 = np.flip(x.shares0, axis)
        x.shares1 = np.flip(x.shares1, axis)
        return x

    def add(x, y):
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 = x.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def mul(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0.dot(y.elements)) % Q
            shares1 = (x.shares1.dot(y.elements)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0.dot(y.shares0)) % Q
            shares1 = (x.shares1.dot(y.shares1)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def repeat(x, repeats, axis):
        x.shares0 = np.repeat(x.shares0, repeats, axis=axis)
        x.shares1 = np.repeat(x.shares1, repeats, axis=axis)
        return x

    def expand_dims(x, axis):
        x.shares0 = np.expand_dims(x.shares0, axis=axis)
        x.shares1 = np.expand_dims(x.shares1, axis=axis)
        return x

    def transpose(x, *axes):
        return PrivateFieldTensor.from_shares(x.shares0.transpose(*axes), x.shares1.transpose(*axes))

    def reshape(self, *shape):
        return PrivateFieldTensor.from_shares(self.shares0.reshape(*shape), self.shares1.reshape(*shape))

    def conv2d(x, y, strides, padding):
        if isinstance(y, PublicFieldTensor):
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

        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def im2col(x, h_filter, w_filter, padding, strides):
        shares0 = im2col(x.shares0, h_filter, w_filter, padding, strides)
        shares1 = im2col(x.shares1, h_filter, w_filter, padding, strides)
        return PrivateFieldTensor.from_shares(shares0, shares1)

    def col2im(x, imshape, field_height, field_width, padding, stride):
        shares0 = col2im(x.shares0, imshape, field_height, field_width, padding, stride)
        shares1 = col2im(x.shares1, imshape, field_height, field_width, padding, stride)
        return PrivateFieldTensor.from_shares(shares0, shares1)


def generate_mul_triple(shape1, shape2, shares_a=None, shares_b=None):
    if shares_a is None:
        a = np.array([random.randrange(Q) for _ in range(np.prod(shape1))]).astype(DTYPE).reshape(shape1)
        shares_a = PrivateFieldTensor.from_elements(a)
    else:
        a = shares_a.reveal(count_communication=False).elements
    if shares_b is None:
        b = np.array([random.randrange(Q) for _ in range(np.prod(shape2))]).astype(DTYPE).reshape(shape2)
        shares_b = PrivateFieldTensor.from_elements(b)
    else:
        b = shares_b.reveal(count_communication=False).elements
    shares_ab = PrivateFieldTensor.from_elements((a * b) % Q)
    return shares_a, shares_b, shares_ab


def generate_dot_triple(m, n, o, shares_a=None, shares_b=None):
    if shares_a is None:
        a = np.array([random.randrange(Q) for _ in range(m * n)]).astype(DTYPE).reshape((m, n))
        shares_a = PrivateFieldTensor.from_elements(a)
    else:
        a = shares_a.reveal(count_communication=False).elements

    if shares_b is None:
        b = np.array([random.randrange(Q) for _ in range(n * o)]).astype(DTYPE).reshape((n, o))
        shares_b = PrivateFieldTensor.from_elements(b)
    else:
        b = shares_b.reveal(count_communication=False).elements

    shares_ab = PrivateFieldTensor.from_elements(np.dot(a, b))
    return shares_a, shares_b, shares_ab


def generate_conv_triple(xshape, yshape, strides, padding):
    h_filter, w_filter, d_filters, n_filters = yshape

    a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
    b = np.array([random.randrange(Q) for _ in range(np.prod(yshape))]).astype(DTYPE).reshape(yshape)

    if use_cython:
        a_col = im2col_cython_object(a, h_filter, w_filter, padding, strides)
    else:
        a_col = im2col_indices(a, field_height=h_filter, field_width=w_filter, padding=padding, stride=strides)

    b_col = b.transpose(3, 2, 0, 1).reshape(n_filters, -1)
    # c is a conv b
    c = np.dot(b_col, a_col)

    return PrivateFieldTensor.from_elements(a), PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(c), PrivateFieldTensor.from_elements(a_col)


def generate_convbw_triple(xshape, yshape, shares_a=None, shares_a_col=None):
    if shares_a is None:
        a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
        shares_a = PrivateFieldTensor.from_elements(a)
    else:
        a = shares_a.reveal(count_communication=False).elements

    if shares_a_col is None:
        a_col = a.im2col()
    else:
        a_col = shares_a_col.reveal(count_communication=False).elements

    b = np.array([random.randrange(Q) for _ in range(np.prod(yshape))]).astype(DTYPE).reshape(yshape)
    shares_b = PrivateFieldTensor.from_elements(b)
    # c is a conv backward b
    shares_c = PrivateFieldTensor.from_elements(b.dot(a_col.transpose()))

    return shares_a, shares_b, shares_c


def generate_conv_pool_bw_triple(xshape, yshape, pool_size, n_filter, shares_a=None, shares_a_col=None,
                                 shares_b=None, shares_b_expanded=None):
    if shares_a is None:
        a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
        shares_a = PrivateFieldTensor.from_elements(a)
    else:
        a = shares_a.reveal(count_communication=False).elements

    if shares_a_col is None:
        a_col = a.im2col()
    else:
        a_col = shares_a_col.reveal(count_communication=False).elements

    if shares_b is None:
        b = np.array([random.randrange(Q) for _ in range(np.prod(yshape))]).astype(DTYPE).reshape(yshape)
        shares_b = PrivateFieldTensor.from_elements(b)
    else:
        b = shares_b.reveal(count_communication=False).elements

    if shares_b_expanded is None:
        b_expanded = b.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3).transpose(1, 2, 3, 0) \
            .reshape(n_filter, -1)
        shares_b_expanded = PrivateFieldTensor.from_elements(b_expanded)
    else:
        b_expanded = shares_b_expanded.reveal(count_communication=False).elements

    # c is a conv pool backward b
    shares_c = PrivateFieldTensor.from_elements(b_expanded.dot(a_col.transpose()))

    return shares_a, shares_b, shares_c, shares_b_expanded


def generate_conv_pool_delta_triple(xshape, yshape, pool_size, n_filter, shares_a=None):
    if shares_a is None:
        a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
    else:
        a = shares_a.reveal(count_communication=False).elements
    b = np.array([random.randrange(Q) for _ in range(np.prod(yshape))]).astype(DTYPE).reshape(yshape)
    b_expanded = b.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3).transpose(1, 2, 3, 0).reshape(n_filter, -1)
    a_reshaped = a.reshape(n_filter, -1).transpose()

    shares_b = PrivateFieldTensor.from_elements(b)
    shares_b_expanded = PrivateFieldTensor.from_elements(b_expanded)
    # c is the backpropagated gradient of weights a and incoming backpropagated gradient b
    shares_c = PrivateFieldTensor.from_elements(a_reshaped.dot(b_expanded)),
    return shares_a, shares_b, shares_c, shares_b_expanded


def generate_square_triple(xshape):
    a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
    aa = np.power(a, 2) % Q
    return PrivateFieldTensor.from_elements(a), PrivateFieldTensor.from_elements(aa)


def stack(tensors, axis=-1):
    """
    Function to stack pond tensors including masks
    :param tensors: pond tensors (same type)
    :param axis: axis for stacking
    :return: stacked tensors
    """
    assert all(isinstance(t, type(tensors[0])) for t in tensors)
    if isinstance(tensors[0], NativeTensor):
        return NativeTensor(np.stack([t.values for t in tensors], axis))
    if isinstance(tensors[0], PublicEncodedTensor):
        return PublicEncodedTensor.from_elements(np.stack([t.elements for t in tensors], axis))
    if isinstance(tensors[0], PrivateEncodedTensor):
        mask, masked = None, None
        if all(t.mask is not None for t in tensors):
            mask = PrivateFieldTensor.from_shares(np.stack([t.mask.shares0 for t in tensors], axis),
                                                  np.stack([t.mask.shares1 for t in tensors], axis))
        if all(t.masked is not None for t in tensors):
            masked = PublicFieldTensor.from_elements(np.stack([t.masked.elements for t in tensors], axis))

        return PrivateEncodedTensor.from_shares(np.stack([t.shares0 for t in tensors], axis),
                                                np.stack([t.shares1 for t in tensors], axis),
                                                mask=mask, masked=masked)


class PrivateEncodedTensor:

    def __init__(self, values, shares0=None, shares1=None, mask=None, masked=None):
        if values is not None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shares0, shares1 = share(encode(values))
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (values, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (values, shares1, type(shares1))
        assert shares0.dtype == shares1.dtype
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1
        self.mask = mask
        self.masked_transformed = None
        self.masked = masked
        self.mask_transformed = None

    @staticmethod
    def from_values(values):
        return PrivateEncodedTensor(values)

    @staticmethod
    def from_elements(elements):
        shares0, shares1 = share(elements)
        return PrivateEncodedTensor(None, shares0, shares1)

    @staticmethod
    def from_shares(shares0, shares1, mask=None, masked=None):
        return PrivateEncodedTensor(None, shares0, shares1, mask, masked)

    def copy(self):
        result = PrivateEncodedTensor(None, self.shares0.copy(), self.shares1.copy())
        if self.mask is not None: result.mask = self.mask.copy()
        if self.masked is not None: result.masked = self.masked.copy()
        if self.mask_transformed is not None: result.mask_transformed = self.mask_transformed.copy()
        if self.masked_transformed is not None: result.masked_transformed = self.masked_transformed.copy()
        return result

    def __repr__(self):
        elements = (self.shares0 + self.shares1) % Q
        return "PrivateEncodedTensor(%s)" % decode(elements)

    def __getitem__(self, index):
        result = PrivateEncodedTensor.from_shares(self.shares0[index], self.shares1[index])
        if self.mask is not None:
            result.mask = self.mask[index]
        if self.masked_transformed is not None:
            result.masked_transformed = self.masked_transformed[index]
        if self.masked is not None:
            result.masked = self.masked[index]
        if self.mask_transformed is not None:
            result.mask_transformed = self.mask_transformed[index]
        return result

    def __setitem__(self, idx, other):
        if isinstance(other, PrivateEncodedTensor):
            self.shares0[idx] = other.shares0
            self.shares1[idx] = other.shares1

            if self.mask is not None and other.mask is not None:
                self.mask[idx] = other.mask
            if self.masked is not None and other.masked is not None:
                self.masked[idx] = other.masked
        else:
            raise TypeError("%s does not support %s" % (type(self), type(other)))

    def concatenate(self, other):
        if isinstance(other, PrivateEncodedTensor):
            shares0 = np.concatenate([self.shares0, other.shares0])
            shares1 = np.concatenate([self.shares1, other.shares1])
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(self), type(other)))

    @property
    def shape(self):
        return self.shares0.shape

    @property
    def size(self):
        return self.shares0.size

    def unwrap(self):
        return decode((self.shares0 + self.shares1) % Q)

    def reveal(self):
        return NativeTensor.from_values(decode((self.shares0 + self.shares1) % Q))

    def truncate(self, amount=PRECISION_FRACTIONAL):
        shares0 = np.floor_divide(self.shares0, BASE ** amount) % Q
        shares1 = (Q - (np.floor_divide(Q - self.shares1, BASE ** amount))) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    def flip(x, axis):
        x.shares0 = np.flip(x.shares0, axis)
        x.shares1 = np.flip(x.shares1, axis)
        if x.mask is not None: x.mask = x.mask.flip(axis)
        if x.masked is not None: x.masked = x.masked.flip(axis)
        if x.mask_transformed is not None: x.mask_transformed = x.mask_transformed.flip(axis)
        if x.masked_transformed is not None: x.masked_transformed = x.masked_transformed.flip(axis)
        return x

    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 = x.shares1 + np.zeros(y.elements.shape, dtype=DTYPE)  # hack to fix broadcasting
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 - y.elements) % Q
            shares1 = x.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y, precomputed=None, reuse_mask=REUSE_MASK):
        # TODO: Add counter for communication
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            a, b, alpha, beta = None, None, None, None
            if reuse_mask: a, alpha, b, beta = x.mask, x.masked, y.mask, y.masked
            if precomputed is None: precomputed = generate_mul_triple(x.shape, y.shape, shares_a=a, shares_b=b)
            a, b, ab = precomputed
            if alpha is None: alpha = (x - a).reveal()
            if beta is None: beta = (y - b).reveal()
            if reuse_mask: x.mask, x.masked, y.mask, y.masked = a, alpha, b, beta
            z = alpha.mul(beta) + alpha.mul(b) + a.mul(beta) + ab
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()

        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 * y.shares0) % Q
            shares1 = (x.shares1 * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y, precomputed=None, reuse_mask=REUSE_MASK):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            assert x.shape[-1] == y.shape[0]
            shares0 = x.shares0.dot(y.elements) % Q
            shares1 = x.shares1.dot(y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            m, n, o = x.shape[0], x.shape[1], y.shape[1]
            assert n == y.shape[0]
            a, b, alpha, beta = None, None, None, None
            if reuse_mask: a, alpha, b, bet = x.mask, x.masked, y.mask, y.masked
            if precomputed is None: precomputed = generate_dot_triple(m, n, o, a, b)
            a, b, ab = precomputed
            if alpha is None: alpha = (x - a).reveal()
            if beta is None: beta = (y - b).reveal()
            z = alpha.dot(beta) + alpha.dot(b) + a.dot(beta) + ab
            # cache masks
            if reuse_mask:
                x.mask, x.masked, y.mask, y.masked = a, alpha, b, beta

            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def dot_QG(x, y, QG=Q, precomputed=None, reuse_mask=REUSE_MASK):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            assert x.shape[-1] == y.shape[0]
            shares0 = x.shares0.dot(y.elements) % QG
            shares1 = x.shares1.dot(y.elements) % QG
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            m, n, o = x.shape[0], x.shape[1], y.shape[1]
            assert n == y.shape[0]
            a, b, alpha, beta = None, None, None, None
            if reuse_mask: a, alpha, b, bet = x.mask, x.masked, y.mask, y.masked
            if precomputed is None: precomputed = generate_dot_triple(m, n, o, a, b)
            a, b, ab = precomputed
            if alpha is None: alpha = (x - a).reveal()
            if beta is None: beta = (y - b).reveal()
            z = alpha.dot(beta) + alpha.dot(b) + a.dot(beta) + ab
            # cache masks
            if reuse_mask:
                x.mask, x.masked, y.mask, y.masked = a, alpha, b, beta

            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def square(x):
        a, aa = generate_square_triple(x.shape)
        alpha = (x - a).reveal()
        z = alpha * alpha + alpha * a + alpha * a + aa
        x.mask, x.masked = a, alpha
        return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()

    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        if isinstance(y, PrivateEncodedTensor):  # WORKS!!!
            # alpha = y.pow_range()
            w0 = (y * 2.0 - 2.9142).neg()
            e0 = (y * w0 - 1.0).neg()
            e1 = e0.square()
            e2 = e1.square()

            out = x * w0 * (e0 + 1.0) * (e1 + 1.0) * (e2 + 1.0)
            # out2 = out * 2**(- alpha)

            return out
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def sqrt(a):
        """
        Newton's method for approximation
        """
        # initial = 2**(int(alpha.values()/2))
        # initial = 0.5  # WORKS!!!
        initial = a * (-0.8099868542) + 1.787727479
        initial2 = initial.reveal().inv().values  # WORKS  # TODO: make private...

        # xn = PrivateEncodedTensor.from_values(np.array([initial2]))
        xn = PrivateEncodedTensor.from_values(initial2)
        for i in range(4):
            xn_1 = (xn + (a / xn)) / 2.0
            xn = xn_1

        return PrivateEncodedTensor.from_shares(xn.shares0, xn.shares1)

    def inv_sqrt(a):
        initial = a * (-0.8099868542) + 1.787727479
        initial2 = initial.reveal().values  # TODO: make private...

        xn = PrivateEncodedTensor.from_values(initial2)
        for i in range(4):
            xn_1 = (xn / 2.0) * ((a * xn.square()).neg() + 3.0)
            xn = xn_1

        return PrivateEncodedTensor.from_shares(xn.shares0, xn.shares1)

    def inv(self):  # WORKS - decent accuracy but error may accumulate quickly
        w0 = (self * 2 - 2.9142).neg()
        e0 = (self * w0 - 1).neg()
        e1 = e0.square()
        e2 = e1.square()

        out = w0 * (e0 + 1) * (e1 + 1) * (e2 + 1)  # ~1/self

        return PrivateEncodedTensor.from_shares(out.shares0, out.shares1)

    def __truediv__(x, y):
        return x.div(y)

    def neg(self):
        minus_one = PublicFieldTensor.from_elements(np.array([Q - 1]))
        z = self.mul(minus_one)
        return PrivateEncodedTensor.from_shares(z.shares0, z.shares1)

    def transpose(self, *axes, reuse_mask=REUSE_MASK):
        if self.mask is not None and reuse_mask:
            out = PrivateEncodedTensor.from_shares(self.shares0.transpose(*axes), self.shares1.transpose(*axes))
            if self.mask is not None: out.mask = self.mask.transpose(*axes)
            if self.masked is not None: out.masked = self.masked.transpose(*axes)
            if self.mask_transformed is not None: out.mask_transformed = self.masked_transformed.transpose(*axes)
            if self.masked_transformed is not None: out.masked_transformed = self.masked_transformed.transpose(*axes)
            return out
        else:
            return PrivateEncodedTensor.from_shares(self.shares0.transpose(*axes), self.shares1.transpose(*axes))

    def expand_dims(self, axis=0):
        self.shares0 = np.expand_dims(self.shares0, axis=axis)
        self.shares1 = np.expand_dims(self.shares1, axis=axis)

        if self.mask is not None:
            self.mask = self.mask.expand_dims(axis=axis)
        if self.masked is not None:
            self.masked = self.masked.expand_dims(axis=axis)
        return self

    def sum(self, axis, keepdims=False):
        shares0 = self.shares0.sum(axis=axis, keepdims=keepdims) % Q
        shares1 = self.shares1.sum(axis=axis, keepdims=keepdims) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    def repeat(self, repeats, axis=None):
        self.shares0 = np.repeat(self.shares0, repeats, axis=axis)
        self.shares1 = np.repeat(self.shares1, repeats, axis=axis)
        if self.mask is not None:
            self.mask = self.mask.repeat(repeats, axis=axis)
        if self.masked is not None:
            self.masked = self.masked.repeat(repeats, axis=axis)
        return self

    def reshape(self, *shape):
        return PrivateEncodedTensor.from_shares(self.shares0.reshape(*shape), self.shares1.reshape(*shape))

    def im2col(x, h_filter, w_filter, padding, strides):
        shares0 = im2col(x.shares0, h_filter, w_filter, padding, strides)
        shares1 = im2col(x.shares1, h_filter, w_filter, padding, strides)
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    def col2im(x, imshape, field_height, field_width, padding, stride):
        shares0 = col2im(x.shares0, imshape, field_height, field_width, padding, stride)
        shares1 = col2im(x.shares1, imshape, field_height, field_width, padding, stride)
        return PrivateEncodedTensor.from_shares(shares0, shares1)


ANALYTIC_STORE = []
NEXT_ID = 0


class AnalyticTensor:

    def __init__(self, values, shape=None, ident=None):
        if values is not None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shape = values.shape
        if ident is None:
            global NEXT_ID
            ident = "tensor_%d" % NEXT_ID
            NEXT_ID += 1
        self.shape = shape
        self.ident = ident

    @staticmethod
    def from_shape(shape, ident=None):
        return AnalyticTensor(None, shape, ident)

    def __repr__(self):
        return "AnalyticTensor(%s, %s)" % (self.shape, self.ident)

    def __getitem__(self, index):
        start, stop, _ = index.indices(self.shape[0])
        shape = list(self.shape)
        shape[0] = stop - start
        ident = "%s_%d,%d" % (self.ident, start, stop)
        return AnalyticTensor.from_shape(tuple(shape), ident)

    @staticmethod
    def reset():
        global ANALYTIC_STORE
        ANALYTIC_STORE = []

    @staticmethod
    def store():
        global ANALYTIC_STORE
        return ANALYTIC_STORE

    @property
    def size(self):
        return np.prod(self.shape)

    def reveal(self):
        return self

    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return AnalyticTensor.from_shape((1,))
        if isinstance(y, np.ndarray): return AnalyticTensor.from_shape(y.shape)
        return y

    def add(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('add', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __add__(x, y):
        return x.add(y)

    def sub(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('sub', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('mul', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('dot', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def div(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('div', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def neg(self):
        ANALYTIC_STORE.append(('neg', self))
        return AnalyticTensor.from_shape(self.shape)

    def transpose(self):
        ANALYTIC_STORE.append(('transpose', self))
        return self

    def sum(self, _):
        ANALYTIC_STORE.append(('sum', self))
        return AnalyticTensor.from_shape(self.shape)
