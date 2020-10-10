import keras
import numpy as np
import time
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from pond.nn import Dense, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten, BatchNorm, PolyActivation, Softmax, Sigmoid, PPoly, PPolyTest
from keras.utils import to_categorical


np.random.seed(42)

# set errors error behaviour for overflow/underflow
_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')

# %%
"""
- Load data from Keras
- Split into training and test data
- Normalise to [0,1] (greyscale images with values in [0,255])
- Transform targets to one-hot representation
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = x_train[:, np.newaxis, :, :]
# x_test = x_test[:, np.newaxis, :, :]
x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

"""
Split into train-test-val
Train in usual way using train and val
Write separate function for testing on test set (completely unseen)
Purpose of testing function is for evaluating quantized network
Compare testing of quantized network with non-quantized network 
Need a way to save the parameters of the trained MPC network 
"""

convnet_shallow = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
    BatchNorm(),
    # PolyActivation(order=3, approx_type='chebyshev', function='relu', interval=(-7, 7)),
    PPoly(order=2),#, initialise='uniform'),
    # Relu(order=3, domain=(-1.5, 1.5)),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, 6272),  # 3136 5408 6272
    Reveal(),
    SoftmaxStable()
])

# convnet_deep = Sequential([
#     Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
#     BatchNorm(),
#     PolyActivation(order=3, approx_type='regression', function='relu'),
#     AveragePooling2D(pool_size=(2, 2)),
#     Conv2D((3, 3, 32, 16), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
#     BatchNorm(),
#     PolyActivation(order=3, approx_type='regression', function='relu'),
#     AveragePooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(10, 784),  # 3136 5408 6272
#     Reveal(),
#     SoftmaxStable()
# ])

# %%
"""
Train on different types of Tensor
"""
train_size = 96
val_size = 64
test_size = 96

x_train = x_train[:train_size]
y_train = y_train[:train_size]
x_val = x_test[:val_size]    # take more rows for use in gatherStats
y_val = y_test[:val_size]
x_test = x_test[val_size:(val_size+test_size)]
y_test = y_test[val_size:(val_size+test_size)]

tensortype = PrivateEncodedTensor
batch_size = 32
input_shape = [batch_size] + list(x_train.shape[1:])
epochs = 1
learning_rate = 0.01

convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_val, wrapper=tensortype),
    y_valid=DataLoader(y_val, wrapper=tensortype),
    x_test=DataLoader(x_test, wrapper=tensortype),
    y_test=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    learning_rate=learning_rate
)
end = time.time()
time_taken = end - start

print("Elapsed time: ", time_taken)
# print("Train accuracy:", accuracy(convnet_shallow, x_train, y_train))
# print("Test accuracy:", accuracy(convnet_shallow, x_test, y_test))

