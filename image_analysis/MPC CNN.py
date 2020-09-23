import keras
import numpy as np
import time
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from pond.nn import Dense, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten, BatchNorm, ReluNormal, Softmax, Sigmoid, PPoly
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

x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
# x_train = x_train[:, np.newaxis, :, :]
# x_test = x_test[:, np.newaxis, :, :]
# x_train = x_train.astype(np.uint8)  # TODO: test this?
# x_test = x_test.astype(np.uint8)
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

# Size of pooling area for max pooling
# pool_size = (2, 2)
# Convolution kernel size (kernel_height, kernel_width, input_channels, num_filters)
# kernel_size = (5, 5, 1, 16)

# convnet_shallow_gal = Sequential([
#     #Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
#     Conv2DQuant((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
#     BatchNorm(),
#     ReluGalois(order=4, mu=0.0, sigma=1.0),
#     AveragePooling2D(pool_size=(2, 2)),
#     Flatten(),
#     # Dense(10, 6272),  # 3136 5408 6272
#     DenseQuant(10, 6272),
#     Reveal(),
#     SoftmaxStable()
# ])

convnet_shallow = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
    BatchNorm(),
    # ReluNormal(order=3, approx_type='chebyshev', function='relu'),
               # , approx_type='lagrange', function='softplus', method='least-squares', point_dist='chebyshev', omega=[-3, 3]),
    # Relu(order=3),
    PPoly(order=2),
    # Sigmoid(order=3),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, 6272),  # 3136 5408 6272
    Reveal(),
    SoftmaxStable()
])

# convnet = Sequential([
#     Conv2D((5, 5, 1, 1), strides=1, padding=1,
#            filter_init=lambda shp: np.random.uniform(low=-0.14, high=0.14, size=shp)),
#     # BatchNormTest(),
#     # ReluNormal(order=3),
#     AveragePooling2D(pool_size=(2,2)),
#     Conv2D((5, 5, 1, 20), strides=1, padding=1,
#            filter_init=lambda shp: np.random.uniform(low=-0.1, high=0.1, size=shp)),
#     # BatchNormTest(),
#     # ReluNormal(order=3),
#     AveragePooling2D(pool_size=(2,2)),
#     Flatten(),
#     Dense(500, 500),
#     BatchNormTest(),
#     ReluNormal(order=3),
#     Dense(10, 500),
#     Reveal(),
#     SoftmaxStable()
# ])


# %%

def accuracy(classifier, x, y, verbose=0, wrapper=NativeTensor):
    predicted_classes = classifier \
        .predict(DataLoader(x, wrapper), verbose=verbose).reveal() \
        .argmax(axis=1)

    correct_classes = NativeTensor(y) \
        .argmax(axis=1)

    matches = predicted_classes.unwrap() == correct_classes.unwrap()
    return sum(matches) / len(matches)


# %%
"""
Train on different types of Tensor
"""
# NativeTensor (like plaintext)
train_size = 512
val_size = 128
test_size = 128

# TODO: shuffle datasets?
x_train = x_train[:train_size]
y_train = y_train[:train_size]
x_val = x_test[:val_size]    # take more rows for use in gatherStats
y_val = y_test[:val_size]
x_test = x_test[val_size:(val_size+test_size)]
y_test = y_test[val_size:(val_size+test_size)]

tensortype = PrivateEncodedTensor  # TODO: Change back to NativeTensor
batch_size = 32
input_shape = [batch_size] + list(x_train.shape[1:])
epochs = 3
learning_rate = 0.01

convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_val, wrapper=tensortype),
    y_valid=DataLoader(y_val, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    learning_rate=learning_rate
)
end = time.time()
time_taken = end - start

print("Elapsed time: ", time_taken)
# print("Train accuracy:", accuracy(convnet, x_train, y_train))
print("Test accuracy:", accuracy(convnet_shallow, x_test, y_test))

# %%
# PublicEncodedTensor (MPC operations on public values, i.e. unencrypted)
# Can train on subset of batches due to long training times
# x_train = x_train[:256]
# y_train = y_train[:256]
# x_test = x_test[:256]
# y_test = y_test[:256]
raise Exception()
tensortype = PublicEncodedTensor
epochs = 1
learning_rate = 0.01

convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    learning_rate=learning_rate
)
end = time.time()
time_taken = end - start

print("Elapsed time: ", time_taken)
# print("Train accuracy:", accuracy(convnet, x_train, y_train))
# print("Test accuracy:", accuracy(convnet, x_test, y_test))

# %%
# PrivateEncodedTensor (full MPC)
x_train = x_train[:256]
y_train = y_train[:256]
x_test = x_test[:256]
y_test = y_test[:256]

tensortype = PrivateEncodedTensor
epochs = 1
learning_rate = 0.01

convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    learning_rate=learning_rate
)
end = time.time()
time_taken = end - start

print("Elapsed time: ", time_taken)
# print("Train accuracy:", accuracy(convnet, x_train, y_train))
# print("Test accuracy:", accuracy(convnet, x_test, y_test))
