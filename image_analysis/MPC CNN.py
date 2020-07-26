import keras
import numpy as np
import time
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from pond.nn import Dense, ReluExact, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten, ConvAveragePooling2D, Square, BatchNorm
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
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Size of pooling area for max pooling
pool_size = (2, 2)
# Convolution kernel size (kernel_height, kernel_width, input_channels, num_filters)
#kernel_size = (5, 5, 1, 16)

convnet_shallow = Sequential([
    Conv2D((3, 3, 1, 16), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
    BatchNorm(),
    Relu(order=3),  # Conv Act Pool ?
    AveragePooling2D(pool_size=(2, 2)),
    #Square(),
    Flatten(),
    Dense(10, 3136),
    Reveal(),
    SoftmaxStable()
])

convnet = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1,
           filter_init=lambda shp: np.random.uniform(low=-0.14, high=0.14, size=shp)),
    AveragePooling2D(pool_size=pool_size),
    Relu(order=3),
    Conv2D((3, 3, 32, 32), strides=1, padding=1,
           filter_init=lambda shp: np.random.uniform(low=-0.1, high=0.1, size=shp)),
    AveragePooling2D(pool_size=pool_size),
    Relu(order=3),
    Flatten(),
    Dense(10, 1568),
    Reveal(),
    SoftmaxStable()
])


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
x_train = x_train[:256]
y_train = y_train[:256]
x_test = x_test[:256]
y_test = y_test[:256]

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
