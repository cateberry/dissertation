#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
import numpy as np
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from pond.nn import Dense, ReluExact, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    Softmax, AveragePooling2D, Flatten, ConvAveragePooling2D, Square
from keras.utils import to_categorical
np.random.seed(42)

# set errors error behaviour for overflow/underflow
_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')


# ## Read data

# In[3]:


# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:,np.newaxis,:,:] / 255.0
x_test = x_test[:,np.newaxis,:,:] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

batch_size = 128
input_shape = [batch_size] + list(x_train.shape[1:])


# ## Define 2 convnets
# We define 2 convnets, one with a single convlayer and one with 2 convlayers. Both can be used, but a single layer convnet is already quite slow. You can also use the ConvAveragePooling2D layer instead of the separated Conv2D and AveragePooling2D layers, to reduce comunication. The Relu layer we use here is an approximation of the regular ReLU, which can be called by replacing ReLU(order=3) with ReluExact().

# In[4]:


convnet_shallow = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
    Relu(order=3),
    #Square(),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, 6272),
    Reveal(),
    SoftmaxStable()
])


convnet_deep = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.uniform(low=-0.14, high=0.14, size=shp)),
    AveragePooling2D(pool_size=(2, 2)),
    Relu(),
    Conv2D((3, 3, 32, 32), strides=1, padding=1, filter_init=lambda shp: np.random.uniform(low=-0.1, high=0.1, size=shp)),
    AveragePooling2D(pool_size=(2, 2)),
    Relu(),
    Flatten(),
    Dense(10, 1568),
    Reveal(),
    SoftmaxStable()
])


# In[5]:


tensortype = NativeTensor
convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=1,
    batch_size=batch_size,
    verbose=1,
    learning_rate=0.01
)


# Took about 3 mins

# ## PublicEncodedTensor
# Train the same network on PublicEncodedTensor, this network **does not** use SPDZ, but works on the 128 bit integer encoding of real numbers which is neccesary for SPDZ, but slows down 100x. (however there are fixes for this) Therefore, we only train on the first two batches.

# In[6]:


# train on small set
# x_train = x_train[:25600]
# y_train = y_train[:25600]
# x_test = x_test[:25600]
# y_test = y_test[:25600]
#
# tensortype = PublicEncodedTensor
# convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)
# convnet_shallow.fit(
#     x_train=DataLoader(x_train, wrapper=tensortype),
#     y_train=DataLoader(y_train, wrapper=tensortype),
#     x_valid=DataLoader(x_test, wrapper=tensortype),
#     y_valid=DataLoader(y_test, wrapper=tensortype),
#     loss=CrossEntropy(),
#     epochs=1,
#     batch_size=batch_size,
#     verbose=1,
#     learning_rate=0.01
# )


# I know, the progressbar has a bug...

# ## PrivateEncodedTensor
# Train the same network on PrivateEncodedTensor, this network **does** use SPDZ, and is therefore even slower. Here, we also train on the first two batches. However, if you have time you can run the network for a full epoch and reach ~86% accuracy

# In[6]:


# train on small set
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:10000]
y_test = y_test[:10000]

tensortype = PrivateEncodedTensor
convnet_shallow.initialize(initializer=tensortype, input_shape=input_shape)
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=1,
    batch_size=batch_size,
    verbose=1,
    learning_rate=0.01
)


# This simple network is already very slow. Therefore, [this](http://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/#basics) is not a bad idea.
