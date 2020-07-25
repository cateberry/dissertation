import keras
import numpy as np
import time
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from pond.nn import Dense, ReluExact, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten, ConvAveragePooling2D, Dropout
from keras.utils import to_categorical

np.random.seed(42)

# set errors error behaviour for overflow/underflow
_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')

# %%
"""
Functions to load and pre-process data
- Load data from Keras
- Split into training and test data, where examples with target digit <5 are used for training and rest for testing
- Normalise to [0,1] (greyscale images with values in [0,255])
- Transform targets to one-hot representation
"""


def preprocess_data(dataset):
    (x_train, y_train), (x_test, y_test) = dataset

    # NOTE: this is the shape used by TensorFlow; other backends may differ
    # Reshape to (B, H, W, C)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalise to [0,1] (greyscale images)
    x_train /= 255
    x_test /= 255

    # One-hot encode targets
    y_train = to_categorical(y_train, 5)
    y_test = to_categorical(y_test, 5)

    return (x_train, y_train), (x_test, y_test)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Take examples with target digit <5 for public training data
    # i.e. non-MPC pre-training
    x_train_public = x_train[y_train < 5]
    y_train_public = y_train[y_train < 5]
    x_test_public = x_test[y_test < 5]
    y_test_public = y_test[y_test < 5]
    public_dataset = (x_train_public, y_train_public), (x_test_public, y_test_public)

    # Take remaining examples for private fine-tuning
    # i.e. MPC implementation
    x_train_private = x_train[y_train >= 5]
    y_train_private = y_train[y_train >= 5] - 5
    x_test_private = x_test[y_test >= 5]
    y_test_private = y_test[y_test >= 5] - 5
    private_dataset = (x_train_private, y_train_private), (x_test_private, y_test_private)

    return preprocess_data(public_dataset), preprocess_data(private_dataset)


# %%
"""
Public pre-training
- Train network on "public" data, i.e. implement in Keras with no encryption/MPC
- Uses equivalent layers to MPC CNN layers
    i.e. sigmoid instead of ReLU and average pooling instead of max pooling
- Separate into feature layers that will act as a trained feature extractor, and classification layers which
  will be fine-tuned with private data
"""

public_dataset, private_dataset = load_data()

# Settings for training
batch_size = 128
num_classes = 5
epochs = 5

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# Number of convolutional filters to use
filters = 32
# Size of pooling area for max pooling
pool_size = 2
# Convolution kernel size
kernel_size = 3

feature_layers = [
    keras.layers.Conv2D(filters, kernel_size, padding='same', input_shape=input_shape),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=pool_size),
    keras.layers.Conv2D(filters, kernel_size, padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=pool_size),
    #keras.layers.Dropout(.25),
    keras.layers.Flatten()
]

classification_layers = [
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    #keras.layers.Dropout(.50),
    keras.layers.Dense(5),
    keras.layers.Activation('softmax')
]

model = keras.models.Sequential(feature_layers + classification_layers)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = public_dataset
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Pre-train model on unencrypted public data
t1 = time.time()
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=batch_size,
    verbose=1,
    validation_data=(x_test, y_test))
t2 = time.time()
print('Training time: %s' % (t2 - t1))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# %%
"""
Take feature layers as a feature extractor and extract features from private data
"""

model.summary()

flatten_layer = model.get_layer(index=6)
assert flatten_layer.name.startswith('flatten')  # Check we're taking the correct layer

# Feature extractor (layers up to FC layers)
extractor = keras.models.Model(
    inputs=model.input,
    outputs=flatten_layer.output
)

(x_train_images, y_train), (x_test_images, y_test) = private_dataset

# Get features from private dataset (unencrypted)
x_train_features = extractor.predict(x_train_images)
x_test_features = extractor.predict(x_test_images)

# %%
# Save extracted features

np.save('x_train_features.npy', x_train_features)
np.save('y_train.npy', y_train)

np.save('x_test_features.npy', x_test_features)
np.save('y_test.npy', y_test)

# %%
# Load extracted features

x_train_features = np.load('x_train_features.npy')
y_train = np.load('y_train.npy')

x_test_features = np.load('x_test_features.npy')
y_test = np.load('y_test.npy')

print(x_train_features.shape, y_train.shape, x_test_features.shape, y_test.shape)

# %%
"""
Fine-tune the network using MPC-defined operations
- Output of final dense layer is revealed to one of the servers who computes the softmax layer to get class 
    probabilities - privacy leakage
"""
classifier = Sequential([
    Dense(128, 1568),
    Relu(order=3),
    Dense(5, 128),
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
Fine-tune using different types of Tensor
"""
# NativeTensor (no encryption or MPC)
tensortype = NativeTensor

input_shape = x_train_features.shape
epochs = 3
learning_rate = 0.01

classifier.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
classifier.fit(
    x_train=DataLoader(x_train_features, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test_features, wrapper=tensortype),
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
print("Train accuracy:", accuracy(classifier, x_train_features, y_train))
print("Test accuracy:",  accuracy(classifier, x_test_features,  y_test))

#%%
# PublicEncodedTensor (MPC operations on public values, i.e. unencrypted)
tensortype = PublicEncodedTensor
epochs = 3
learning_rate = 0.01

# Can train on subset of batches due to long training times
# x_train = x_train[:256]
# y_train = y_train[:256]
# x_test = x_test[:256]
# y_test = y_test[:256]

classifier.initialize(initializer=tensortype, input_shape=input_shape)

start = time.time()
classifier.fit(
    x_train=DataLoader(x_train_features, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test_features, wrapper=tensortype),
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
print("Train accuracy:", accuracy(classifier, x_train_features, y_train))
print("Test accuracy:",  accuracy(classifier, x_test_features,  y_test))

#%%
# PrivateEncodedTensor (full MPC)
tensortype = PrivateEncodedTensor
epochs = 3
learning_rate = 0.01

# x_train = x_train[:256]
# y_train = y_train[:256]
# x_test = x_test[:256]
# y_test = y_test[:256]

classifier.initialize(input_shape=input_shape, initializer=tensortype)

start = time.time()
classifier.fit(
    x_train=DataLoader(x_train_features, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test_features, wrapper=tensortype),
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
print("Train accuracy:", accuracy(classifier, x_train_features, y_train))
print("Test accuracy:",  accuracy(classifier, x_test_features,  y_test))
