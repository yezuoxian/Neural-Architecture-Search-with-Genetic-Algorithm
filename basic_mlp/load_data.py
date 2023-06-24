import os

import numpy as np


def load_mnist_2d(data_dir):
    # Load the MNIST dataset
    with np.load('basic_mlp/mnist.npz') as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    x_train = (x_train.reshape(-1, 28*28).astype('float32') - 128.0) / 255.
    x_test = (x_test.reshape(-1, 28*28).astype('float32') - 128.0)/ 255.

    return x_train, x_test, y_train, y_test
