import numpy as np
import math



# This file contains activation functions and their derivatives, learning rate functions, and general helper
# functions.

########### BEGIN LEARNING RATE FUNCTIONS #########


# Function for a static learning rate, used for convenience
def constant_learning_rate(network):
    return network.lrn_rate_modifier


# Defines a learning rate that decreases
# as we near the end of the dataset.
def neg_linear_learning_rate(modifier, iterations_complete, iterations_remaining):
    pass


LEARNING_FUNCTION_MAP = {
    "constant": constant_learning_rate
}

########### END LEARNING RATE FUNCTIONS #########

########### BEGIN ACTIVATION FUNCTIONS ##########


### Calculate tanh of x
def tanh(x):
    x = np.round(x, 5)
    return np.divide((np.exp(2 * x) - 1.0), (np.exp(2 * x) + 1.0))


### Calculate derivative of tanh of x
def tanh_prime(x):
    x = np.round(x, 5)
    return 1.0 - np.square(x)


### Calculate sigmoid of x
def sigmoid(x):
    return np.piecewise(x, [x > 0],
                        [lambda i: 1 / (1 + np.exp(-i)),
                         lambda i: np.exp(i) / (1 + np.exp(i))])


### Calculate sigmoid of x, different method
def sigmoid_two(x):
    result = []
    for num in x:
        if num >= 0:
            result.append(1. / (1. + np.exp(-num)))
        else:
            result.append(np.exp(num) / (1. + np.exp(num)))
    return result


### Calculate derivative of sigmoid of x
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)


### Calculate relu
def relu(self, x):
    return np.maximum(x, 0)


### Calculate derivative of relu
def relu_prime(x):
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
    return output.reshape(output.size, 1)


# calculate leaky relu
def leaky_relu(x):
    results = []
    for num in x:
        if num > 0:
            results.append(num)
        else:
            results.append(0.01 * num)
    return results


### Calculate derivative of leaky relu
def leaky_relu_prime(x):
    output = []
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
        else:
            output[i] = 0.01
    return output.reshape(output.size, 1)


def softmax(x):
    output = np.exp(x - x.max())
    return output / np.sum(output)


def softmax_prime(x):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# Map activations to functions
ACTIVATION_FUNCTION_MAP = {
    "leaky_relu": (leaky_relu, leaky_relu_prime),
    "relu": (relu, relu_prime),
    "sigmoid": (sigmoid_two, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
    "softmax": (softmax, softmax_prime)
}
