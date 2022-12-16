import numpy as np
# We are using Numba to improve the performance of some mathematical functions
from numba import jit, float32


# This file contains activation functions and their derivatives, learning rate functions, and general helper
# functions.

########### BEGIN LEARNING RATE FUNCTIONS #########

# Function for a static learning rate, used for convenience.

def constant_learning_rate(network):
    return network.lrn_rate_modifier

# Learn rate decreases linearly as batch accuracy decreases.
def inv_batch_accuracy_lrn_rate(network):
    return (1 - network.last_batch_accuracy) * network.lrn_rate_modifier


LEARNING_FUNCTION_MAP = {
    "constant": constant_learning_rate,
    "inverse_batch_accuracy": inv_batch_accuracy_lrn_rate
}

########### END LEARNING RATE FUNCTIONS #########

########### BEGIN ACTIVATION FUNCTIONS ##########

# No activation function, just returns the original input.
def no_activation(x: np.ndarray):
    return x

# Derivative of no activation function is zero, since it returns a constant.
def no_activation_prime(x: np.ndarray):
    return np.zeros(x.shape)

### Calculate tanh of x
def tanh(x: np.ndarray):
    x = np.round(x, 5)
    return np.divide((np.exp(2 * x) - 1.0), (np.exp(2 * x) + 1.0))


### Calculate derivative of tanh of x
def tanh_prime(x: np.ndarray):
    x = np.round(x, 5)
    return 1.0 - np.square(x)


### Calculate sigmoid of x
def sigmoid(x: np.ndarray):
    return np.piecewise(x, [x > 0],
                        [lambda i: 1 / (1 + np.exp(-i)),
                         lambda i: np.exp(i) / (1 + np.exp(i))])


### Calculate sigmoid of x, different method
def sigmoid_two(x: np.ndarray):
    result = [float32]
    for num in x:
        if num >= 0:
            result.append(1. / (1. + np.exp(-num)))
        else:
            result.append(np.exp(num) / (1. + np.exp(num)))
    return result


### Calculate derivative of sigmoid of x
def sigmoid_prime(x: np.ndarray):
    sig = sigmoid(x)
    return sig * (1.0 - sig)


### Calculate relu
def relu(x: np.ndarray):
    return np.maximum(x, 0)


### Calculate derivative of relu
@jit(nopython=True)
def relu_prime(x: np.ndarray):
    result = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            result[i] = 1
    return result


# calculate leaky relu
@jit(nopython=True)
def leaky_relu(x: np.ndarray):
    results = []
    for num in x:
        if num > 0:
            results.append(num)
        else:
            results.append(0.01 * num)
    return np.asarray(results)


### Calculate derivative of leaky relu
@jit(nopython=True)
def leaky_relu_prime(x: np.ndarray):
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
        else:
            output[i] = 0.01
    return output

@jit(nopython=True, parallel=True)
def softmax(x):
    output = np.exp(x - x.max())
    return output / np.sum(output)

@jit
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
    "softmax": (softmax, softmax_prime),
    "None": (no_activation, no_activation_prime)
}
