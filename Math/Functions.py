import numpy as np
import torch


# This file contains activation functions and their derivatives, learning rate functions, and general helper
# functions.

####### BEGIN LOSS FUNCTIONS #######


def squared_error(expected, actual):
    result = 0
    for index in range(len(expected)):
        minuend = expected[index]
        subtrahend = actual[index]
        result += torch.sum(torch.square(minuend - subtrahend))
    return result


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
    result = []
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


# calculate leaky relu
def leaky_relu(x: np.ndarray):
    results = []
    for num in x:
        if num > 0:
            results.append(num)
        else:
            results.append(0.01 * num)
    return np.asarray(results)


### Calculate derivative of leaky relu
def leaky_relu_prime(x: np.ndarray):
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
        else:
            output[i] = 0.01
    return output

def softmax(x):
    output = np.exp(x - x.max())
    return output / np.sum(output)

def softmax_prime(x):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)
