import numpy as np
import math
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
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_two(numIn):
    results = []
    for i in numIn:
        if i == 0:
            results.append(0)
        else:
            results.append((math.exp(2 * i) - 1) / (math.exp(2 * i) + 1))
    return np.asarray(results)


### Calculate derivative of sigmoid of x
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)


### Calculate relu
def relu(self, x):
    return np.maximum(x, 0)


### Calculate derivative of relu
def relu_prime(x):
    output = []
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