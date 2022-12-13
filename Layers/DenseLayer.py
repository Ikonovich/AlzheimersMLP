from random import random

import numba
import numpy as np
import numba as nb
from numba import int32, float32
from numba.experimental import jitclass

from Math import Functions


# Represents a layer of a neural network
from Layers.Layer import Layer

# Defines the specifications of the parameters Numba is expected to operate on.
spec = [
    ('input_size', int32),
    ('output_size', int32),
    ('bias_lrn_modifier', float32),
    ('dropout', float32),
    ('weights', float32[:]),
    ('activation', numba.typeof(Functions.sigmoid_two)),
    ('derivative', numba.typeof(Functions.sigmoid_prime)),
    ('last_input', float32[:]),
    ('last_output', float32[:]),
    ('last_delta', float32[:]),
    ('bias', float32[:]),
]

@jitclass(spec)
class DenseLayer:
    def __init__(
            self,
            activation,
            derivative,
            input_size,
            output_size,
            bias_lrn_modifier=0.0,
            dropout_modifier=0.0):

        # Set the dropout rate and activation function
        # Also initializes a pretty printer as pprint
        # super('Layer').__init__(activation, dropout_modifier)

        # The bias learning modifier, applied on top of the
        # regular learning rate. I.E: Setting to 0.1 will cause the learning rate to
        # be 1/10th of the regular learning rate.
        # Set to 0 to disable bias.
        if bias_lrn_modifier is None:
            self.bias_lrn_modifier = 0.0
        else:
            self.bias_lrn_modifier = bias_lrn_modifier

        # defines the drop-out rate
        self.dropout = dropout_modifier

        # Uses the passed in activation name to acquire the
        # activation function and its derivative from the map
        self.activation = activation  # Functions.ACTIVATION_FUNCTION_MAP[activation][0]
        self.derivative = derivative  # Functions.ACTIVATION_FUNCTION_MAP[activation][1]
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights to random normally distributed values between -1 and 1
        vals = []
        for i in range(input_size * output_size):
            vals.append(random.uniform(-1, 1))
        return np.asarray(vals)
        self.weights =
        # Stores the last provided input
        self.last_input = None
        # Stores the last calculated output
        self.last_output = None
        # Stores the last calculated delta
        self.last_delta = None

        # Stores the per-neuron bias
        self.bias = np.zeros(output_size, dtype=np.float32)

    # Handles forward propagation when given input data
    # The size of data_in must match the input size of this layer
    def forward_prop(self, data_in: float32[:]):

        input_in = data_in

        if len(input_in) != self.input_size:
            raise Exception("Invalid input size to layer.forward_prop.")

        self.last_input = input_in
        self.last_output = (np.dot(input_in, self.weights) / self.input_size) + self.bias
        return self.last_output

    def get_output(self):
        return self.last_output

    # Performs backprop for the output layer. Takes in a list of expected results.
    def output_back_prop(self, expected: float32[:], learn_rate: float32):

        result = [0] * self.output_size
        result[np.argmax(self.last_output)] = 1

        # The results are the last output of this layer
        error = -(expected - self.last_output)
        prime = self.derivative(self.last_output)  # .flatten()

        self.last_delta = error * prime
        self.back_prop(learn_rate=learn_rate)

    def hidden_back_prop(self, learn_rate: float32):
        lower_layer = self.next_layer
        prime = self.derivative(self.last_output).flatten()
        self.last_delta = np.dot(lower_layer.last_delta, lower_layer.weights.T) * prime
        self.back_prop(learn_rate=learn_rate)

    def back_prop(self, learn_rate: float32):
        adjustedInput = self.last_input[np.newaxis]
        adjustedDelta = self.last_delta[np.newaxis]
        weight_updates = np.dot(adjustedInput.T, adjustedDelta * learn_rate)
        bias_updates = self.last_delta * learn_rate * self.bias_lrn_modifier
        self.weights -= weight_updates
        self.bias -= bias_updates

    def get_layer_string(self):
        return f"\nLayer Input Size: {self.input_size}" \
               + f"\nLayer Output Size: {self.output_size}" \
               + f"\nLayer Activation: {self.activation_string}\n" \
               + f"\nBias Modifier: {self.bias_lrn_modifier}\n" \
               + f"\nDropout Rate: None\n"
