import numpy as np

from Math import Functions


# Represents a layer of a neural network
from Layers.Layer import Layer

class DenseLayer(Layer):
    def __init__(
            self,
            output_size,
            bias_lrn_modifier=0.0,
            dropout_modifier=0.0,
            activation_string: str = "None",
            previous_layer=None,
            input_size=None,):

        # If we're given an input size, set it and create weights.
        # Otherwise, this must be done before running the network.

        self.weights = None
        if input_size is not None:
            self.set_input_size(input_size)

        # Set the input and output sizes, dropout rate, and activation function
        # Also initializes a pretty printer as pprint
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dropout_modifier=dropout_modifier,
            activation_string=activation_string,
            previous_layer=previous_layer)


        # The bias learning modifier, applied on top of the
        # regular learning rate. I.E: Setting to 0.1 will cause the learning rate to
        # be 1/10th of the regular learning rate.
        # Set to 0 to disable bias.
        if bias_lrn_modifier is None:
            self.bias_lrn_modifier = 0.0
        else:
            self.bias_lrn_modifier = bias_lrn_modifier

        # Stores the per-neuron bias
        self.bias = np.zeros(output_size, dtype=np.float32)


    def set_input_size(self, input_size):
        # Flatten multi-dimensional input sizes
        if type(input_size) is tuple:
            size = 1
            for i in range(len(input_size)):
                size = size * input_size[i]
            input_size = size

        self.input_size = input_size
        # Initialize weights to random normally distributed values between -1 and 1
        self.weights = np.random.default_rng().random((input_size, self.output_size))


    # Handles forward propagation when given input data
    # The size of data_in must match the input size of this layer
    def forward_prop(self, data_in):

        input_in = data_in.ravel()

        if len(input_in) != self.input_size:
            raise Exception("Invalid input size to layer.forward_prop.")

        self.last_input = input_in
        self.last_output = (np.matmul(input_in, self.weights) / self.input_size) + self.bias
        return self.last_output

    def get_output(self):
        return self.last_output

    # Performs backprop for the output layer. Takes in a list of expected results.
    def output_back_prop(self, expected, learn_rate):

        result = [0] * self.output_size
        result[np.argmax(self.last_output)] = 1

        # The results are the last output of this layer
        error = -(expected - self.last_output)
        prime = self.derivative(self.last_output)

        self.last_delta = error * prime
        self.back_prop(learn_rate=learn_rate)

    def hidden_back_prop(self, learn_rate):
        lower_layer = self.next_layer
        prime = self.derivative(self.last_output).ravel()
        self.last_delta = np.matmul(lower_layer.last_delta, lower_layer.weights.T) * prime
        self.back_prop(learn_rate=learn_rate)

    def back_prop(self, learn_rate):
        adjustedInput = self.last_input[np.newaxis]
        adjustedDelta = self.last_delta[np.newaxis]
        weight_updates = np.matmul(adjustedInput.T, adjustedDelta * learn_rate)
        bias_updates = self.last_delta * learn_rate * self.bias_lrn_modifier
        self.weights -= weight_updates
        self.bias -= bias_updates

    def get_layer_string(self):
        return f"\nLayer Input Size: {self.input_size}" \
               + f"\nLayer Output Size: {self.output_size}" \
               + f"\nLayer Activation: {self.activation.__name__}\n" \
               + f"\nBias Modifier: {self.bias_lrn_modifier}\n" \
               + f"\nDropout Rate: None\n"
