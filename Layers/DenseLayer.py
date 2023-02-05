import numpy as np

from Math import Functions


# Represents a layer of a neural network
from Layers.Layer import Layer

class DenseLayer(Layer):
    def __init__(
            self,
            output_shape,
            bias_lrn_modifier=0.0,
            dropout_modifier=0.0,
            activation_string: str = "None",
            previous_layer=None,
            input_shape=None):

        # Stores the input size for normalization
        self.input_size = None

        # Set the input and output sizes, dropout rate, and activation function
        # Also initializes a pretty printer as pprint
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
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
        self.bias = np.zeros(output_shape, dtype=np.float32)





    # Handles forward propagation when given input data
    # The size of data_in must match the input size of this layer
    def forward_prop(self, data_in):

        input_in = data_in.ravel()

        self.last_input = input_in
        self.output = (np.matmul(input_in, self.weights) / self.input_size) + self.bias
        return self.output

    def get_output(self):
        return self.output

    # Performs backprop for the output layer. Takes in a list of expected results.
    def output_back_prop(self, expected, learn_rate):
        # The results are the last output of this layer
        error = -(expected - self.output)
        prime = self.derivative(self.output)

        self.loss = error * prime
        self.back_prop(learn_rate=learn_rate)

    def hidden_back_prop(self, learn_rate):
        lower_layer = self.next_layer
        prime = self.derivative(self.output).ravel()
        self.loss = np.matmul(lower_layer.loss, lower_layer.weights.T) * prime
        self.back_prop(learn_rate=learn_rate)

    def back_prop(self, learn_rate):
        adjustedInput = self.last_input[np.newaxis]
        adjustedDelta = self.loss[np.newaxis]
        weight_updates = np.matmul(adjustedInput.T, adjustedDelta * learn_rate)
        bias_updates = self.loss * learn_rate * self.bias_lrn_modifier
        self.weights -= weight_updates
        self.bias -= bias_updates

    def get_layer_string(self):
        return f"\nLayer Input Size: {self.input_shape}" \
               + f"\nLayer Output Size: {self.output_shape}" \
               + f"\nLayer Activation: {self.activation.__name__}\n" \
               + f"\nBias Modifier: {self.bias_lrn_modifier}\n" \
               + f"\nDropout Rate: None\n"

    # Set the input shape and initialize the weights
    # Set the input shape and initialize the weights
    def set_input_shape(self, input_shape: tuple[int, int] | int) -> None:
        self.input_shape = input_shape

        # Combine the shapes of the input and output to get the shape
        # of the weight matrix.
        dim_list = list()
        if type(self.input_shape) is int:
            dim_list.append(self.input_shape)
            self.input_size = self.input_shape
        else:
            dim = 1
            for dimension in self.input_shape:
                dim = dim * dimension
            dim_list.append(dim)
            self.input_size = dim
        if type(self.output_shape) is int:
            dim_list.append(self.output_shape)
        else:
            dim = 1
            for dimension in self.output_shape:
                dim = dim * dimension
            dim_list.append(dim)

        # Generate the weights
        self.weights = np.random.default_rng().random(tuple(dim_list))